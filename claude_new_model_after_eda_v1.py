# ============================================================
# WiDS 2025 Wildfire — V6 (Full Rewrite)
# ============================================================
# Changes from V5:
#   FIX-1:  dist_risk_close threshold corrected 3000m → 5000m (was missing 13 hit fires)
#   FIX-2:  Dropped 3 dead features (fast_hit_signal, low_res_x_urgency, low_res_x_growth)
#   FIX-3:  Dropped 11 report-flagged junk/duplicate features
#   FIX-4:  LightGBM now uses XGBoost-AFT instead (LGB regression was wrong objective)
#   FIX-5:  Percentile features computed from train, applied consistently to test
#   FIX-6:  Added BRIER_WEIGHTS[12] — 12h was unconstrained in V5
#   FIX-7:  Monotonicity enforced with vectorized cummax (no loop clipping bug)
#   FIX-8:  Stacking meta-model risk target fixed (was logically inverted for censored)
#   FIX-9:  KM prior re-enabled and scoped correctly (72h was systematically low)
#   ADD-1:  log_num_perimeters (corr=0.405 — was missing entirely)
#   ADD-2:  bearing_is_real flag (corr=0.314 — fixes spread_bearing_cos artifact)
#   ADD-3:  low_res_x_log_dist, low_res_x_area (replaces dead interactions)
#   ADD-4:  close_and_closing, dist_x_alignment (new meaningful interactions)
#   ARCH-1: Two-segment evaluation (close vs far) for calibration diagnostics
#   ARCH-2: KM prior blend scoped to 72h only (where bias is largest)
# ============================================================

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, rankdata

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter

# Optional W&B — set False to skip
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ============================================================
# 0.  CONFIGURATION
# ============================================================
DATA_DIR  = r"d:\wids"          # ← change to your path
USE_WANDB = False               # set True if wandb is installed

OPTUNA_TRIALS = 20          # 50→20 for practical runtime (~15-20min)
N_SEEDS       = 2           # 3→2 seeds (good coverage, half the time)
SEED_LIST     = [42, 123]
N_FOLDS       = 5
N_REPEATS     = 1           # 3→1 repeats; use seeds for variance reduction instead

# Horizons — all four are real evaluation targets
HORIZONS = [12, 24, 48, 72]

# Weighted Brier score weights — 12h now included (FIX-6)
BRIER_WEIGHTS = {12: 0.15, 24: 0.25, 48: 0.35, 72: 0.25}

# KM prior blending: only correct 72h (most biased horizon — FIX-9)
# Increased 72h weight to 0.40 — v4 experiments showed strong 72h anchor
# is the #1 driver of leaderboard score (v4=0.9599, v6_final=0.9577 at 0.15).
KM_BLEND = {12: 0.0, 24: 0.0, 48: 0.0, 72: 0.40}

# Distance threshold — verified from data: max hit distance = 4674m (FIX-1)
CLOSE_THRESHOLD_M = 5000

CONFIG = {
    "version": "v6",
    "n_folds": N_FOLDS,
    "n_repeats": N_REPEATS,
    "optuna_trials": OPTUNA_TRIALS,
    "n_seeds": N_SEEDS,
    "close_threshold_m": CLOSE_THRESHOLD_M,
    "km_blend": KM_BLEND,
    "brier_weights": BRIER_WEIGHTS,
}

if USE_WANDB and _WANDB_AVAILABLE:
    wandb.init(
        project="WiDS2025-Wildfire",
        name=f"v6_{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
        config=CONFIG,
    )

print("=" * 70)
print("  WiDS 2025 WILDFIRE — V6 (All V5 Bugs Fixed)")
print("=" * 70)

# ============================================================
# 1.  LOAD DATA
# ============================================================
train_df   = pd.read_csv(os.path.join(DATA_DIR, "train_imputed_advanced.csv"))
test_df    = pd.read_csv(os.path.join(DATA_DIR, "test_imputed_advanced.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = ["event_id", "time_to_hit_hours", "event"]

print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")
print(f"  Hit rate: {train_df['event'].mean():.1%}  ({train_df['event'].sum()} hits)")

# ============================================================
# 2.  FEATURE ENGINEERING
# ============================================================

# Columns confirmed as exact duplicates or zero-signal — always excluded (FIX-3)
COLS_TO_DROP = {
    "relative_growth_0_5h",      # exact dup of area_growth_rel_0_5h
    "projected_advance_m",        # exact negative of dist_change_ci_0_5h
    "alignment_cos",              # strictly weaker than alignment_abs (0.12 vs 0.35)
    "spread_bearing_deg",         # raw angle — use sin/cos encoding only
    "along_track_speed",          # corr = 0.008
    "cross_track_component",      # corr = -0.058
    "dist_accel_m_per_h2",       # corr = -0.073, 84% zeros
    "event_start_hour",           # corr = 0.047 — too weak
    "event_start_dayofweek",      # corr = -0.119, no causal mechanism
    "log_area_ratio_0_5h",        # corr 0.81 with log1p_growth, strictly weaker
    "event_id",
}


def engineer_features(df: pd.DataFrame, ref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build all engineered features.

    Args:
        df:      DataFrame to transform (train or test).
        ref_df:  Reference DataFrame for percentile computation.
                 Pass training DataFrame when transforming test set.
                 Pass None (or omit) when transforming training set.
    Returns:
        DataFrame with new columns added.
    """
    df = df.copy()
    ref = ref_df if ref_df is not None else df

    # ── TIER 1: Always available (221/221) ───────────────────
    df["log_dist_min"]         = np.log1p(df["dist_min_ci_0_5h"])
    df["log_num_perimeters"]   = np.log1p(df["num_perimeters_0_5h"])   # ADD-1: corr=0.405
    df["is_close"]             = (df["dist_min_ci_0_5h"] < CLOSE_THRESHOLD_M).astype(int)  # FIX-1
    df["bearing_is_real"]      = (df["spread_bearing_cos"] != 1.0).astype(int)  # ADD-2: corr=0.314
    df["log1p_area_first"]     = np.log1p(df["area_first_ha"])
    df["is_summer"]            = df["event_start_month"].isin([6, 7, 8]).astype(int)
    df["is_night"]             = (
        (df["event_start_hour"] >= 20) | (df["event_start_hour"] <= 6)
    ).astype(int)

    # ── Low-res interactions — FIXED: only interact with always-available features ──
    # V5 had low_res_x_urgency and low_res_x_growth which were identically 0 (FIX-2)
    df["low_res_x_log_dist"]   = df["low_temporal_resolution_0_5h"] * df["log_dist_min"]   # ADD-3
    df["low_res_x_area"]       = df["low_temporal_resolution_0_5h"] * df["log1p_area_first"]  # ADD-3

    # ── TIER 2: Multi-perimeter fires (61/221) ───────────────
    df["has_growth"]           = (df["area_growth_abs_0_5h"] > 0).astype(int)
    df["urgency_ratio"]        = df["closing_speed_m_per_h"] / (df["dist_min_ci_0_5h"] + 1)
    df["log_urgency"]          = np.log1p(df["urgency_ratio"])
    df["directional_threat"]   = df["alignment_abs"] * df["closing_speed_abs_m_per_h"]
    df["dist_x_alignment"]     = df["log_dist_min"] * df["alignment_abs"]               # ADD-4

    # ── TIER 3: Moving fires (25/221) ────────────────────────
    df["wind_driven_index"]    = df["centroid_speed_m_per_h"] / (
        df["radial_growth_rate_m_per_h"] + 1
    )
    df["explosive_growth"]     = df["area_growth_rel_0_5h"] / (
        np.log1p(df["area_first_ha"]) + 0.1
    )
    df["night_growth_anomaly"] = df["is_night"] * df["area_growth_rate_ha_per_h"]
    df["fire_momentum"]        = df["area_growth_rate_ha_per_h"] * df["centroid_speed_m_per_h"]
    df["close_and_closing"]    = df["is_close"] * df["closing_speed_abs_m_per_h"]       # ADD-4
    df["confidence_urgency"]   = df["urgency_ratio"] * (1 - df["low_temporal_resolution_0_5h"])

    # ── Percentile features — computed from train, applied to test (FIX-5) ──
    # Only for always-available features with meaningful distributions
    for col in ["dist_min_ci_0_5h", "area_first_ha"]:
        ref_vals = ref[col].values
        if ref_df is not None:
            # Test set: percentile relative to training distribution
            df[f"{col}_pctile"] = df[col].apply(
                lambda x: percentileofscore(ref_vals, x, kind="rank") / 100.0
            )
        else:
            # Training set: self-rank
            df[f"{col}_pctile"] = df[col].rank(pct=True)
    # NOTE: closing_speed percentile removed — 88% zeros makes it uninformative

    return df


train_eng = engineer_features(train_df, ref_df=None)
test_eng  = engineer_features(test_df,  ref_df=train_df)   # FIX-5: pass train as ref

# Build final feature list — drop all flagged columns
feature_cols = [
    c for c in train_eng.columns
    if c not in TARGET_COLS
    and c not in COLS_TO_DROP
    and train_eng[c].dtype != object
]

# Safety check: no NaN-correlation (constant) features
# FIX: train_eng[c].corr(train_eng["event"]) returns a scalar — use pd.isna() directly
nan_corr_cols = [
    c for c in feature_cols
    if pd.isna(train_eng[c].corr(train_eng["event"]))
    or (train_eng[c].std() == 0)
]
if nan_corr_cols:
    print(f"  ⚠ Dropping {len(nan_corr_cols)} constant/NaN-corr features: {nan_corr_cols}")
    feature_cols = [c for c in feature_cols if c not in nan_corr_cols]

print(f"\n  Features: {len(feature_cols)}")

# ── Prepare arrays ────────────────────────────────────────────
X_train = train_eng[feature_cols].values.astype(np.float64)
X_test  = test_eng[feature_cols].values.astype(np.float64)
y_time  = train_eng["time_to_hit_hours"].values
y_event = train_eng["event"].values.astype(bool)
y_surv  = np.array(
    list(zip(y_event, y_time)),
    dtype=[("event", bool), ("time", float)]
)

scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train)
X_test_sc      = scaler.transform(X_test)

# ── Segment mask for diagnostics ─────────────────────────────
is_close_train = train_eng["dist_min_ci_0_5h"].values < CLOSE_THRESHOLD_M
is_close_test  = test_eng["dist_min_ci_0_5h"].values  < CLOSE_THRESHOLD_M
print(f"  Close fires (train): {is_close_train.sum()}  |  Far: {(~is_close_train).sum()}")
print(f"  Close fires (test):  {is_close_test.sum()}   |  Far: {(~is_close_test).sum()}")

# ============================================================
# 3.  KAPLAN-MEIER PRIORS  (FIX-9: used selectively, not disabled)
# ============================================================
print("\n─── Kaplan-Meier Priors ───")
kmf_all = KaplanMeierFitter()
kmf_all.fit(y_time, event_observed=y_event)

km_prior = {}
for h in HORIZONS:
    km_prior[h] = float(1.0 - kmf_all.predict(h))
    print(f"  KM P(hit ≤ {h:2d}h) = {km_prior[h]:.4f}  "
          f"[blend_weight={KM_BLEND[h]:.2f}]")

# ============================================================
# 4.  METRICS
# ============================================================

def brier_censor_aware(y_event, y_time, probs, h):
    """
    Censoring-aware Brier score at horizon h.
    Fires censored before h (and never hit) are excluded — their label is unknown.
    """
    mask   = np.ones(len(y_event), dtype=bool)
    labels = np.zeros(len(y_event))
    for i in range(len(y_event)):
        if y_event[i]:                          # observed hit
            labels[i] = float(y_time[i] <= h)
        else:                                   # censored
            if y_time[i] >= h:
                labels[i] = 0.0                # still alive at h — safe to include
            else:
                mask[i] = False                # censored before h — exclude
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((probs[mask] - labels[mask]) ** 2))


def weighted_brier(y_event, y_time, probs_dict):
    return sum(
        w * brier_censor_aware(y_event, y_time, probs_dict[h], h)
        for h, w in BRIER_WEIGHTS.items()
    )


def hybrid_score(y_event, y_time, risk, probs_dict):
    c_idx, *_ = concordance_index_censored(y_event, y_time, risk)
    wbs       = weighted_brier(y_event, y_time, probs_dict)
    return 0.3 * c_idx + 0.7 * (1.0 - wbs), c_idx, wbs


# ============================================================
# 5.  PROBABILITY HELPERS
# ============================================================

def survfn_to_probs(surv_fns, horizons):
    """Convert sksurv step-function objects to P(hit by h)."""
    probs = {h: np.zeros(len(surv_fns)) for h in horizons}
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.x[0], fn.x[-1]
        for h in horizons:
            t_clamped      = np.clip(h, t_min, t_max)
            probs[h][i]    = np.clip(1.0 - fn(t_clamped), 1e-6, 1 - 1e-6)
    return probs


def risk_to_probs_iso(risk_tr, e_tr, t_tr, risk_query, horizons):
    """
    Map risk scores to calibrated probabilities via isotonic regression.
    Fit on training fold (risk_tr), predict on query set (risk_query).
    """
    probs = {}
    for h in horizons:
        mask   = np.ones(len(e_tr), dtype=bool)
        labels = np.zeros(len(e_tr))
        for i in range(len(e_tr)):
            if e_tr[i]:
                labels[i] = float(t_tr[i] <= h)
            else:
                if t_tr[i] >= h:
                    labels[i] = 0.0
                else:
                    mask[i] = False
        if mask.sum() < 10:
            probs[h] = np.full(len(risk_query), km_prior[h])
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(risk_tr[mask], labels[mask])
        probs[h] = np.clip(iso.predict(risk_query), 1e-6, 1 - 1e-6)
    return probs


# ============================================================
# 6.  OPTUNA-TUNED MODEL FACTORIES
# ============================================================
print("\n" + "=" * 70)
print("  MODEL TRAINING  (GBSA · RSF · XGB-Cox · XGB-AFT)")
print("=" * 70)

MODEL_LIST = ["GBSA", "RSF", "XGB_COX", "XGB_AFT"]


def _run_optuna(objective_fn, n_trials, direction="maximize"):
    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── GBSA ─────────────────────────────────────────────────────
def train_gbsa(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        m = GradientBoostingSurvivalAnalysis(
            random_state=42,
            n_estimators      = trial.suggest_int("n_estimators", 50, 500),
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 2, 6),
            min_samples_split = trial.suggest_int("min_samples_split", 3, 50),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 2, 25),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        )
        m.fit(X_tr, y_tr)
        risk  = m.predict(X_val)
        probs = survfn_to_probs(m.predict_survival_function(X_val), HORIZONS)
        hs, *_ = hybrid_score(y_val["event"], y_val["time"], risk, probs)
        return hs

    best_params, _ = _run_optuna(objective, OPTUNA_TRIALS)
    model = GradientBoostingSurvivalAnalysis(random_state=42, **best_params)
    model.fit(X_tr, y_tr)
    return model, best_params


# ── RSF ──────────────────────────────────────────────────────
def train_rsf(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        m = RandomSurvivalForest(
            random_state=42, n_jobs=-1,
            n_estimators      = trial.suggest_int("n_estimators", 50, 500),
            max_depth         = trial.suggest_int("max_depth", 2, 12),
            min_samples_split = trial.suggest_int("min_samples_split", 3, 50),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 2, 30),
            max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        )
        m.fit(X_tr, y_tr)
        risk  = m.predict(X_val)
        probs = survfn_to_probs(m.predict_survival_function(X_val), HORIZONS)
        hs, *_ = hybrid_score(y_val["event"], y_val["time"], risk, probs)
        return hs

    best_params, _ = _run_optuna(objective, OPTUNA_TRIALS)
    model = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params)
    model.fit(X_tr, y_tr)
    return model, best_params


# ── XGBoost Cox ───────────────────────────────────────────────
def _xgb_labels_cox(y_surv):
    """Cox label encoding: positive for events, negative for censored."""
    return np.where(y_surv["event"], y_surv["time"], -y_surv["time"])


def train_xgb_cox(X_tr, y_tr, X_val, y_val):
    y_tr_xgb  = _xgb_labels_cox(y_tr)
    y_val_xgb = _xgb_labels_cox(y_val)

    def objective(trial):
        m = xgb.XGBRegressor(
            objective        = "survival:cox",
            eval_metric      = "cox-nloglik",
            verbosity        = 0,
            random_state     = 42,
            n_estimators     = trial.suggest_int("n_estimators", 50, 500),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 2, 8),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 15),
            gamma            = trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        )
        m.fit(X_tr, y_tr_xgb, eval_set=[(X_val, y_val_xgb)], verbose=False)
        risk_tr  = m.predict(X_tr)
        risk_val = m.predict(X_val)
        probs    = risk_to_probs_iso(risk_tr, y_tr["event"], y_tr["time"], risk_val, HORIZONS)
        hs, *_   = hybrid_score(y_val["event"], y_val["time"], risk_val, probs)
        return hs

    best_params, _ = _run_optuna(objective, OPTUNA_TRIALS)
    model = xgb.XGBRegressor(
        objective="survival:cox", eval_metric="cox-nloglik",
        verbosity=0, random_state=42, **best_params
    )
    model.fit(X_tr, y_tr_xgb, verbose=False)
    return model, best_params


# ── XGBoost AFT  (FIX-4: replaces LightGBM regression which was wrong) ──────
def train_xgb_aft(X_tr, y_tr, X_val, y_val):
    """
    XGBoost with Accelerated Failure Time loss.
    Unlike 'regression', AFT understands censoring natively.
    Labels must be (lower_bound, upper_bound):
      - Events:   (time, time)
      - Censored: (time, +inf)  → represented as (time, 1e9)
    """
    def make_aft_labels(y_surv):
        lower = y_surv["time"].copy()
        upper = np.where(y_surv["event"], y_surv["time"], 1e9)
        return lower, upper

    y_tr_lo,  y_tr_hi  = make_aft_labels(y_tr)
    y_val_lo, y_val_hi = make_aft_labels(y_val)

    dtrain = xgb.DMatrix(X_tr)
    dtrain.set_float_info("label_lower_bound", y_tr_lo)
    dtrain.set_float_info("label_upper_bound", y_tr_hi)

    dval = xgb.DMatrix(X_val)
    dval.set_float_info("label_lower_bound", y_val_lo)
    dval.set_float_info("label_upper_bound", y_val_hi)

    dtest_placeholder = xgb.DMatrix(X_tr)   # for risk extraction inside optuna

    def objective(trial):
        params = {
            "objective":                   "survival:aft",
            "eval_metric":                 "aft-nloglik",
            "aft_loss_distribution":       "normal",
            "aft_loss_distribution_scale": trial.suggest_float("aft_scale", 0.5, 2.0),
            "learning_rate":               trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth":                   trial.suggest_int("max_depth", 2, 8),
            "subsample":                   trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":            trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":            trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha":                   trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":                  trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbosity":                   0,
            "seed":                        42,
        }
        n_rounds = trial.suggest_int("n_estimators", 50, 500)
        booster  = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        # AFT predicts log(time) — negate for risk (lower predicted time = higher risk)
        risk_tr  = -booster.predict(dtrain)
        risk_val = -booster.predict(dval)
        probs    = risk_to_probs_iso(risk_tr, y_tr["event"], y_tr["time"], risk_val, HORIZONS)
        hs, *_   = hybrid_score(y_val["event"], y_val["time"], risk_val, probs)
        return hs

    best_params, _ = _run_optuna(objective, OPTUNA_TRIALS)

    # Refit on full training fold with best params
    n_rounds = best_params.pop("n_estimators", 200)
    aft_scale = best_params.pop("aft_scale", 1.0)
    final_params = dict(
        best_params,
        objective="survival:aft",
        eval_metric="aft-nloglik",
        aft_loss_distribution="normal",
        aft_loss_distribution_scale=aft_scale,
        verbosity=0, seed=42,
    )
    booster = xgb.train(
        final_params, dtrain,
        num_boost_round=n_rounds,
        verbose_eval=False,
    )
    return booster, best_params


def predict_xgb_aft(booster, X):
    """Return risk scores from AFT booster (higher = more dangerous)."""
    return -booster.predict(xgb.DMatrix(X))


# ============================================================
# 7.  SEED-AVERAGED CROSS-VALIDATION
# ============================================================
print(f"\n{'═'*70}")
print(f"  SEED-AVERAGED CV  ({N_SEEDS} seeds × {N_FOLDS}×{N_REPEATS} folds)")
print(f"{'═'*70}")

# Accumulators
seed_oof  = {m: {"risk": np.zeros(len(X_train)),
                  "probs": {h: np.zeros(len(X_train)) for h in HORIZONS}}
             for m in MODEL_LIST}
seed_test = {m: {"risk": np.zeros(len(X_test)),
                  "probs": {h: np.zeros(len(X_test)) for h in HORIZONS}}
             for m in MODEL_LIST}

for seed_idx, seed in enumerate(SEED_LIST):
    print(f"\n{'━'*70}")
    print(f"  SEED {seed_idx+1}/{N_SEEDS}  (seed={seed})")
    print(f"{'━'*70}")

    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=seed)

    oof   = {m: {"risk": np.zeros(len(X_train)),
                  "probs": {h: np.zeros(len(X_train)) for h in HORIZONS}}
             for m in MODEL_LIST}
    test_ = {m: {"risk": np.zeros(len(X_test)),
                  "probs": {h: np.zeros(len(X_test)) for h in HORIZONS}}
             for m in MODEL_LIST}
    oof_counts = np.zeros(len(X_train))
    total_folds = N_FOLDS * N_REPEATS

    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_train_sc, y_event), 1):
        print(f"\n  ── Seed {seed} │ Fold {fold_idx}/{total_folds}")

        X_tr,  X_val  = X_train_sc[tr_idx],  X_train_sc[val_idx]
        y_tr,  y_val  = y_surv[tr_idx],      y_surv[val_idx]
        e_tr,  t_tr   = y_tr["event"],        y_tr["time"]
        e_val, t_val  = y_val["event"],       y_val["time"]

        # ── 1/4  GBSA ────────────────────────────────────────
        print("    [1/4] GBSA ...", end=" ", flush=True)
        gbsa, _ = train_gbsa(X_tr, y_tr, X_val, y_val)
        r_val   = gbsa.predict(X_val)
        p_val   = survfn_to_probs(gbsa.predict_survival_function(X_val), HORIZONS)
        r_test  = gbsa.predict(X_test_sc)
        p_test  = survfn_to_probs(gbsa.predict_survival_function(X_test_sc), HORIZONS)
        hs, ci, _ = hybrid_score(e_val, t_val, r_val, p_val)
        print(f"H={hs:.4f}  C={ci:.4f}")
        oof["GBSA"]["risk"][val_idx]   += r_val
        test_["GBSA"]["risk"]          += r_test / total_folds
        for h in HORIZONS:
            oof["GBSA"]["probs"][h][val_idx]  += p_val[h]
            test_["GBSA"]["probs"][h]         += p_test[h] / total_folds

        # ── 2/4  RSF ─────────────────────────────────────────
        print("    [2/4] RSF  ...", end=" ", flush=True)
        rsf, _ = train_rsf(X_tr, y_tr, X_val, y_val)
        r_val  = rsf.predict(X_val)
        p_val  = survfn_to_probs(rsf.predict_survival_function(X_val), HORIZONS)
        r_test = rsf.predict(X_test_sc)
        p_test = survfn_to_probs(rsf.predict_survival_function(X_test_sc), HORIZONS)
        hs, ci, _ = hybrid_score(e_val, t_val, r_val, p_val)
        print(f"H={hs:.4f}  C={ci:.4f}")
        oof["RSF"]["risk"][val_idx]   += r_val
        test_["RSF"]["risk"]          += r_test / total_folds
        for h in HORIZONS:
            oof["RSF"]["probs"][h][val_idx]  += p_val[h]
            test_["RSF"]["probs"][h]         += p_test[h] / total_folds

        # ── 3/4  XGB-Cox ─────────────────────────────────────
        print("    [3/4] XGB-Cox ...", end=" ", flush=True)
        xgb_cox, _ = train_xgb_cox(X_tr, y_tr, X_val, y_val)
        y_tr_cox    = _xgb_labels_cox(y_tr)
        r_tr        = xgb_cox.predict(X_tr)
        r_val       = xgb_cox.predict(X_val)
        r_test      = xgb_cox.predict(X_test_sc)
        p_val       = risk_to_probs_iso(r_tr, e_tr, t_tr, r_val, HORIZONS)
        p_test      = risk_to_probs_iso(r_tr, e_tr, t_tr, r_test, HORIZONS)
        hs, ci, _   = hybrid_score(e_val, t_val, r_val, p_val)
        print(f"H={hs:.4f}  C={ci:.4f}")
        oof["XGB_COX"]["risk"][val_idx]   += r_val
        test_["XGB_COX"]["risk"]          += r_test / total_folds
        for h in HORIZONS:
            oof["XGB_COX"]["probs"][h][val_idx]  += p_val[h]
            test_["XGB_COX"]["probs"][h]         += p_test[h] / total_folds

        # ── 4/4  XGB-AFT  (FIX-4: replaces LGB regression) ──
        print("    [4/4] XGB-AFT ...", end=" ", flush=True)
        xgb_aft, _ = train_xgb_aft(X_tr, y_tr, X_val, y_val)
        r_tr        = predict_xgb_aft(xgb_aft, X_tr)
        r_val       = predict_xgb_aft(xgb_aft, X_val)
        r_test      = predict_xgb_aft(xgb_aft, X_test_sc)
        p_val       = risk_to_probs_iso(r_tr, e_tr, t_tr, r_val, HORIZONS)
        p_test      = risk_to_probs_iso(r_tr, e_tr, t_tr, r_test, HORIZONS)
        hs, ci, _   = hybrid_score(e_val, t_val, r_val, p_val)
        print(f"H={hs:.4f}  C={ci:.4f}")
        oof["XGB_AFT"]["risk"][val_idx]   += r_val
        test_["XGB_AFT"]["risk"]          += r_test / total_folds
        for h in HORIZONS:
            oof["XGB_AFT"]["probs"][h][val_idx]  += p_val[h]
            test_["XGB_AFT"]["probs"][h]         += p_test[h] / total_folds

        oof_counts[val_idx] += 1

    # Average repeated-fold OOF within this seed
    for m in MODEL_LIST:
        oof[m]["risk"] /= oof_counts
        for h in HORIZONS:
            oof[m]["probs"][h] /= oof_counts

    # Print per-seed OOF summary
    print(f"\n  ── Seed {seed} OOF Summary ──")
    for m in MODEL_LIST:
        hs, ci, wbs = hybrid_score(
            y_event, y_time, oof[m]["risk"],
            {h: oof[m]["probs"][h] for h in HORIZONS}
        )
        print(f"    {m:<10}  H={hs:.4f}  C={ci:.4f}  WBS={wbs:.4f}")

    # Accumulate across seeds
    for m in MODEL_LIST:
        seed_oof[m]["risk"]  += oof[m]["risk"]  / N_SEEDS
        seed_test[m]["risk"] += test_[m]["risk"] / N_SEEDS
        for h in HORIZONS:
            seed_oof[m]["probs"][h]  += oof[m]["probs"][h]  / N_SEEDS
            seed_test[m]["probs"][h] += test_[m]["probs"][h] / N_SEEDS

oof_preds  = seed_oof
test_preds = seed_test

# ── Save intermediate predictions ───────────────────────────
_inter = {
    "oof_preds": oof_preds, "test_preds": test_preds,
    "y_event": y_event, "y_time": y_time,
    "feature_cols": feature_cols, "model_list": MODEL_LIST,
}
_inter_path = os.path.join(DATA_DIR, "v6_intermediate.pkl")
with open(_inter_path, "wb") as f:
    pickle.dump(_inter, f)
print(f"\n  Intermediate predictions saved → {_inter_path}")

# ============================================================
# 8.  ENSEMBLE — BLENDING WITH GRID SEARCH
#     (safer than stacking at n=221)
# ============================================================
print("\n" + "=" * 70)
print("  ENSEMBLE: Blended Weighted Average")
print("=" * 70)

best_hs, best_w = -1.0, None
step = 0.1
for w0 in np.arange(0.0, 1.01, step):
    for w1 in np.arange(0.0, 1.01 - w0, step):
        for w2 in np.arange(0.0, 1.01 - w0 - w1, step):
            w3 = round(1.0 - w0 - w1 - w2, 6)
            if w3 < -1e-9:
                continue
            w = dict(zip(MODEL_LIST, [w0, w1, w2, w3]))
            blend_risk  = sum(wt * rankdata(oof_preds[m]["risk"]) for m, wt in w.items())
            blend_probs = {
                h: np.clip(
                    sum(wt * oof_preds[m]["probs"][h] for m, wt in w.items()),
                    1e-6, 1 - 1e-6
                )
                for h in HORIZONS
            }
            hs, *_ = hybrid_score(y_event, y_time, blend_risk, blend_probs)
            if hs > best_hs:
                best_hs, best_w = hs, w

print(f"  Best blend weights: {best_w}")
print(f"  Best blend OOF Hybrid: {best_hs:.4f}")

# Also try stacking and pick whichever wins
print("\n" + "─" * 50)
print("  STACKING META-MODEL  (Ridge, for comparison)")

def _build_meta(preds_dict, n):
    cols = {}
    for m in MODEL_LIST:
        cols[f"{m}_risk_rank"] = rankdata(preds_dict[m]["risk"])
        for h in HORIZONS:
            cols[f"{m}_prob_{h}h"] = preds_dict[m]["probs"][h]
    return pd.DataFrame(cols)

meta_tr  = _build_meta(oof_preds,  len(X_train))
meta_te  = _build_meta(test_preds, len(X_test))

# Risk meta-model — FIX-8: use rank-average of base models as target
# (V5 used 1/time which was logically inverted for censored fires)
risk_target = np.mean(
    [rankdata(oof_preds[m]["risk"]) for m in MODEL_LIST], axis=0
)
risk_meta = Ridge(alpha=10.0)
risk_meta.fit(meta_tr.values, risk_target)
stacked_risk = risk_meta.predict(meta_tr.values)
ci_stk, *_  = concordance_index_censored(y_event, y_time, stacked_risk)
print(f"  Stacked risk C-index (OOF): {ci_stk:.4f}")

# Per-horizon probability meta-models
meta_prob_models = {}
stacked_probs    = {}
for h in HORIZONS:
    mask   = np.ones(len(X_train), dtype=bool)
    labels = np.zeros(len(X_train))
    for i in range(len(X_train)):
        if y_event[i]:
            labels[i] = float(y_time[i] <= h)
        else:
            if y_time[i] >= h:
                labels[i] = 0.0
            else:
                mask[i] = False
    prob_cols = [f"{m}_prob_{h}h" for m in MODEL_LIST]
    rank_cols = [f"{m}_risk_rank" for m in MODEL_LIST]
    use_cols  = prob_cols + rank_cols
    meta_h    = Ridge(alpha=5.0)
    meta_h.fit(meta_tr.loc[mask, use_cols].values, labels[mask])
    meta_prob_models[h] = (meta_h, use_cols)
    pred_h              = np.clip(meta_h.predict(meta_tr[use_cols].values), 1e-6, 1 - 1e-6)
    stacked_probs[h]    = pred_h
    bs                  = brier_censor_aware(y_event, y_time, pred_h, h)
    print(f"    Horizon {h:2d}h │ Stacked Brier={bs:.4f}")

hs_stk, ci_stk, wbs_stk = hybrid_score(y_event, y_time, stacked_risk, stacked_probs)
print(f"  Stacked OOF Hybrid: {hs_stk:.4f}")

# Choose winner
if hs_stk >= best_hs:
    METHOD = "STACKING"
    BEST_H = hs_stk
    print(f"\n  → Using STACKING  (H={hs_stk:.4f} ≥ blend H={best_hs:.4f})")
else:
    METHOD = "BLENDING"
    BEST_H = best_hs
    print(f"\n  → Using BLENDING  (H={best_hs:.4f} > stack H={hs_stk:.4f})")

# ============================================================
# 9.  GENERATE TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("  GENERATING FINAL TEST PREDICTIONS")
print("=" * 70)

if METHOD == "STACKING":
    final_risk = risk_meta.predict(meta_te.values)
    final_probs = {
        h: np.clip(meta_prob_models[h][0].predict(meta_te[meta_prob_models[h][1]].values),
                   1e-6, 1 - 1e-6)
        for h in HORIZONS
    }
else:
    final_risk = sum(
        best_w[m] * rankdata(test_preds[m]["risk"]) for m in MODEL_LIST
    )
    final_probs = {
        h: np.clip(
            sum(best_w[m] * test_preds[m]["probs"][h] for m in MODEL_LIST),
            1e-6, 1 - 1e-6
        )
        for h in HORIZONS
    }

# ── KM prior blend — targeted at 72h only  (FIX-9) ──────────
print(f"\n  KM prior blending (scoped to horizons with weight > 0):")
for h in HORIZONS:
    w = KM_BLEND[h]
    if w > 0:
        raw      = final_probs[h].copy()
        blended  = (1.0 - w) * raw + w * km_prior[h]
        final_probs[h] = np.clip(blended, 1e-6, 1 - 1e-6)
        print(f"    {h:2d}h: [{raw.min():.4f}, {raw.max():.4f}]"
              f" → [{final_probs[h].min():.4f}, {final_probs[h].max():.4f}]"
              f"  (KM={km_prior[h]:.4f}, w={w:.2f})")
    else:
        print(f"    {h:2d}h: no blend  [{final_probs[h].min():.4f}, {final_probs[h].max():.4f}]")

# ============================================================
# 10.  BUILD SUBMISSION
# ============================================================
submission = pd.DataFrame({
    "event_id": test_df["event_id"],
    "prob_12h": final_probs[12],
    "prob_24h": final_probs[24],
    "prob_48h": final_probs[48],
    "prob_72h": final_probs[72],
})

# ── Monotonicity: vectorized cummax  (FIX-7: replaces buggy loop) ────
prob_cols = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
submission[prob_cols] = submission[prob_cols].cummax(axis=1).clip(0.001, 0.999)

# ============================================================
# 11.  CALIBRATION DIAGNOSTICS  (two-segment + overall)
# ============================================================
print("\n" + "=" * 70)
print("  CALIBRATION DIAGNOSTICS")
print("=" * 70)

# OOF calibration checks
oof_probs = {
    h: np.clip(
        sum((best_w if METHOD == "BLENDING" else {m: 1/len(MODEL_LIST) for m in MODEL_LIST})[m]
            * oof_preds[m]["probs"][h] for m in MODEL_LIST),
        1e-6, 1 - 1e-6
    )
    for h in HORIZONS
}

print("\n  OOF predictions — overall vs KM baseline:")
print(f"  {'Horizon':<10} {'OOF mean':>10} {'KM prior':>10} {'Naive rate':>12}")
naive_rates = {12: 0.222, 24: 0.285, 48: 0.299, 72: 0.312}
for h in HORIZONS:
    print(f"  {h:2d}h        {oof_probs[h].mean():>10.4f} {km_prior[h]:>10.4f} {naive_rates[h]:>12.4f}")

print("\n  OOF predictions — close vs far fire segments:")
print(f"  {'Horizon':<10} {'Close mean':>12} {'Far mean':>10} {'Expected close':>16}")
expected_close = {12: 0.710, 24: 0.913, 48: 0.957, 72: 1.000}
for h in HORIZONS:
    p_close = oof_probs[h][is_close_train].mean()
    p_far   = oof_probs[h][~is_close_train].mean()
    print(f"  {h:2d}h        {p_close:>12.4f} {p_far:>10.4f} {expected_close[h]:>16.4f}")

# Test set sanity
print("\n  Test set prediction ranges:")
for col in prob_cols:
    v = submission[col]
    print(f"    {col}: [{v.min():.4f}, {v.max():.4f}]  mean={v.mean():.4f}  "
          f"std={v.std():.4f}  unique={v.nunique()}")

# ============================================================
# 12.  SANITY CHECKS
# ============================================================
print("\n" + "=" * 70)
print("  SANITY CHECKS")
print("=" * 70)

passed = True

# 1. Monotonicity
mono_ok = (
    (submission["prob_12h"] <= submission["prob_24h"]).all() and
    (submission["prob_24h"] <= submission["prob_48h"]).all() and
    (submission["prob_48h"] <= submission["prob_72h"]).all()
)
status = "✓" if mono_ok else "✗ FAIL"
print(f"  {status} Monotonicity (12h≤24h≤48h≤72h)")
if not mono_ok:
    passed = False

# 2. Bounds
for col in prob_cols:
    in_bounds = (submission[col] > 0) & (submission[col] < 1)
    status = "✓" if in_bounds.all() else "✗ FAIL"
    print(f"  {status} {col} in (0,1): {in_bounds.sum()}/{len(submission)}")
    if not in_bounds.all():
        passed = False

# 3. Variance (not collapsed)
for col in prob_cols:
    ok = submission[col].std() > 0.01
    status = "✓" if ok else "✗ FAIL — predictions collapsed!"
    print(f"  {status} {col} std={submission[col].std():.4f}")
    if not ok:
        passed = False

# 4. Row count
row_ok = len(submission) == len(sample_sub)
status = "✓" if row_ok else "✗ FAIL"
print(f"  {status} Row count: {len(submission)} (expected {len(sample_sub)})")
if not row_ok:
    passed = False

# 5. 72h calibration check — should not be near naive 0.31
p72_mean = submission["prob_72h"].mean()
if p72_mean < 0.35:
    print(f"  ⚠ WARNING: prob_72h mean={p72_mean:.4f} — may be underestimated")
    print(f"    KM suggests true rate ~0.55 (accounting for censoring)")
else:
    print(f"  ✓ prob_72h mean={p72_mean:.4f} (KM baseline 0.55)")

print(f"\n  → {'✅ ALL CHECKS PASSED' if passed else '❌ REVIEW FAILURES ABOVE'}")

# ============================================================
# 13.  SAVE SUBMISSION
# ============================================================
sub_path = os.path.join(DATA_DIR, "submission_claude_v1_imputed_data.csv")
submission.to_csv(sub_path, index=False)
print(f"\n  Submission saved → {sub_path}")
print(f"\n{submission.describe().to_string()}")

# ============================================================
# 14.  FINAL SUMMARY
# ============================================================
print(f"\n{'═'*60}")
print(f"  V6 FINAL RESULTS")
print(f"{'═'*60}")
print(f"  Ensemble method : {METHOD}")
print(f"  OOF Hybrid      : {BEST_H:.4f}")
print(f"  Seeds averaged  : {N_SEEDS}")
print(f"  Optuna trials   : {OPTUNA_TRIALS}/model")
print(f"  Features        : {len(feature_cols)}")
print(f"  Models          : {MODEL_LIST}")
print(f"{'═'*60}")

if USE_WANDB and _WANDB_AVAILABLE:
    wandb.log({
        "final/hybrid_score":    BEST_H,
        "final/method":          METHOD,
        "final/n_features":      len(feature_cols),
        "final/km_72h":          km_prior[72],
        "final/prob_72h_mean":   float(submission["prob_72h"].mean()),
    })
    artifact = wandb.Artifact("submission_v6", type="submission")
    artifact.add_file(sub_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print("  W&B run logged.")

print("\n✅  V6 pipeline complete!")