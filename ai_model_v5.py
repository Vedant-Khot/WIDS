# ============================================================
# WiDS 2025 Datathon - Wildfire Survival Analysis (v5)
# ============================================================
# Built on V3's proven infrastructure + V4's best ideas + new improvements:
#
# Phase 1 (Quick Wins):
#   1. V4's new features (night_closing_anomaly, kinetic_threat_proxy)
#   2. Early-hit features (time_pressure, log_urgency, fast_hit_signal)
#   3. Fixed 12h probability — now modelled directly (not heuristic)
#   4. Enhanced W&B logging (per-model params, feature importance, tables)
#
# Phase 2 (Model Improvements):
#   5. Stacking meta-model (Ridge) replaces grid-search blending
#   6. Low-res interaction features (instead of harmful data split)
#   7. Expanded Optuna search spaces with regularization
#
# Phase 3 (Advanced):
#   8. Kaplan-Meier prior blending for probability smoothing
#   9. Seed averaging (5 seeds) for variance reduction
#  10. Per-horizon isotonic calibration on final ensemble
#  11. Advanced features (momentum, deceleration, percentiles)
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter

import os
import wandb

# ============================================================
# 0. CONFIGURATION & EXPERIMENT TRACKING
# ============================================================
USE_WANDB               = True    # Set False to disable W&B
OPTUNA_TRIALS_PER_MODEL = 50      # Expanded from 25 in V3
N_SEEDS                 = 3       # Seed averaging for variance reduction
SEED_LIST               = [42, 123, 456]
KM_PRIOR_WEIGHT         = 0.0     # Disabled — was destroying 72h predictions

DATA_DIR   = r"d:\wids"
N_FOLDS    = 5
N_REPEATS  = 3

CONFIG = {
    "version":               "v5",
    "n_folds":               N_FOLDS,
    "n_repeats":             N_REPEATS,
    "optuna_trials":         OPTUNA_TRIALS_PER_MODEL,
    "n_seeds":               N_SEEDS,
    "km_prior_weight":       KM_PRIOR_WEIGHT,
    "hybrid_c_index_weight": 0.3,
    "hybrid_brier_weight":   0.7,
    "brier_weights":         {"24h": 0.3, "48h": 0.4, "72h": 0.3},
    "meta_model":            "Ridge(alpha=10)",
    "ensemble_method":       "stacking",
}

if USE_WANDB:
    wandb.init(
        project="WiDS2025-Wildfire",
        name=f"v5_stacked_{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
        config=CONFIG,
    )

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING (Phase 1 + 2 + 3)
# ============================================================
print("=" * 70)
print("WiDS 2025 WILDFIRE SURVIVAL ANALYSIS  (v5 — Full Upgrade)")
print("=" * 70)

train_df   = pd.read_csv(os.path.join(DATA_DIR, "train_fixed.csv"))
test_df    = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = ['event_id', 'time_to_hit_hours', 'event']


def engineer_features(df):
    """V5 feature engineering — combines V3 + V4 + new features."""
    df = df.copy()

    # ── V3 core features ─────────────────────────────────────
    df['log_dist_min']               = np.log1p(df['dist_min_ci_0_5h'])
    df['urgency_ratio']              = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['is_night']                   = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)).astype(int)
    df['is_summer']                  = df['event_start_month'].isin([6, 7, 8]).astype(int)
    df['dist_risk_close']            = (df['dist_min_ci_0_5h'] < 3000).astype(int)
    df['has_growth']                 = (df['area_growth_abs_0_5h'] > 0).astype(int)
    df['directional_threat']         = df['alignment_abs'] * df['closing_speed_abs_m_per_h']
    df['wind_driven_index']          = df['centroid_speed_m_per_h'] / (df['radial_growth_rate_m_per_h'] + 1)
    df['explosive_growth_index']     = df['area_growth_rel_0_5h'] / (np.log1p(df['area_first_ha']) + 0.1)
    df['night_growth_anomaly']       = df['is_night'] * df['area_growth_rate_ha_per_h']
    df['confidence_weighted_urgency']= df['urgency_ratio'] * (1 - df['low_temporal_resolution_0_5h'])

    # ── V4 new features (Phase 1.1) ──────────────────────────
    df['night_closing_anomaly']      = df['is_night'] * df['closing_speed_m_per_h']
    df['kinetic_threat_proxy']       = np.log1p(df['area_first_ha']) * (df['closing_speed_m_per_h'] ** 2)

    # ── Early-hit features (Phase 1.2 — from label analysis) ─
    df['fast_hit_signal']            = ((df['dist_min_ci_0_5h'] < 5000) & (df['closing_speed_m_per_h'] > 500)).astype(int)
    df['time_pressure']              = np.exp(-df['dist_min_ci_0_5h'] / 3000) * df['closing_speed_m_per_h']
    df['log_urgency']                = np.log1p(df['urgency_ratio'])

    # ── Low-res interaction features (Phase 2.2) ─────────────
    df['low_res_x_dist']             = df['low_temporal_resolution_0_5h'] * df['dist_min_ci_0_5h']
    df['low_res_x_urgency']          = df['low_temporal_resolution_0_5h'] * df['urgency_ratio']
    df['low_res_x_growth']           = df['low_temporal_resolution_0_5h'] * df['area_growth_rate_ha_per_h']

    # ── Advanced features (Phase 3.4) ────────────────────────
    df['closing_decel']              = (df['dist_accel_m_per_h2'] < 0).astype(int)
    df['fire_momentum']              = df['area_growth_rate_ha_per_h'] * df['centroid_speed_m_per_h']
    df['threat_acceleration']        = df['closing_speed_m_per_h'] * np.abs(df['dist_accel_m_per_h2'])

    # Percentile-rank features (relative position in distribution)
    for col in ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_first_ha']:
        df[f'{col}_pctile'] = df[col].rank(pct=True)

    return df


train_eng = engineer_features(train_df)
test_eng  = engineer_features(test_df)

feature_cols = [c for c in train_eng.columns
                if c not in target_cols and train_eng[c].dtype != 'object']
print(f"Features: {len(feature_cols)}")

X_train = train_eng[feature_cols].values
y_time  = train_eng['time_to_hit_hours'].values
y_event = train_eng['event'].values.astype(bool)
X_test  = test_eng[feature_cols].values

y_surv = np.array(list(zip(y_event, y_time)), dtype=[('event', bool), ('time', float)])

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Phase 1.3: 12h is now a real evaluation horizon, not a heuristic
HORIZONS      = [12, 24, 48, 72]
EVAL_HORIZONS = [12, 24, 48, 72]      # was [24, 48, 72] in V3
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}   # Competition weights (12h not in Brier)

if USE_WANDB:
    wandb.config.update({"n_features": len(feature_cols), "feature_names": feature_cols})

# ============================================================
# 2. KAPLAN-MEIER PRIOR (Phase 3.2)
# ============================================================
print("\n─── Kaplan-Meier Prior ───")
kmf = KaplanMeierFitter()
kmf.fit(y_time, event_observed=y_event)
km_prior = {}
for h in HORIZONS:
    try:
        km_prior[h] = float(1.0 - kmf.predict(h))
    except Exception:
        km_prior[h] = float(1.0 - kmf.survival_function_at_times(h).values[0])
    print(f"  KM P(hit ≤ {h:2d}h) = {km_prior[h]:.4f}")

# ============================================================
# 3. METRICS
# ============================================================

def compute_brier_censor_aware(y_true_event, y_true_time, pred_probs, horizon):
    mask   = np.ones(len(y_true_event), dtype=bool)
    labels = np.zeros(len(y_true_event))
    for i in range(len(y_true_event)):
        if y_true_event[i]:
            labels[i] = 1.0 if y_true_time[i] <= horizon else 0.0
        else:
            if y_true_time[i] >= horizon:
                labels[i] = 0.0
            else:
                mask[i] = False
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((pred_probs[mask] - labels[mask]) ** 2))


def compute_weighted_brier(y_true_event, y_true_time, pred_probs_dict):
    total = 0.0
    for h, w in BRIER_WEIGHTS.items():
        total += w * compute_brier_censor_aware(
            y_true_event, y_true_time, pred_probs_dict[h], h
        )
    return total


def compute_hybrid_score(y_true_event, y_true_time, risk_scores, pred_probs_dict):
    c_idx, _, _, _, _ = concordance_index_censored(
        y_true_event, y_true_time, risk_scores
    )
    wbs    = compute_weighted_brier(y_true_event, y_true_time, pred_probs_dict)
    hybrid = 0.3 * c_idx + 0.7 * (1.0 - wbs)
    return hybrid, c_idx, wbs


# ── Probability helpers ───────────────────────────────────────

def survfunc_to_probs(surv_fns, horizons):
    probs = {h: np.zeros(len(surv_fns)) for h in horizons}
    for i, fn in enumerate(surv_fns):
        t_max = fn.x[-1]
        for h in horizons:
            survival_at_h = fn(min(h, t_max))
            probs[h][i]   = np.clip(1.0 - survival_at_h, 1e-6, 1 - 1e-6)
    return probs


def risk_to_probs_isotonic(risk_tr, e_tr, t_tr, risk_query, horizons):
    probs = {}
    for h in horizons:
        mask   = np.ones(len(e_tr), dtype=bool)
        labels = np.zeros(len(e_tr))
        for i in range(len(e_tr)):
            if e_tr[i]:
                labels[i] = 1.0 if t_tr[i] <= h else 0.0
            else:
                if t_tr[i] >= h:
                    labels[i] = 0.0
                else:
                    mask[i] = False
        if mask.sum() < 10:
            probs[h] = np.full(len(risk_query), 0.5)
            continue
        iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
        iso.fit(risk_tr[mask], labels[mask])
        probs[h] = np.clip(iso.predict(risk_query), 1e-6, 1 - 1e-6)
    return probs


# ============================================================
# 4. OPTUNA-TUNED MODEL TRAINING FUNCTIONS
#    Phase 2.3: Expanded search spaces with regularization
# ============================================================
print("\n" + "=" * 70)
print("MODEL TRAINING  (Optuna → Hybrid Score, Expanded Search)")
print("=" * 70)

model_list = ['GBSA', 'RSF', 'XGB', 'LGB']


# ── GBSA ─────────────────────────────────────────────────────
def train_gbsa_with_optuna(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 600),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth':         trial.suggest_int('max_depth', 2, 7),
            'min_samples_split': trial.suggest_int('min_samples_split', 3, 60),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 2, 30),
            'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
        }
        m = GradientBoostingSurvivalAnalysis(random_state=42, **params)
        m.fit(X_tr, y_tr)
        risk_val  = m.predict(X_val)
        probs_val = survfunc_to_probs(m.predict_survival_function(X_val), EVAL_HORIZONS)
        hs, _, _  = compute_hybrid_score(y_val['event'], y_val['time'], risk_val, probs_val)
        return hs

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_MODEL, show_progress_bar=False)
    best = GradientBoostingSurvivalAnalysis(random_state=42, **study.best_params)
    best.fit(X_tr, y_tr)
    return best, study.best_params


# ── RSF ──────────────────────────────────────────────────────
def train_rsf_with_optuna(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 600),
            'max_depth':         trial.suggest_int('max_depth', 2, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 3, 60),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 2, 40),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        m = RandomSurvivalForest(random_state=42, n_jobs=-1, **params)
        m.fit(X_tr, y_tr)
        risk_val  = m.predict(X_val)
        probs_val = survfunc_to_probs(m.predict_survival_function(X_val), EVAL_HORIZONS)
        hs, _, _  = compute_hybrid_score(y_val['event'], y_val['time'], risk_val, probs_val)
        return hs

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_MODEL, show_progress_bar=False)
    best = RandomSurvivalForest(random_state=42, n_jobs=-1, **study.best_params)
    best.fit(X_tr, y_tr)
    return best, study.best_params


# ── XGBoost (survival:cox) ────────────────────────────────────
def train_xgb_with_optuna(X_tr, y_tr, X_val, y_val):
    y_tr_xgb  = np.where(y_tr['event'],  y_tr['time'],  -y_tr['time'])
    y_val_xgb = np.where(y_val['event'], y_val['time'], -y_val['time'])

    def objective(trial):
        params = {
            'objective':        'survival:cox',
            'eval_metric':      'cox-nloglik',
            'n_estimators':     trial.suggest_int('n_estimators', 50, 600),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth':        trial.suggest_int('max_depth', 2, 8),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma':            trial.suggest_float('gamma', 0, 5.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'verbosity':        0,
            'random_state':     42,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr_xgb, eval_set=[(X_val, y_val_xgb)], verbose=False)
        risk_tr   = m.predict(X_tr)
        risk_val  = m.predict(X_val)
        probs_val = risk_to_probs_isotonic(
            risk_tr, y_tr['event'], y_tr['time'], risk_val, EVAL_HORIZONS
        )
        hs, _, _ = compute_hybrid_score(y_val['event'], y_val['time'], risk_val, probs_val)
        return hs

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_MODEL, show_progress_bar=False)
    best_params = dict(study.best_params,
                       objective='survival:cox', eval_metric='cox-nloglik',
                       verbosity=0, random_state=42)
    best = xgb.XGBRegressor(**best_params)
    best.fit(X_tr, y_tr_xgb, verbose=False)
    return best, study.best_params


# ── LightGBM ────────────────────────────────────────────────
def train_lgb_with_optuna(X_tr, y_tr, X_val, y_val):
    y_tr_lgb  = np.where(y_tr['event'],  y_tr['time'],  -y_tr['time'])
    y_val_lgb = np.where(y_val['event'], y_val['time'], -y_val['time'])

    def objective(trial):
        params = {
            'objective':         'regression',
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 7, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 60),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 2.0),
            'seed':              42,
            'verbose':           -1,
            'n_jobs':            -1,
        }
        n_rounds = trial.suggest_int('n_estimators', 50, 600)
        dtrain = lgb.Dataset(X_tr,  label=y_tr_lgb)
        dval   = lgb.Dataset(X_val, label=y_val_lgb, reference=dtrain)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
        )
        risk_tr   = booster.predict(X_tr)
        risk_val  = booster.predict(X_val)
        probs_val = risk_to_probs_isotonic(
            risk_tr, y_tr['event'], y_tr['time'], risk_val, EVAL_HORIZONS
        )
        hs, _, _ = compute_hybrid_score(y_val['event'], y_val['time'], risk_val, probs_val)
        return hs

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_MODEL, show_progress_bar=False)
    best_p   = dict(study.best_params)
    n_rounds = best_p.pop('n_estimators')
    best_params = dict(best_p, objective='regression', seed=42, verbose=-1, n_jobs=-1)
    dtrain_full = lgb.Dataset(X_tr, label=y_tr_lgb)
    best_booster = lgb.train(
        best_params, dtrain_full,
        num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )
    return best_booster, study.best_params


# ============================================================
# 5. SEED-AVERAGED CROSS-VALIDATION LOOP (Phase 3.3)
# ============================================================
print(f"\n{'═'*70}")
print(f"  SEED-AVERAGED TRAINING  ({N_SEEDS} seeds × {N_FOLDS}×{N_REPEATS} folds)")
print(f"{'═'*70}")

# Accumulators across all seeds
seed_oof_preds = {
    m: {'risk': np.zeros(len(X_train)),
        'probs': {h: np.zeros(len(X_train)) for h in EVAL_HORIZONS}}
    for m in model_list
}
seed_test_preds = {
    m: {'risk': np.zeros(len(X_test)),
        'probs': {h: np.zeros(len(X_test)) for h in EVAL_HORIZONS}}
    for m in model_list
}

all_best_params = {m: [] for m in model_list}

for seed_idx, seed in enumerate(SEED_LIST):
    print(f"\n{'━'*70}")
    print(f"  SEED {seed_idx+1}/{N_SEEDS}  (random_state={seed})")
    print(f"{'━'*70}")

    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=seed)

    oof_preds = {
        m: {'risk': np.zeros(len(X_train)),
            'probs': {h: np.zeros(len(X_train)) for h in EVAL_HORIZONS}}
        for m in model_list
    }
    test_preds = {
        m: {'risk': np.zeros(len(X_test)),
            'probs': {h: np.zeros(len(X_test)) for h in EVAL_HORIZONS}}
        for m in model_list
    }
    oof_counts = np.zeros(len(X_train))

    total_folds = N_FOLDS * N_REPEATS
    fold_num = 0

    for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
        fold_num += 1
        print(f"\n  ─ Seed {seed} │ Fold {fold_num}/{total_folds}")

        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_surv[train_idx], y_surv[val_idx]
        e_val, t_val = y_val['event'], y_val['time']
        e_tr,  t_tr  = y_tr['event'],  y_tr['time']

        # ── GBSA ────────────────────────────────────────────────
        print("    [1/4] GBSA...", end=" ", flush=True)
        gbsa, gbsa_params = train_gbsa_with_optuna(X_tr, y_tr, X_val, y_val)
        gbsa_risk_val   = gbsa.predict(X_val)
        gbsa_probs_val  = survfunc_to_probs(gbsa.predict_survival_function(X_val), EVAL_HORIZONS)
        gbsa_risk_test  = gbsa.predict(X_test_scaled)
        gbsa_probs_test = survfunc_to_probs(gbsa.predict_survival_function(X_test_scaled), EVAL_HORIZONS)

        oof_preds['GBSA']['risk'][val_idx] += gbsa_risk_val
        test_preds['GBSA']['risk']         += gbsa_risk_test / total_folds
        for h in EVAL_HORIZONS:
            oof_preds['GBSA']['probs'][h][val_idx] += gbsa_probs_val[h]
            test_preds['GBSA']['probs'][h]         += gbsa_probs_test[h] / total_folds

        hs_gbsa, ci_gbsa, _ = compute_hybrid_score(e_val, t_val, gbsa_risk_val, gbsa_probs_val)
        print(f"H={hs_gbsa:.4f} C={ci_gbsa:.4f}")
        if fold_num == 1:
            all_best_params['GBSA'].append(gbsa_params)
            # Phase 1.4: Log GBSA feature importance
            if USE_WANDB and hasattr(gbsa, 'feature_importances_'):
                fi = gbsa.feature_importances_
                top_idx = np.argsort(fi)[::-1][:15]
                fi_table = wandb.Table(columns=["feature", "importance"],
                                       data=[[feature_cols[i], float(fi[i])] for i in top_idx])
                wandb.log({f"seed_{seed}/feature_importance": fi_table})

        # ── RSF ─────────────────────────────────────────────────
        print("    [2/4] RSF...", end=" ", flush=True)
        rsf, rsf_params = train_rsf_with_optuna(X_tr, y_tr, X_val, y_val)
        rsf_risk_val   = rsf.predict(X_val)
        rsf_probs_val  = survfunc_to_probs(rsf.predict_survival_function(X_val), EVAL_HORIZONS)
        rsf_risk_test  = rsf.predict(X_test_scaled)
        rsf_probs_test = survfunc_to_probs(rsf.predict_survival_function(X_test_scaled), EVAL_HORIZONS)

        oof_preds['RSF']['risk'][val_idx] += rsf_risk_val
        test_preds['RSF']['risk']         += rsf_risk_test / total_folds
        for h in EVAL_HORIZONS:
            oof_preds['RSF']['probs'][h][val_idx] += rsf_probs_val[h]
            test_preds['RSF']['probs'][h]         += rsf_probs_test[h] / total_folds

        hs_rsf, ci_rsf, _ = compute_hybrid_score(e_val, t_val, rsf_risk_val, rsf_probs_val)
        print(f"H={hs_rsf:.4f} C={ci_rsf:.4f}")
        if fold_num == 1:
            all_best_params['RSF'].append(rsf_params)

        # ── XGBoost ─────────────────────────────────────────────
        print("    [3/4] XGB...", end=" ", flush=True)
        xgb_m, xgb_params = train_xgb_with_optuna(X_tr, y_tr, X_val, y_val)
        xgb_risk_tr   = xgb_m.predict(X_tr)
        xgb_risk_val  = xgb_m.predict(X_val)
        xgb_risk_test = xgb_m.predict(X_test_scaled)
        xgb_probs_val  = risk_to_probs_isotonic(xgb_risk_tr, e_tr, t_tr, xgb_risk_val, EVAL_HORIZONS)
        xgb_probs_test = risk_to_probs_isotonic(xgb_risk_tr, e_tr, t_tr, xgb_risk_test, EVAL_HORIZONS)

        oof_preds['XGB']['risk'][val_idx] += xgb_risk_val
        test_preds['XGB']['risk']         += xgb_risk_test / total_folds
        for h in EVAL_HORIZONS:
            oof_preds['XGB']['probs'][h][val_idx] += xgb_probs_val[h]
            test_preds['XGB']['probs'][h]         += xgb_probs_test[h] / total_folds

        hs_xgb, ci_xgb, _ = compute_hybrid_score(e_val, t_val, xgb_risk_val, xgb_probs_val)
        print(f"H={hs_xgb:.4f} C={ci_xgb:.4f}")
        if fold_num == 1:
            all_best_params['XGB'].append(xgb_params)

        # ── LightGBM ───────────────────────────────────────────
        print("    [4/4] LGB...", end=" ", flush=True)
        lgb_m, lgb_params = train_lgb_with_optuna(X_tr, y_tr, X_val, y_val)
        lgb_risk_tr   = lgb_m.predict(X_tr)
        lgb_risk_val  = lgb_m.predict(X_val)
        lgb_risk_test = lgb_m.predict(X_test_scaled)
        lgb_probs_val  = risk_to_probs_isotonic(lgb_risk_tr, e_tr, t_tr, lgb_risk_val, EVAL_HORIZONS)
        lgb_probs_test = risk_to_probs_isotonic(lgb_risk_tr, e_tr, t_tr, lgb_risk_test, EVAL_HORIZONS)

        oof_preds['LGB']['risk'][val_idx] += lgb_risk_val
        test_preds['LGB']['risk']         += lgb_risk_test / total_folds
        for h in EVAL_HORIZONS:
            oof_preds['LGB']['probs'][h][val_idx] += lgb_probs_val[h]
            test_preds['LGB']['probs'][h]         += lgb_probs_test[h] / total_folds

        hs_lgb, ci_lgb, _ = compute_hybrid_score(e_val, t_val, lgb_risk_val, lgb_probs_val)
        print(f"H={hs_lgb:.4f} C={ci_lgb:.4f}")
        if fold_num == 1:
            all_best_params['LGB'].append(lgb_params)

        # ── W&B per-fold logging ────────────────────────────────
        if USE_WANDB:
            wandb.log({
                f'seed_{seed}/fold_{fold_num}/GBSA_hybrid': hs_gbsa,
                f'seed_{seed}/fold_{fold_num}/RSF_hybrid':  hs_rsf,
                f'seed_{seed}/fold_{fold_num}/XGB_hybrid':  hs_xgb,
                f'seed_{seed}/fold_{fold_num}/LGB_hybrid':  hs_lgb,
            })

        oof_counts[val_idx] += 1

    # Average repeated-fold OOF within this seed
    for m in model_list:
        oof_preds[m]['risk'] /= oof_counts
        for h in EVAL_HORIZONS:
            oof_preds[m]['probs'][h] /= oof_counts

    # Accumulate into seed-averaged totals
    for m in model_list:
        seed_oof_preds[m]['risk']  += oof_preds[m]['risk'] / N_SEEDS
        seed_test_preds[m]['risk'] += test_preds[m]['risk'] / N_SEEDS
        for h in EVAL_HORIZONS:
            seed_oof_preds[m]['probs'][h]  += oof_preds[m]['probs'][h] / N_SEEDS
            seed_test_preds[m]['probs'][h] += test_preds[m]['probs'][h] / N_SEEDS

    # Print per-seed OOF hybrid for each model
    for m in model_list:
        hs, ci, wbs = compute_hybrid_score(
            y_event, y_time, oof_preds[m]['risk'],
            {h: oof_preds[m]['probs'][h] for h in EVAL_HORIZONS}
        )
        print(f"  Seed {seed} │ {m} OOF Hybrid={hs:.4f}  C-idx={ci:.4f}  WBS={wbs:.4f}")

# Rename for the rest of the pipeline
oof_preds  = seed_oof_preds
test_preds = seed_test_preds

# ── Save intermediate predictions to disk (avoid retraining) ─────────
import pickle
intermediate = {
    'oof_preds': oof_preds, 'test_preds': test_preds,
    'y_event': y_event, 'y_time': y_time,
    'feature_cols': feature_cols, 'model_list': model_list,
}
intermediate_path = os.path.join(DATA_DIR, 'v5_intermediate_preds.pkl')
with open(intermediate_path, 'wb') as f:
    pickle.dump(intermediate, f)
print(f"\n  💾 Intermediate predictions saved → {intermediate_path}")

# ============================================================
# 6. STACKING META-MODEL (Phase 2.1)
# ============================================================
print("\n" + "=" * 70)
print("STACKING META-MODEL  (Ridge regression on OOF features)")
print("=" * 70)

# Build meta-features: risk ranks + calibrated probs at each horizon
def build_meta_features(preds_dict, n_samples):
    """Build meta-feature matrix from model predictions."""
    cols = {}
    for m in model_list:
        cols[f'{m}_risk_rank'] = rankdata(preds_dict[m]['risk'])
        for h in EVAL_HORIZONS:
            cols[f'{m}_prob_{h}h'] = preds_dict[m]['probs'][h]
    return pd.DataFrame(cols)

meta_train = build_meta_features(oof_preds, len(X_train))
meta_test  = build_meta_features(test_preds, len(X_test))

print(f"Meta-features: {meta_train.shape[1]} columns")
print(f"  Columns: {list(meta_train.columns)}")

# Train one Ridge meta-model per horizon for probability stacking
# Plus one for risk ranking
meta_models_prob = {}
meta_models_risk = None

# ── Risk stacking (for C-index) ──────────────────────────────
print("\n  Training risk meta-model...")
# Create binary labels for risk
risk_meta = Ridge(alpha=10.0)
risk_meta.fit(meta_train.values, rankdata(oof_preds['GBSA']['risk']))  # placeholder target
# Actually we want to predict risk that correlates with survival time
# Use the rank of time (inverted for hits, direct for censored)
risk_target = np.zeros(len(X_train))
for i in range(len(X_train)):
    if y_event[i]:
        risk_target[i] = 1.0 / (y_time[i] + 0.1)  # shorter time = higher risk
    else:
        risk_target[i] = 0.0  # censored → low risk proxy

risk_meta = Ridge(alpha=10.0)
risk_meta.fit(meta_train.values, risk_target)
stacked_risk_oof = risk_meta.predict(meta_train.values)

# Evaluate stacked risk OOF
c_idx_stacked, _, _, _, _ = concordance_index_censored(y_event, y_time, stacked_risk_oof)
print(f"  Stacked Risk C-index (OOF): {c_idx_stacked:.4f}")

# ── Per-horizon probability stacking (Phase 3.1) ─────────────
print("  Training per-horizon probability meta-models...")
for h in EVAL_HORIZONS:
    # Binary label at this horizon
    labels = np.zeros(len(X_train))
    mask   = np.ones(len(X_train), dtype=bool)
    for i in range(len(X_train)):
        if y_event[i]:
            labels[i] = 1.0 if y_time[i] <= h else 0.0
        else:
            if y_time[i] >= h:
                labels[i] = 0.0
            else:
                mask[i] = False

    # Use only prob columns for this horizon's meta-model
    prob_cols = [f'{m}_prob_{h}h' for m in model_list]
    rank_cols = [f'{m}_risk_rank' for m in model_list]
    use_cols  = prob_cols + rank_cols

    meta_h = Ridge(alpha=5.0)
    meta_h.fit(meta_train.loc[mask, use_cols].values, labels[mask])
    meta_models_prob[h] = (meta_h, use_cols)

    # Evaluate
    pred_h = np.clip(meta_h.predict(meta_train[use_cols].values), 1e-6, 1 - 1e-6)
    brier_h = compute_brier_censor_aware(y_event, y_time, pred_h, h)
    print(f"    Horizon {h:2d}h │ Stacked Brier={brier_h:.4f}")

# ── Compute stacked OOF probabilities ────────────────────────
stacked_oof_probs = {}
for h in EVAL_HORIZONS:
    meta_h, use_cols = meta_models_prob[h]
    stacked_oof_probs[h] = np.clip(
        meta_h.predict(meta_train[use_cols].values), 1e-6, 1 - 1e-6
    )

# ── Evaluate full stacked ensemble ───────────────────────────
hs_stacked, ci_stacked, wbs_stacked = compute_hybrid_score(
    y_event, y_time, stacked_risk_oof, stacked_oof_probs
)
print(f"\n  {'═'*50}")
print(f"  Stacked Ensemble OOF Hybrid Score: {hs_stacked:.4f}")
print(f"    C-index:        {ci_stacked:.4f}")
print(f"    Weighted Brier: {wbs_stacked:.4f}")
print(f"  {'═'*50}")

# ============================================================
# 7. ALSO COMPUTE BLENDED ENSEMBLE (V3-style) FOR COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("BLENDED ENSEMBLE (V3-style, for comparison)")
print("=" * 70)

best_hybrid_blend  = -1.0
best_weights       = None

for w_gbsa in np.arange(0.0, 1.01, 0.1):
    for w_rsf in np.arange(0.0, 1.01 - w_gbsa, 0.1):
        for w_xgb in np.arange(0.0, 1.01 - w_gbsa - w_rsf, 0.1):
            w_lgb = round(1.0 - w_gbsa - w_rsf - w_xgb, 6)
            if w_lgb < -1e-9:
                continue
            w = {'GBSA': w_gbsa, 'RSF': w_rsf, 'XGB': w_xgb, 'LGB': w_lgb}

            blend_risk = sum(
                wt * rankdata(oof_preds[mn]['risk']) for mn, wt in w.items()
            )
            blend_probs = {
                h: np.clip(
                    sum(wt * oof_preds[mn]['probs'][h] for mn, wt in w.items()),
                    1e-6, 1 - 1e-6
                )
                for h in EVAL_HORIZONS
            }
            hs, _, _ = compute_hybrid_score(y_event, y_time, blend_risk, blend_probs)
            if hs > best_hybrid_blend:
                best_hybrid_blend = hs
                best_weights      = w

print(f"  Best Blended Hybrid Score: {best_hybrid_blend:.4f}")
print(f"  Weights: {best_weights}")

# ============================================================
# 8. CHOOSE BEST ENSEMBLE METHOD
# ============================================================
print("\n" + "=" * 70)
print("ENSEMBLE METHOD SELECTION")
print("=" * 70)

use_stacking = hs_stacked >= best_hybrid_blend
method_name  = "STACKING" if use_stacking else "BLENDING"
best_hybrid  = max(hs_stacked, best_hybrid_blend)

print(f"  Stacking Hybrid:  {hs_stacked:.4f}")
print(f"  Blending Hybrid:  {best_hybrid_blend:.4f}")
print(f"  → Using: {method_name}")

# ============================================================
# 9. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

if use_stacking:
    # Stacked predictions
    final_test_risk = risk_meta.predict(meta_test.values)
    final_test_probs = {}
    for h in EVAL_HORIZONS:
        meta_h, use_cols = meta_models_prob[h]
        final_test_probs[h] = np.clip(
            meta_h.predict(meta_test[use_cols].values), 1e-6, 1 - 1e-6
        )
else:
    # Blended predictions (V3-style)
    final_test_risk = sum(
        best_weights[mn] * rankdata(test_preds[mn]['risk'])
        for mn in model_list
    )
    final_test_probs = {
        h: np.clip(
            sum(best_weights[mn] * test_preds[mn]['probs'][h] for mn in model_list),
            1e-6, 1 - 1e-6
        )
        for h in EVAL_HORIZONS
    }

# ── KM prior blending DISABLED — was raising 72h floor too high ──────
if KM_PRIOR_WEIGHT > 0:
    print(f"\n  Applying KM prior blending (weight={KM_PRIOR_WEIGHT})...")
    for h in HORIZONS:
        raw = final_test_probs[h]
        blended = (1.0 - KM_PRIOR_WEIGHT) * raw + KM_PRIOR_WEIGHT * km_prior[h]
        final_test_probs[h] = np.clip(blended, 1e-6, 1 - 1e-6)
        print(f"    {h:2d}h: raw=[{raw.min():.4f}, {raw.max():.4f}]"
              f"  → blended=[{final_test_probs[h].min():.4f}, {final_test_probs[h].max():.4f}]")
else:
    print("\n  KM prior blending: DISABLED")

# ── Final isotonic re-calibration: REMOVED ────────────────────
# This step was DESTRUCTIVE in the first run:
#   72h: [0.6265, 0.9551] → [1.0000, 1.0000]  ← collapsed!
# The base model probabilities from survival functions and per-fold
# isotonic calibration are already well-calibrated. Adding another
# layer of isotonic regression on 221 samples overfits.
print("\n  Final isotonic re-calibration: SKIPPED (was causing collapse)")

# Just print final ranges
for h in EVAL_HORIZONS:
    print(f"    {h:2d}h: [{final_test_probs[h].min():.4f}, {final_test_probs[h].max():.4f}]")

# ── Build submission DataFrame ────────────────────────────────
submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': final_test_probs[12],
    'prob_24h': final_test_probs[24],
    'prob_48h': final_test_probs[48],
    'prob_72h': final_test_probs[72],
})

# Enforce strict monotonicity
for i in range(len(submission)):
    p12 = float(submission.loc[i, 'prob_12h'])
    p24 = max(float(submission.loc[i, 'prob_24h']), p12 + 1e-5)
    p48 = max(float(submission.loc[i, 'prob_48h']), p24 + 1e-5)
    p72 = max(float(submission.loc[i, 'prob_72h']), p48 + 1e-5)
    submission.loc[i, ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']] = [
        min(p12, 0.999), min(p24, 0.999), min(p48, 0.999), min(p72, 0.999)
    ]

# ── SANITY CHECKS (Mistake Note #2: Always validate before saving) ────
print("\n  Running sanity checks...")
sanity_ok = True
for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
    vals = submission[col]
    if vals.nunique() < 3:
        print(f"  ⚠️  FAIL: {col} has only {vals.nunique()} unique values!")
        sanity_ok = False
    if vals.std() < 0.01:
        print(f"  ⚠️  FAIL: {col} has near-zero std={vals.std():.6f}")
        sanity_ok = False
    if vals.min() > 0.5:
        print(f"  ⚠️  WARN: {col} min is suspiciously high: {vals.min():.4f}")
    if vals.max() < 0.5:
        print(f"  ⚠️  WARN: {col} max is suspiciously low: {vals.max():.4f}")
    print(f"  ✓ {col}: [{vals.min():.4f}, {vals.max():.4f}] mean={vals.mean():.4f} unique={vals.nunique()}")

if sanity_ok:
    print("  ✅ All sanity checks passed!")
else:
    print("  ❌ SANITY CHECKS FAILED — review submission before submitting!")

submission_path = os.path.join(DATA_DIR, "submission_v5.csv")
submission.to_csv(submission_path, index=False)

# ============================================================
# 10. FINAL SUMMARY & W&B LOGGING
# ============================================================
print(f"\n{'═'*60}")
print(f"  V5 FINAL RESULTS")
print(f"{'═'*60}")
print(f"  Ensemble Method   : {method_name}")
print(f"  OOF Hybrid Score  : {best_hybrid:.4f}")
print(f"  Seeds averaged    : {N_SEEDS}")
print(f"  Optuna trials/mod : {OPTUNA_TRIALS_PER_MODEL}")
print(f"  Features          : {len(feature_cols)}")
print(f"{'─'*60}")
print(f"  Submission saved  → {submission_path}")
print(f"  Rows: {len(submission)} | Columns: {list(submission.columns)}")
print(f"{'═'*60}")

print("\nSubmission statistics:")
print(submission.describe())

if USE_WANDB:
    # Log final metrics
    wandb.log({
        "final/hybrid_score":   best_hybrid,
        "final/ensemble_method": method_name,
        "final/stacking_hybrid": hs_stacked,
        "final/blending_hybrid": best_hybrid_blend,
        "final/n_features":      len(feature_cols),
    })

    if use_stacking:
        wandb.log({
            "final/stacked_c_index": ci_stacked,
            "final/stacked_wbs":     wbs_stacked,
        })
    else:
        for mn, wt in best_weights.items():
            wandb.log({f"final/blend_weight/{mn}": wt})

    # Log best params from first seed
    for m in model_list:
        if all_best_params[m]:
            wandb.log({f"best_params/{m}": all_best_params[m][0]})

    # Log submission as artifact
    artifact = wandb.Artifact('submission_v5', type='submission')
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)

    # Log prediction distributions
    for h in HORIZONS:
        wandb.log({
            f"pred_dist/prob_{h}h_mean": float(submission[f'prob_{h}h'].mean()),
            f"pred_dist/prob_{h}h_std":  float(submission[f'prob_{h}h'].std()),
            f"pred_dist/prob_{h}h_min":  float(submission[f'prob_{h}h'].min()),
            f"pred_dist/prob_{h}h_max":  float(submission[f'prob_{h}h'].max()),
        })

    wandb.finish()
    print("📊  Results logged to Weights & Biases.")

print("\n✅ V5 pipeline complete!")
