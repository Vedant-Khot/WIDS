# ============================================================
# WiDS 2025 Datathon - Wildfire Survival Analysis (v4)
# ============================================================
# Upgrades from v3:
# 1. Full Hybrid Score evaluation in every Optuna trial:
#       Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)
#    Weighted Brier = 0.3×Brier@24h + 0.4×Brier@48h + 0.3×Brier@72h
# 2. OOF survival probabilities properly collected:
#    - GBSA / RSF : via predict_survival_function()
#    - XGB / LGB  : via isotonic regression calibration of risk scores
# 3. Fixed XGB bug: XGBRegressor (not XGBRFRegressor) for survival:cox.
# 4. Full GBSA and LGB training functions added (no more placeholders).
# 5. Ensemble blending optimised on the Hybrid Score (not just C-index).
# 6. W&B logs per-fold and final hybrid score components.
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import wandb

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

import os

# ============================================================
# 0. CONFIGURATION & EXPERIMENT TRACKING
# ============================================================
USE_WANDB               = True   # Set False to disable W&B
OPTUNA_TRIALS_PER_MODEL = 25     # Trials per model per fold

if USE_WANDB:
    wandb.init(
        project="WiDS2025-Wildfire",
        name=f"v4_hybrid_{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
        config={
            "n_folds":              5,
            "n_repeats":            3,
            "optuna_trials":        OPTUNA_TRIALS_PER_MODEL,
            "hybrid_c_index_weight": 0.3,
            "hybrid_brier_weight":   0.7,
            "brier_weights":         {"24h": 0.3, "48h": 0.4, "72h": 0.3},
        }
    )

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================
print("=" * 70)
print("WiDS 2025 WILDFIRE SURVIVAL ANALYSIS (v4 — Hybrid Score)")
print("=" * 70)

DATA_DIR   = r"d:\wids"
train_df   = pd.read_csv(os.path.join(DATA_DIR, "train_imputed_advanced.csv"))
test_df    = pd.read_csv(os.path.join(DATA_DIR, "test_imputed_advanced.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = ['event_id', 'time_to_hit_hours', 'event']

def engineer_features(df):
    df = df.copy()
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

HORIZONS      = [12, 24, 48, 72]
EVAL_HORIZONS = [24, 48, 72]          # Brier score horizons
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}

# ============================================================
# 2. METRICS
# ============================================================

def compute_brier_censor_aware(y_true_event, y_true_time, pred_probs, horizon):
    """
    Censor-aware Brier score at a single time horizon H.
      - Event fires  : label = 1 if hit ≤ H, else 0
      - Censored ≥ H : label = 0  (included)
      - Censored < H : excluded   (we don't know their outcome)
    """
    mask   = np.ones(len(y_true_event), dtype=bool)
    labels = np.zeros(len(y_true_event))
    for i in range(len(y_true_event)):
        if y_true_event[i]:
            labels[i] = 1.0 if y_true_time[i] <= horizon else 0.0
        else:
            if y_true_time[i] >= horizon:
                labels[i] = 0.0
            else:
                mask[i] = False   # censored before horizon → exclude
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((pred_probs[mask] - labels[mask]) ** 2))


def compute_weighted_brier(y_true_event, y_true_time, pred_probs_dict):
    """
    Weighted Brier Score:
        WBS = 0.3×Brier@24h + 0.4×Brier@48h + 0.3×Brier@72h
    pred_probs_dict : {24: array, 48: array, 72: array}
    """
    total = 0.0
    for h, w in BRIER_WEIGHTS.items():
        total += w * compute_brier_censor_aware(
            y_true_event, y_true_time, pred_probs_dict[h], h
        )
    return total


def compute_hybrid_score(y_true_event, y_true_time, risk_scores, pred_probs_dict):
    """
    Primary competition metric:
        Hybrid = 0.3 × C-index + 0.7 × (1 − Weighted Brier Score)

    Args:
        risk_scores     : 1-D array, higher = higher risk (used for C-index)
        pred_probs_dict : {24: array, 48: array, 72: array}
                          P(hit by horizon H) for each eval horizon
    Returns:
        (hybrid, c_index, weighted_brier_score)
    """
    c_idx, _, _, _, _ = concordance_index_censored(
        y_true_event, y_true_time, risk_scores
    )
    wbs    = compute_weighted_brier(y_true_event, y_true_time, pred_probs_dict)
    hybrid = 0.3 * c_idx + 0.7 * (1.0 - wbs)
    return hybrid, c_idx, wbs


# ── Probability helpers ───────────────────────────────────────────────────────

def survfunc_to_probs(surv_fns, horizons):
    """
    Convert sksurv step-function objects to P(hit by H) = 1 − S(H).
    Clamps query to the last observed time if H exceeds the support.
    """
    probs = {h: np.zeros(len(surv_fns)) for h in horizons}
    for i, fn in enumerate(surv_fns):
        t_max = fn.x[-1]
        for h in horizons:
            survival_at_h  = fn(min(h, t_max))
            probs[h][i]    = np.clip(1.0 - survival_at_h, 1e-6, 1 - 1e-6)
    return probs


def risk_to_probs_isotonic(risk_tr, e_tr, t_tr, risk_query, horizons):
    """
    Calibrate risk scores → P(hit by H) using isotonic regression.
    Used for XGB / LGB which output log-hazard but no survival function.

    Fits on training fold (risk_tr, binary label at H),
    then predicts on risk_query (val or test fold).
    """
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
# 3. OPTUNA-TUNED MODEL TRAINING FUNCTIONS
#    Each objective is the Hybrid Score (not just C-index).
# ============================================================
print("\n" + "=" * 70)
print("MODEL TRAINING  (Optuna → Hybrid Score objective)")
print("=" * 70)

n_folds   = 5
n_repeats = 3
rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

model_list = ['GBSA', 'RSF', 'XGB', 'LGB']
oof_preds  = {
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


# ── GBSA ─────────────────────────────────────────────────────────────────────
def train_gbsa_with_optuna(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':         trial.suggest_int('max_depth', 2, 6),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
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
    return best


# ── RSF ──────────────────────────────────────────────────────────────────────
def train_rsf_with_optuna(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 5, 50),
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
    return best


# ── XGBoost (survival:cox) ────────────────────────────────────────────────────
def train_xgb_with_optuna(X_tr, y_tr, X_val, y_val):
    # XGBoost Cox: positive label = event time, negative = censored time
    y_tr_xgb  = np.where(y_tr['event'],  y_tr['time'],  -y_tr['time'])
    y_val_xgb = np.where(y_val['event'], y_val['time'], -y_val['time'])

    def objective(trial):
        params = {
            'objective':        'survival:cox',
            'eval_metric':      'cox-nloglik',
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth':        trial.suggest_int('max_depth', 2, 8),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'verbosity':        0,
            'random_state':     42,
        }
        # XGBRegressor (not XGBRFRegressor) is needed for survival:cox
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr_xgb,
              eval_set=[(X_val, y_val_xgb)],
              verbose=False)
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
    return best


# ── LightGBM (regression objective — cox not available in this LGB build) ──────
# Uses regression on survival times with sign encoding:
#   positive label = event time, negative label = censored time
# This encodes the same survival structure as Cox; risk scores are then
# calibrated to probabilities via isotonic regression (same as XGB path).
def train_lgb_with_optuna(X_tr, y_tr, X_val, y_val):
    # Survival label encoding: positive = event, negative = censored
    y_tr_lgb  = np.where(y_tr['event'],  y_tr['time'],  -y_tr['time'])
    y_val_lgb = np.where(y_val['event'], y_val['time'], -y_val['time'])

    def objective(trial):
        params = {
            'objective':         'regression',
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'seed':              42,
            'verbose':           -1,
            'n_jobs':            -1,
        }
        n_rounds = trial.suggest_int('n_estimators', 100, 500)
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
    return best_booster


# ============================================================
# 4. MAIN CROSS-VALIDATION LOOP
# ============================================================
total_folds = n_folds * n_repeats
fold_num    = 0
fold_metrics = []

for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
    fold_num += 1
    print(f"\n{'─'*70}")
    print(f"  FOLD {fold_num}/{total_folds}")
    print(f"{'─'*70}")

    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_surv[train_idx], y_surv[val_idx]
    e_val, t_val = y_val['event'], y_val['time']
    e_tr,  t_tr  = y_tr['event'],  y_tr['time']

    # ── GBSA ────────────────────────────────────────────────────────────────
    print("  [1/4] Tuning GBSA...")
    gbsa = train_gbsa_with_optuna(X_tr, y_tr, X_val, y_val)

    gbsa_risk_val   = gbsa.predict(X_val)
    gbsa_probs_val  = survfunc_to_probs(gbsa.predict_survival_function(X_val),         EVAL_HORIZONS)
    gbsa_risk_test  = gbsa.predict(X_test_scaled)
    gbsa_probs_test = survfunc_to_probs(gbsa.predict_survival_function(X_test_scaled),  EVAL_HORIZONS)

    oof_preds['GBSA']['risk'][val_idx] += gbsa_risk_val
    test_preds['GBSA']['risk']         += gbsa_risk_test / total_folds
    for h in EVAL_HORIZONS:
        oof_preds['GBSA']['probs'][h][val_idx] += gbsa_probs_val[h]
        test_preds['GBSA']['probs'][h]         += gbsa_probs_test[h] / total_folds

    hs_gbsa, ci_gbsa, wb_gbsa = compute_hybrid_score(e_val, t_val, gbsa_risk_val, gbsa_probs_val)
    print(f"       GBSA  │ Hybrid={hs_gbsa:.4f}  C-idx={ci_gbsa:.4f}  WBS={wb_gbsa:.4f}")

    # ── RSF ─────────────────────────────────────────────────────────────────
    print("  [2/4] Tuning RSF...")
    rsf = train_rsf_with_optuna(X_tr, y_tr, X_val, y_val)

    rsf_risk_val   = rsf.predict(X_val)
    rsf_probs_val  = survfunc_to_probs(rsf.predict_survival_function(X_val),         EVAL_HORIZONS)
    rsf_risk_test  = rsf.predict(X_test_scaled)
    rsf_probs_test = survfunc_to_probs(rsf.predict_survival_function(X_test_scaled),  EVAL_HORIZONS)

    oof_preds['RSF']['risk'][val_idx] += rsf_risk_val
    test_preds['RSF']['risk']         += rsf_risk_test / total_folds
    for h in EVAL_HORIZONS:
        oof_preds['RSF']['probs'][h][val_idx] += rsf_probs_val[h]
        test_preds['RSF']['probs'][h]         += rsf_probs_test[h] / total_folds

    hs_rsf, ci_rsf, wb_rsf = compute_hybrid_score(e_val, t_val, rsf_risk_val, rsf_probs_val)
    print(f"       RSF   │ Hybrid={hs_rsf:.4f}  C-idx={ci_rsf:.4f}  WBS={wb_rsf:.4f}")

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("  [3/4] Tuning XGB...")
    xgb_m = train_xgb_with_optuna(X_tr, y_tr, X_val, y_val)

    xgb_risk_tr   = xgb_m.predict(X_tr)
    xgb_risk_val  = xgb_m.predict(X_val)
    xgb_risk_test = xgb_m.predict(X_test_scaled)
    xgb_probs_val  = risk_to_probs_isotonic(xgb_risk_tr, e_tr, t_tr, xgb_risk_val,  EVAL_HORIZONS)
    xgb_probs_test = risk_to_probs_isotonic(xgb_risk_tr, e_tr, t_tr, xgb_risk_test, EVAL_HORIZONS)

    oof_preds['XGB']['risk'][val_idx] += xgb_risk_val
    test_preds['XGB']['risk']         += xgb_risk_test / total_folds
    for h in EVAL_HORIZONS:
        oof_preds['XGB']['probs'][h][val_idx] += xgb_probs_val[h]
        test_preds['XGB']['probs'][h]         += xgb_probs_test[h] / total_folds

    hs_xgb, ci_xgb, wb_xgb = compute_hybrid_score(e_val, t_val, xgb_risk_val, xgb_probs_val)
    print(f"       XGB   │ Hybrid={hs_xgb:.4f}  C-idx={ci_xgb:.4f}  WBS={wb_xgb:.4f}")

    # ── LightGBM ────────────────────────────────────────────────────────────
    print("  [4/4] Tuning LGB...")
    lgb_m = train_lgb_with_optuna(X_tr, y_tr, X_val, y_val)

    lgb_risk_tr   = lgb_m.predict(X_tr)
    lgb_risk_val  = lgb_m.predict(X_val)
    lgb_risk_test = lgb_m.predict(X_test_scaled)
    lgb_probs_val  = risk_to_probs_isotonic(lgb_risk_tr, e_tr, t_tr, lgb_risk_val,  EVAL_HORIZONS)
    lgb_probs_test = risk_to_probs_isotonic(lgb_risk_tr, e_tr, t_tr, lgb_risk_test, EVAL_HORIZONS)

    oof_preds['LGB']['risk'][val_idx] += lgb_risk_val
    test_preds['LGB']['risk']         += lgb_risk_test / total_folds
    for h in EVAL_HORIZONS:
        oof_preds['LGB']['probs'][h][val_idx] += lgb_probs_val[h]
        test_preds['LGB']['probs'][h]         += lgb_probs_test[h] / total_folds

    hs_lgb, ci_lgb, wb_lgb = compute_hybrid_score(e_val, t_val, lgb_risk_val, lgb_probs_val)
    print(f"       LGB   │ Hybrid={hs_lgb:.4f}  C-idx={ci_lgb:.4f}  WBS={wb_lgb:.4f}")

    fold_metrics.append({
        'fold': fold_num,
        'GBSA_hybrid': hs_gbsa, 'RSF_hybrid': hs_rsf,
        'XGB_hybrid':  hs_xgb,  'LGB_hybrid': hs_lgb,
    })
    if USE_WANDB:
        wandb.log({
            f'fold_{fold_num}/GBSA_hybrid': hs_gbsa,
            f'fold_{fold_num}/RSF_hybrid':  hs_rsf,
            f'fold_{fold_num}/XGB_hybrid':  hs_xgb,
            f'fold_{fold_num}/LGB_hybrid':  hs_lgb,
        })

    oof_counts[val_idx] += 1

# ── Average repeated-fold OOF accumulations ──────────────────────────────────
for m in model_list:
    oof_preds[m]['risk'] /= oof_counts
    for h in EVAL_HORIZONS:
        oof_preds[m]['probs'][h] /= oof_counts

# ============================================================
# 5. ENSEMBLE BLENDING  —  optimised on Hybrid Score
# ============================================================
print("\n" + "=" * 70)
print("OPTIMAL ENSEMBLE BLENDING  (Hybrid Score objective)")
print("=" * 70)

best_hybrid  = -1.0
best_weights = None
best_ci      = None
best_wbs     = None

for w_gbsa in np.arange(0.0, 1.01, 0.1):
    for w_rsf in np.arange(0.0, 1.01 - w_gbsa, 0.1):
        for w_xgb in np.arange(0.0, 1.01 - w_gbsa - w_rsf, 0.1):
            w_lgb = round(1.0 - w_gbsa - w_rsf - w_xgb, 6)
            if w_lgb < -1e-9:
                continue
            w = {'GBSA': w_gbsa, 'RSF': w_rsf, 'XGB': w_xgb, 'LGB': w_lgb}

            # Rank-normalise risk before blending → comparable scale across models
            blend_risk = sum(
                wt * rankdata(oof_preds[mn]['risk']) for mn, wt in w.items()
            )

            # Linear blend of calibrated probabilities at each horizon
            blend_probs = {
                h: np.clip(
                    sum(wt * oof_preds[mn]['probs'][h] for mn, wt in w.items()),
                    1e-6, 1 - 1e-6
                )
                for h in EVAL_HORIZONS
            }

            hs, c_idx, wbs = compute_hybrid_score(
                y_event, y_time, blend_risk, blend_probs
            )

            if hs > best_hybrid:
                best_hybrid  = hs
                best_weights = w
                best_ci      = c_idx
                best_wbs     = wbs

# ── Print decomposed best score ───────────────────────────────────────────────
best_blend_probs = {
    h: np.clip(
        sum(best_weights[mn] * oof_preds[mn]['probs'][h] for mn in model_list),
        1e-6, 1 - 1e-6
    )
    for h in EVAL_HORIZONS
}
b24 = compute_brier_censor_aware(y_event, y_time, best_blend_probs[24], 24)
b48 = compute_brier_censor_aware(y_event, y_time, best_blend_probs[48], 48)
b72 = compute_brier_censor_aware(y_event, y_time, best_blend_probs[72], 72)

print(f"\n{'═'*55}")
print(f"  OOF Hybrid Score  :  {best_hybrid:.4f}")
print(f"{'─'*55}")
print(f"  C-index           :  {best_ci:.4f}  (× 0.30  → {0.3*best_ci:.4f})")
print(f"  1 − Weighted Brier:  {1-best_wbs:.4f}  (× 0.70  → {0.7*(1-best_wbs):.4f})")
print(f"{'─'*55}")
print(f"  Weighted Brier    :  {best_wbs:.4f}")
print(f"    Brier @ 24h     :  {b24:.4f}  (× 0.30)")
print(f"    Brier @ 48h     :  {b48:.4f}  (× 0.40)  ← highest weight")
print(f"    Brier @ 72h     :  {b72:.4f}  (× 0.30)")
print(f"{'─'*55}")
print("  Best blend weights :")
for mn, wt in best_weights.items():
    print(f"    {mn}: {wt:.2f}")
print(f"{'═'*55}")

# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

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

# 12h probability: not directly modelled — derive as 75% of the 24h estimate
final_test_probs[12] = np.clip(final_test_probs[24] * 0.75, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': final_test_probs[12],
    'prob_24h': final_test_probs[24],
    'prob_48h': final_test_probs[48],
    'prob_72h': final_test_probs[72],
})

# Enforce strict monotonicity: P(hit ≤ H) must not decrease as H grows
for i in range(len(submission)):
    p12 = float(submission.loc[i, 'prob_12h'])
    p24 = max(float(submission.loc[i, 'prob_24h']), p12 + 1e-5)
    p48 = max(float(submission.loc[i, 'prob_48h']), p24 + 1e-5)
    p72 = max(float(submission.loc[i, 'prob_72h']), p48 + 1e-5)
    submission.loc[i, ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']] = [
        min(p12, 0.999), min(p24, 0.999), min(p48, 0.999), min(p72, 0.999)
    ]

submission_path = os.path.join(DATA_DIR, "submission_v3.csv")
submission.to_csv(submission_path, index=False)
print(f"\n✅ Submission saved → {submission_path}")
print(f"   Rows: {len(submission)} | Columns: {list(submission.columns)}")

# ── W&B final logging ────────────────────────────────────────────────────────
if USE_WANDB:
    wandb.log({
        "oof/hybrid_score":   best_hybrid,
        "oof/c_index":        best_ci,
        "oof/weighted_brier": best_wbs,
        "oof/brier_24h":      b24,
        "oof/brier_48h":      b48,
        "oof/brier_72h":      b72,
        "best_weights/GBSA":  best_weights['GBSA'],
        "best_weights/RSF":   best_weights['RSF'],
        "best_weights/XGB":   best_weights['XGB'],
        "best_weights/LGB":   best_weights['LGB'],
    })
    artifact = wandb.Artifact('submission_v4', type='submission')
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print("📊  Results logged to Weights & Biases.")