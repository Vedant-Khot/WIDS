"""
WiDS 2025 Datathon - Wildfire Survival Analysis Model (v2)
==========================================================
Improved approach with better calibration for small datasets.
Uses direct survival function output rather than over-aggressive isotonic calibration.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from lifelines import WeibullAFTFitter, CoxPHFitter

import os

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70)
print("WiDS 2025 WILDFIRE SURVIVAL ANALYSIS (v2)")
print("=" * 70)

DATA_DIR = r"d:\wids"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_imputed_advanced.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_imputed_advanced.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
target_cols = ['event_id', 'time_to_hit_hours', 'event']

def engineer_features(df):
    """Create additional features from existing ones."""
    df = df.copy()
    
    # Log distance to evacuation zone
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    
    # Interaction: closing speed × alignment
    df['closing_x_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    
    # Urgency: closing speed / distance
    df['urgency_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    
    # ETA estimate
    df['eta_hours'] = np.where(
        df['closing_speed_m_per_h'] > 0,
        df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 1e-6),
        999
    )
    df['eta_hours'] = df['eta_hours'].clip(0, 999)
    df['log_eta'] = np.log1p(df['eta_hours'])
    
    # Growth indicators
    df['has_growth'] = (df['area_growth_abs_0_5h'] > 0).astype(int)
    df['growth_x_closing'] = df['area_growth_rate_ha_per_h'] * df['closing_speed_m_per_h']
    df['is_closing'] = (df['closing_speed_m_per_h'] > 0).astype(int)
    df['abs_dist_accel'] = np.abs(df['dist_accel_m_per_h2'])
    df['radial_x_alignment'] = df['radial_growth_rate_m_per_h'] * df['alignment_abs']
    
    # Temporal
    df['is_night'] = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)).astype(int)
    df['is_weekend'] = (df['event_start_dayofweek'] >= 5).astype(int)
    df['is_summer'] = df['event_start_month'].isin([6, 7, 8]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    
    # Perimeter density
    df['perimeter_rate'] = df['num_perimeters_0_5h'] / (df['dt_first_last_0_5h'] + 0.1)
    
    # Distance risk categories
    df['dist_risk_close'] = (df['dist_min_ci_0_5h'] < 3000).astype(int)
    df['dist_risk_medium'] = ((df['dist_min_ci_0_5h'] >= 3000) & (df['dist_min_ci_0_5h'] < 10000)).astype(int)
    df['dist_risk_far'] = (df['dist_min_ci_0_5h'] >= 50000).astype(int)
    
    # Cross features
    df['speed_x_dist'] = df['centroid_speed_m_per_h'] * df['dist_min_ci_0_5h']
    df['area_x_growth'] = df['area_first_ha'] * df['area_growth_rel_0_5h']
    df['along_track_urgency'] = df['along_track_speed'] / (df['dist_min_ci_0_5h'] + 1)
    df['proj_advance_ratio'] = df['projected_advance_m'] / (df['dist_min_ci_0_5h'] + 1)
    df['dist_min_sq'] = df['dist_min_ci_0_5h'] ** 2 / 1e10
    df['log_area_first'] = np.log1p(df['area_first_ha'])
    
    # Additional features for v2
    # Interaction between distance and growth
    df['dist_growth_interaction'] = df['log_dist_min'] * df['has_growth']
    
    # Ratio of radial growth to distance
    df['radial_to_dist'] = df['radial_growth_m'] / (df['dist_min_ci_0_5h'] + 1)
    
    # Combined directional signal
    df['directional_threat'] = df['alignment_abs'] * df['closing_speed_abs_m_per_h']
    
    # Fire intensity proxy: area × growth rate
    df['intensity_proxy'] = np.log1p(df['area_first_ha'] * df['area_growth_rate_ha_per_h'])
    
    return df

train_eng = engineer_features(train_df)
test_eng = engineer_features(test_df)

feature_cols = [c for c in train_eng.columns if c not in target_cols]
print(f"Features: {len(feature_cols)}")

X_train = train_eng[feature_cols].values
y_time = train_eng['time_to_hit_hours'].values
y_event = train_eng['event'].values.astype(bool)
X_test = test_eng[feature_cols].values

y_surv = np.array(list(zip(y_event, y_time)), 
                  dtype=[('event', bool), ('time', float)])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

HORIZONS = [12, 24, 48, 72]

# ============================================================
# METRICS
# ============================================================
def compute_brier_censor_aware(y_true_event, y_true_time, pred_probs, horizon):
    mask = np.ones(len(y_true_event), dtype=bool)
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
    return np.mean((pred_probs[mask] - labels[mask]) ** 2)

def compute_weighted_brier(y_true_event, y_true_time, pred_probs_dict):
    weights = {24: 0.3, 48: 0.4, 72: 0.3}
    total = 0
    for h, w in weights.items():
        total += w * compute_brier_censor_aware(y_true_event, y_true_time, pred_probs_dict[h], h)
    return total

# ============================================================
# 3. MODEL TRAINING - REPEATED 5-FOLD CV FOR STABILITY
# ============================================================
print("\n" + "=" * 70)
print("MODEL TRAINING (Repeated CV for stability)")
print("=" * 70)

n_folds = 5
n_repeats = 3
rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

# ---- GBSA ----
print("\n--- Gradient Boosting Survival Analysis ---")
gbsa_configs = [
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_split': 5, 'subsample': 0.8},
    {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 10, 'subsample': 0.7},
    {'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 2, 'min_samples_split': 5, 'subsample': 0.9},
    {'n_estimators': 200, 'learning_rate': 0.08, 'max_depth': 4, 'min_samples_split': 8, 'subsample': 0.75},
    {'n_estimators': 250, 'learning_rate': 0.03, 'max_depth': 3, 'min_samples_split': 7, 'subsample': 0.85},
]

test_probs_gbsa = {h: np.zeros(len(X_test)) for h in HORIZONS}
test_risk_gbsa = np.zeros(len(X_test))
oof_probs_gbsa = {h: np.zeros(len(X_train)) for h in HORIZONS}
oof_risk_gbsa = np.zeros(len(X_train))
oof_counts = np.zeros(len(X_train))

fold_c_indices = []
fold_hybrids = []

total_folds = n_folds * n_repeats
fold_num = 0

for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
    fold_num += 1
    
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr = y_surv[train_idx]
    
    fold_probs = {h: np.zeros(len(val_idx)) for h in HORIZONS}
    fold_risk = np.zeros(len(val_idx))
    test_fold_probs = {h: np.zeros(len(X_test)) for h in HORIZONS}
    test_fold_risk = np.zeros(len(X_test))
    
    for config in gbsa_configs:
        model = GradientBoostingSurvivalAnalysis(random_state=42 + fold_num, **config)
        model.fit(X_tr, y_tr)
        
        surv_fns = model.predict_survival_function(X_val)
        risk = model.predict(X_val)
        
        for i, fn in enumerate(surv_fns):
            for h in HORIZONS:
                try:
                    prob = 1 - fn(h)
                except:
                    prob = 1.0 if h > fn.x[-1] else 0.0
                fold_probs[h][i] += prob / len(gbsa_configs)
        fold_risk += risk / len(gbsa_configs)
        
        test_surv_fns = model.predict_survival_function(X_test_scaled)
        test_risk_pred = model.predict(X_test_scaled)
        
        for i, fn in enumerate(test_surv_fns):
            for h in HORIZONS:
                try:
                    prob = 1 - fn(h)
                except:
                    prob = 1.0 if h > fn.x[-1] else 0.0
                test_fold_probs[h][i] += prob / len(gbsa_configs)
        test_fold_risk += test_risk_pred / len(gbsa_configs)
    
    # Accumulate OOF
    for h in HORIZONS:
        oof_probs_gbsa[h][val_idx] += fold_probs[h]
    oof_risk_gbsa[val_idx] += fold_risk
    oof_counts[val_idx] += 1
    
    # Accumulate test
    for h in HORIZONS:
        test_probs_gbsa[h] += test_fold_probs[h] / total_folds
    test_risk_gbsa += test_fold_risk / total_folds
    
    c_idx = concordance_index_censored(y_event[val_idx], y_time[val_idx], fold_risk)[0]
    fold_c_indices.append(c_idx)
    
    if fold_num % n_folds == 0:
        print(f"  Repeat {fold_num // n_folds}/{n_repeats} avg C-index: {np.mean(fold_c_indices[-n_folds:]):.4f}")

# Average OOF
for h in HORIZONS:
    oof_probs_gbsa[h] /= oof_counts
oof_risk_gbsa /= oof_counts

c_gbsa = concordance_index_censored(y_event, y_time, oof_risk_gbsa)[0]
wbs_gbsa = compute_weighted_brier(y_event, y_time, oof_probs_gbsa)
hybrid_gbsa = 0.3 * c_gbsa + 0.7 * (1 - wbs_gbsa)
print(f"  GBSA OOF: C-index={c_gbsa:.4f}, WBS={wbs_gbsa:.4f}, Hybrid={hybrid_gbsa:.4f}")

# ---- RSF ----
print("\n--- Random Survival Forest ---")
rsf_configs = [
    {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
    {'n_estimators': 500, 'max_depth': 7, 'min_samples_split': 8, 'min_samples_leaf': 3},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 12, 'min_samples_leaf': 4},
]

test_probs_rsf = {h: np.zeros(len(X_test)) for h in HORIZONS}
test_risk_rsf = np.zeros(len(X_test))
oof_probs_rsf = {h: np.zeros(len(X_train)) for h in HORIZONS}
oof_risk_rsf = np.zeros(len(X_train))
oof_counts_rsf = np.zeros(len(X_train))

fold_num = 0
for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
    fold_num += 1
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr = y_surv[train_idx]
    
    fold_probs = {h: np.zeros(len(val_idx)) for h in HORIZONS}
    fold_risk = np.zeros(len(val_idx))
    test_fold_probs = {h: np.zeros(len(X_test)) for h in HORIZONS}
    test_fold_risk = np.zeros(len(X_test))
    
    for config in rsf_configs:
        rsf = RandomSurvivalForest(random_state=42 + fold_num, n_jobs=-1, **config)
        rsf.fit(X_tr, y_tr)
        
        surv_fns = rsf.predict_survival_function(X_val)
        risk = rsf.predict(X_val)
        
        for i, fn in enumerate(surv_fns):
            for h in HORIZONS:
                try:
                    prob = 1 - fn(h)
                except:
                    prob = 1.0 if h > fn.x[-1] else 0.0
                fold_probs[h][i] += prob / len(rsf_configs)
        fold_risk += risk / len(rsf_configs)
        
        test_surv_fns = rsf.predict_survival_function(X_test_scaled)
        test_risk_pred = rsf.predict(X_test_scaled)
        for i, fn in enumerate(test_surv_fns):
            for h in HORIZONS:
                try:
                    prob = 1 - fn(h)
                except:
                    prob = 1.0 if h > fn.x[-1] else 0.0
                test_fold_probs[h][i] += prob / len(rsf_configs)
        test_fold_risk += test_risk_pred / len(rsf_configs)
    
    for h in HORIZONS:
        oof_probs_rsf[h][val_idx] += fold_probs[h]
    oof_risk_rsf[val_idx] += fold_risk
    oof_counts_rsf[val_idx] += 1
    
    for h in HORIZONS:
        test_probs_rsf[h] += test_fold_probs[h] / total_folds
    test_risk_rsf += test_fold_risk / total_folds
    
    if fold_num % n_folds == 0:
        print(f"  Repeat {fold_num // n_folds}/{n_repeats} done")

for h in HORIZONS:
    oof_probs_rsf[h] /= oof_counts_rsf
oof_risk_rsf /= oof_counts_rsf

c_rsf = concordance_index_censored(y_event, y_time, oof_risk_rsf)[0]
wbs_rsf = compute_weighted_brier(y_event, y_time, oof_probs_rsf)
hybrid_rsf = 0.3 * c_rsf + 0.7 * (1 - wbs_rsf)
print(f"  RSF OOF: C-index={c_rsf:.4f}, WBS={wbs_rsf:.4f}, Hybrid={hybrid_rsf:.4f}")

# ---- Cox PH via lifelines ----
print("\n--- Cox PH ---")
test_probs_cox = {h: np.zeros(len(X_test)) for h in HORIZONS}
oof_probs_cox = {h: np.zeros(len(X_train)) for h in HORIZONS}
oof_risk_cox = np.zeros(len(X_train))
oof_counts_cox = np.zeros(len(X_train))

fold_num = 0
for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
    fold_num += 1
    train_fold_df = pd.DataFrame(X_train_scaled[train_idx], columns=feature_cols)
    train_fold_df['time'] = y_time[train_idx]
    train_fold_df['event'] = y_event[train_idx].astype(int)
    val_fold_df = pd.DataFrame(X_train_scaled[val_idx], columns=feature_cols)
    test_fold_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    try:
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
        cph.fit(train_fold_df, duration_col='time', event_col='event', show_progress=False)
        
        val_surv = cph.predict_survival_function(val_fold_df)
        test_surv = cph.predict_survival_function(test_fold_df)
        
        for h in HORIZONS:
            # Use interpolation for exact horizon values
            closest_time = val_surv.index[np.argmin(np.abs(val_surv.index - h))]
            oof_probs_cox[h][val_idx] += 1 - val_surv.loc[closest_time].values
            
            closest_time_test = test_surv.index[np.argmin(np.abs(test_surv.index - h))]
            test_probs_cox[h] += (1 - test_surv.loc[closest_time_test].values) / total_folds
        
        val_risk = cph.predict_partial_hazard(val_fold_df).values.flatten()
        oof_risk_cox[val_idx] += val_risk
        oof_counts_cox[val_idx] += 1
    except:
        oof_counts_cox[val_idx] += 1
        for h in HORIZONS:
            oof_probs_cox[h][val_idx] += 0.5
    
    if fold_num % n_folds == 0:
        print(f"  Repeat {fold_num // n_folds}/{n_repeats} done")

for h in HORIZONS:
    oof_probs_cox[h] /= oof_counts_cox
oof_risk_cox /= np.maximum(oof_counts_cox, 1)

c_cox = concordance_index_censored(y_event, y_time, oof_risk_cox)[0]
wbs_cox = compute_weighted_brier(y_event, y_time, oof_probs_cox)
hybrid_cox = 0.3 * c_cox + 0.7 * (1 - wbs_cox)
print(f"  Cox PH OOF: C-index={c_cox:.4f}, WBS={wbs_cox:.4f}, Hybrid={hybrid_cox:.4f}")

# ---- Weibull AFT ----
print("\n--- Weibull AFT ---")
imp_feats = [
    'dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_first_ha',
    'area_growth_rate_ha_per_h', 'alignment_abs', 'radial_growth_rate_m_per_h',
    'centroid_speed_m_per_h', 'log_dist_min', 'urgency_ratio', 'has_growth',
    'is_closing', 'dist_risk_close', 'log_area_first', 'projected_advance_m',
    'along_track_speed', 'num_perimeters_0_5h'
]

test_probs_weibull = {h: np.zeros(len(X_test)) for h in HORIZONS}
oof_probs_weibull = {h: np.zeros(len(X_train)) for h in HORIZONS}
oof_risk_weibull = np.zeros(len(X_train))
oof_counts_weibull = np.zeros(len(X_train))
weibull_fold_count = 0

fold_num = 0
for train_idx, val_idx in rskf.split(X_train_scaled, y_event):
    fold_num += 1
    train_fold_df = pd.DataFrame(X_train[train_idx], columns=feature_cols)
    train_fold_df['T'] = np.clip(y_time[train_idx], 0.001, None)
    train_fold_df['E'] = y_event[train_idx].astype(int)
    val_fold_df = pd.DataFrame(X_train[val_idx], columns=feature_cols)
    test_fold_df = pd.DataFrame(X_test, columns=feature_cols)
    
    try:
        waft = WeibullAFTFitter(penalizer=0.05)
        waft.fit(train_fold_df[imp_feats + ['T', 'E']], duration_col='T', event_col='E', show_progress=False)
        
        val_surv = waft.predict_survival_function(val_fold_df[imp_feats])
        test_surv = waft.predict_survival_function(test_fold_df[imp_feats])
        
        for h in HORIZONS:
            closest_time = val_surv.index[np.argmin(np.abs(val_surv.index - h))]
            oof_probs_weibull[h][val_idx] += 1 - val_surv.loc[closest_time].values
            closest_time_test = test_surv.index[np.argmin(np.abs(test_surv.index - h))]
            test_probs_weibull[h] += (1 - test_surv.loc[closest_time_test].values) / total_folds
        
        val_median = waft.predict_median(val_fold_df[imp_feats]).values.flatten()
        oof_risk_weibull[val_idx] += -val_median
        oof_counts_weibull[val_idx] += 1
        weibull_fold_count += 1
    except Exception as e:
        oof_counts_weibull[val_idx] += 1
        for h in HORIZONS:
            oof_probs_weibull[h][val_idx] += 0.5
    
    if fold_num % n_folds == 0:
        print(f"  Repeat {fold_num // n_folds}/{n_repeats} done")

for h in HORIZONS:
    oof_probs_weibull[h] /= np.maximum(oof_counts_weibull, 1)
oof_risk_weibull /= np.maximum(oof_counts_weibull, 1)

c_weibull = concordance_index_censored(y_event, y_time, oof_risk_weibull)[0]
wbs_weibull = compute_weighted_brier(y_event, y_time, oof_probs_weibull)
hybrid_weibull = 0.3 * c_weibull + 0.7 * (1 - wbs_weibull)
print(f"  Weibull AFT OOF: C-index={c_weibull:.4f}, WBS={wbs_weibull:.4f}, Hybrid={hybrid_weibull:.4f}")

# ============================================================
# 4. OPTIMAL ENSEMBLE BLENDING
# ============================================================
print("\n" + "=" * 70)
print("OPTIMAL ENSEMBLE BLENDING")
print("=" * 70)

model_results = {
    'GBSA': {'hybrid': hybrid_gbsa, 'oof_probs': oof_probs_gbsa, 'oof_risk': oof_risk_gbsa,
             'test_probs': test_probs_gbsa, 'test_risk': test_risk_gbsa},
    'RSF': {'hybrid': hybrid_rsf, 'oof_probs': oof_probs_rsf, 'oof_risk': oof_risk_rsf,
            'test_probs': test_probs_rsf, 'test_risk': test_risk_rsf},
    'Cox': {'hybrid': hybrid_cox, 'oof_probs': oof_probs_cox, 'oof_risk': oof_risk_cox,
            'test_probs': test_probs_cox, 'test_risk': np.zeros(len(X_test))},
    'Weibull': {'hybrid': hybrid_weibull, 'oof_probs': oof_probs_weibull, 'oof_risk': oof_risk_weibull,
                'test_probs': test_probs_weibull, 'test_risk': np.zeros(len(X_test))},
}

print("\nIndividual model scores:")
for name, res in model_results.items():
    print(f"  {name}: Hybrid={res['hybrid']:.4f}")

# Grid search for optimal blend weights
best_hybrid = -1
best_weights = None

# Focus primarily on GBSA and RSF (the non-parametric models which typically perform best)
# Also include Cox and Weibull with smaller weights
for w_gbsa in np.arange(0.2, 0.7, 0.05):
    for w_rsf in np.arange(0.2, 0.7, 0.05):
        remaining = 1.0 - w_gbsa - w_rsf
        if remaining < 0 or remaining > 0.5:
            continue
        for w_cox in np.arange(0, remaining + 0.01, 0.05):
            w_weibull = remaining - w_cox
            if w_weibull < -0.01 or w_weibull > 0.4:
                continue
            w_weibull = max(0, w_weibull)
            
            ws = [w_gbsa, w_rsf, w_cox, w_weibull]
            ws_sum = sum(ws)
            ws = [w / ws_sum for w in ws]
            
            # Blend probabilities
            blend_probs = {}
            for h in HORIZONS:
                blend_probs[h] = np.zeros(len(X_train))
                for name, w in zip(model_results.keys(), ws):
                    blend_probs[h] += w * model_results[name]['oof_probs'][h]
            
            # Blend risk (using rank-based blending for C-index)
            blend_risk = np.zeros(len(X_train))
            for name, w in zip(model_results.keys(), ws):
                blend_risk += w * rankdata(model_results[name]['oof_risk']) / len(X_train)
            
            c_idx = concordance_index_censored(y_event, y_time, blend_risk)[0]
            wbs = compute_weighted_brier(y_event, y_time, blend_probs)
            hybrid = 0.3 * c_idx + 0.7 * (1 - wbs)
            
            if hybrid > best_hybrid:
                best_hybrid = hybrid
                best_weights = dict(zip(model_results.keys(), ws))

print(f"\nBest blend weights:")
for name, w in best_weights.items():
    print(f"  {name}: {w:.4f}")
print(f"Best blend Hybrid: {best_hybrid:.4f}")

# Generate final blended predictions
final_test_probs = {}
for h in HORIZONS:
    final_test_probs[h] = np.zeros(len(X_test))
    for name, w in best_weights.items():
        final_test_probs[h] += w * model_results[name]['test_probs'][h]

# ============================================================
# 5. SMOOTH CALIBRATION (NOT ISOTONIC - too aggressive for small data)
# ============================================================
print("\n" + "=" * 70)
print("SMOOTH PROBABILITY CALIBRATION")
print("=" * 70)

# Use a simple beta calibration approach: fit a logistic mapping
from sklearn.linear_model import LogisticRegression

for h in HORIZONS:
    # Create binary labels
    labels = np.zeros(len(y_event))
    mask = np.ones(len(y_event), dtype=bool)
    for i in range(len(y_event)):
        if y_event[i]:
            labels[i] = 1.0 if y_time[i] <= h else 0.0
        else:
            if y_time[i] >= h:
                labels[i] = 0.0
            else:
                mask[i] = False
    
    n_valid = mask.sum()
    n_pos = labels[mask].sum()
    n_neg = n_valid - n_pos
    
    # Blend OOF predictions for calibration
    oof_blend_h = np.zeros(len(X_train))
    for name, w in best_weights.items():
        oof_blend_h += w * model_results[name]['oof_probs'][h]
    
    if n_valid > 20 and n_pos > 3 and n_neg > 3:
        # Use logistic regression (Platt scaling) - smoother than isotonic
        # Transform predictions to log-odds space
        oof_clip = np.clip(oof_blend_h[mask], 0.001, 0.999)
        log_odds = np.log(oof_clip / (1 - oof_clip)).reshape(-1, 1)
        
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(log_odds, labels[mask])
        
        # Calibrate test predictions
        test_clip = np.clip(final_test_probs[h], 0.001, 0.999)
        test_log_odds = np.log(test_clip / (1 - test_clip)).reshape(-1, 1)
        final_test_probs[h] = lr.predict_proba(test_log_odds)[:, 1]
        
        # Evaluate calibration
        oof_calib = lr.predict_proba(log_odds)[:, 1]
        brier_before = np.mean((oof_blend_h[mask] - labels[mask]) ** 2)
        brier_after = np.mean((oof_calib - labels[mask]) ** 2)
        print(f"  {h}h: valid={n_valid}, pos={int(n_pos)}, Brier: {brier_before:.4f} -> {brier_after:.4f}")
    else:
        print(f"  {h}h: insufficient data for calibration (valid={n_valid}, pos={int(n_pos)})")

# ============================================================
# 6. MONOTONICITY ENFORCEMENT
# ============================================================
print("\n" + "=" * 70)
print("MONOTONICITY ENFORCEMENT")
print("=" * 70)

# Clip to valid range
for h in HORIZONS:
    final_test_probs[h] = np.clip(final_test_probs[h], 0.001, 0.999)

# Enforce monotonicity: P(t<=12) <= P(t<=24) <= P(t<=48) <= P(t<=72)
for i in range(len(X_test)):
    probs = [final_test_probs[h][i] for h in HORIZONS]
    
    # Use isotonic regression on the 4 values to enforce monotonicity
    # (pool adjacent violators algorithm)
    corrected = list(probs)
    for j in range(1, 4):
        if corrected[j] < corrected[j-1]:
            # Set both to their average
            avg = (corrected[j] + corrected[j-1]) / 2
            corrected[j] = min(avg + 0.005, 0.999)
            corrected[j-1] = max(avg - 0.005, 0.001)
    
    # Final pass to guarantee strict monotonicity
    for j in range(1, 4):
        if corrected[j] <= corrected[j-1]:
            corrected[j] = min(corrected[j-1] + 0.01, 0.999)
    
    for j, h in enumerate(HORIZONS):
        final_test_probs[h][i] = corrected[j]

violations = sum(1 for i in range(len(X_test)) 
                 for j in range(1, 4) 
                 if final_test_probs[HORIZONS[j]][i] < final_test_probs[HORIZONS[j-1]][i])
print(f"Monotonicity violations: {violations}")

# ============================================================
# 7. GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': final_test_probs[12],
    'prob_24h': final_test_probs[24],
    'prob_48h': final_test_probs[48],
    'prob_72h': final_test_probs[72],
})

submission = submission.set_index('event_id').loc[sample_sub['event_id']].reset_index()

submission_path = os.path.join(DATA_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Saved to: {submission_path}")
print(f"\nPrediction statistics:")
for h in HORIZONS:
    col = f'prob_{h}h'
    v = submission[col]
    print(f"  {col}: mean={v.mean():.4f}, std={v.std():.4f}, min={v.min():.4f}, max={v.max():.4f}")

# Distribution check
print(f"\nPrediction distribution (prob_48h):")
for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
    count = (submission['prob_48h'] >= threshold).sum()
    print(f"  >= {threshold}: {count}/{len(submission)} ({count/len(submission)*100:.1f}%)")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nBest individual model: {max(model_results, key=lambda k: model_results[k]['hybrid'])}"
      f" (Hybrid: {max(r['hybrid'] for r in model_results.values()):.4f})")
print(f"Optimized ensemble:    Hybrid: {best_hybrid:.4f}")
print(f"\n✅ Submission file ready: {submission_path}")