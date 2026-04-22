import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# --------------------------------------------------------------------------
# 1. LOAD DATA & SEASONAL MAPPING
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

# Representative California Seasonal Fire Risk Features (Historical Climatology)
# month: (fire_hazard_prior, avg_temp_c, avg_humidity_pct)
seasonal_data = {
    1:  (0.05, 12, 65),
    2:  (0.06, 14, 60),
    3:  (0.12, 18, 55),
    4:  (0.25, 22, 45),
    5:  (0.45, 28, 35),
    6:  (0.75, 32, 25),
    7:  (0.95, 35, 20),
    8:  (0.92, 34, 22),
    9:  (0.65, 29, 30),
    10: (0.35, 23, 40),
    11: (0.15, 17, 50),
    12: (0.08, 13, 60)
}

# --------------------------------------------------------------------------
# 2. EXPERT PHYSICS ENGINE + EXTERNAL FEATURE JOIN
# --------------------------------------------------------------------------
def expert_physics_engine(df):
    d = df.copy()
    eps = 1e-6

    # --- A. EXTERNAL SEASONAL FEATURES ---
    d['seasonal_hazard'] = d['event_start_month'].map(lambda x: seasonal_data[x][0])
    d['seasonal_temp']   = d['event_start_month'].map(lambda x: seasonal_data[x][1])
    d['seasonal_hum']    = d['event_start_month'].map(lambda x: seasonal_data[x][2])
    d['seasonal_dryness'] = 100 - d['seasonal_hum']

    # --- B. CORE PHYSICS (from 0.964 optimized versions) ---
    d['dist_sigmoid'] = 1 / (1 + np.exp((d['dist_min_ci_0_5h'] - 4837) / 200))
    d['log_dist'] = np.log1p(d['dist_min_ci_0_5h'])

    d['total_speed'] = d['closing_speed_m_per_h'] + d['radial_growth_rate_m_per_h']
    dist_to_buffer = np.maximum(0, d['dist_min_ci_0_5h'] - 5000)
    d['tti_estimate'] = np.where(d['total_speed'] > 0.5,
                                  dist_to_buffer / (d['total_speed'] + eps),
                                  999) 
    d['tti_log'] = np.log1p(d['tti_estimate'])

    d['log_area'] = np.log1p(d['area_first_ha'])
    d['danger_momentum'] = d['total_speed'] * d['alignment_abs']
    
    d['radius_vs_dist'] = d['radial_growth_m'] / (d['dist_min_ci_0_5h'] + eps)
    d['bearing_toward_evac'] = d['alignment_abs'] * d['alignment_cos']
    d['aligned_closing'] = d['alignment_cos'] * d['closing_speed_m_per_h']

    # --- C. TEMPORAL META ---
    d['is_morning']   = ((d['event_start_hour'] >= 6) & (d['event_start_hour'] <= 12)).astype(int)
    d['is_afternoon'] = ((d['event_start_hour'] > 12) & (d['event_start_hour'] <= 18)).astype(int)
    d['is_night']     = ((d['event_start_hour'] >= 22) | (d['event_start_hour'] <= 5)).astype(int)

    return d

train_fe = expert_physics_engine(train)
test_fe  = expert_physics_engine(test)

FEATURES = [
    'low_temporal_resolution_0_5h', 'dist_sigmoid', 'log_dist',
    'tti_estimate', 'tti_log', 'danger_momentum', 'log_area',
    'alignment_abs', 'total_speed', 'radius_vs_dist',
    'bearing_toward_evac', 'aligned_closing',
    'seasonal_hazard', 'seasonal_temp', 'seasonal_dryness',
    'is_morning', 'is_afternoon', 'is_night',
    'dist_min_ci_0_5h', 'closing_speed_m_per_h',
    'radial_growth_rate_m_per_h', 'centroid_speed_m_per_h',
    'alignment_cos', 'area_growth_abs_0_5h',
    'dist_change_ci_0_5h', 'dist_slope_ci_0_5h',
    'event_start_hour', 'event_start_month'
]

# --------------------------------------------------------------------------
# 3. DIGITAL TWINS AUGMENTATION (Fold-Aware)
# --------------------------------------------------------------------------
def produce_digital_twins(df, multiplier=15, seed=42):
    np.random.seed(seed)
    aug_df = df.copy().reset_index(drop=True)
    aug_df['sample_weight'] = 1.0
    
    dynamic_mask = aug_df['low_temporal_resolution_0_5h'] == 0
    dynamic_df = aug_df[dynamic_mask].copy()
    
    adjusted_weight = 1.0 / (1.0 + multiplier)
    aug_df.loc[dynamic_mask, 'sample_weight'] = adjusted_weight
    
    synthetic_pool = [aug_df]
    if len(dynamic_df) > 0:
        for _ in range(multiplier):
            twin = dynamic_df.copy()
            twin['sample_weight'] = adjusted_weight
            jitter = np.random.uniform(0.94, 1.06, size=len(twin))
            twin['dist_min_ci_0_5h'] *= jitter
            twin['closing_speed_m_per_h'] *= jitter
            twin['radial_growth_rate_m_per_h'] *= jitter
            twin['time_to_hit_hours'] *= np.random.uniform(0.98, 1.02, size=len(twin))
            # Physic engine recalculation
            twin = expert_physics_engine(twin)
            synthetic_pool.append(twin)

    return pd.concat(synthetic_pool, axis=0).reset_index(drop=True)

# --------------------------------------------------------------------------
# 4. ULTIMATE TRAINING: RANK-THEN-CALIBRATE (Protects C-index)
# --------------------------------------------------------------------------
horizons = [12, 24, 48, 72]
N_FOLDS = 5
N_SEEDS = 3

test_preds = {h: np.zeros(len(test)) for h in horizons}

print("=== Starting Ultimate Seasonal Model Pipeline (v6) ===")

for h in horizons:
    print(f"\nHorizon: {h}h")
    
    # 1. Subset for this horizon (drop censored fires that finished before h)
    df_h = train_fe.copy()
    unknown_mask = (df_h['event'] == 0) & (df_h['time_to_hit_hours'] < h)
    df_h = df_h[~unknown_mask].reset_index(drop=True)
    y_h = ((df_h['event'] == 1) & (df_h['time_to_hit_hours'] <= h)).astype(int).values
    
    print(f"  Valid training samples: {len(df_h)} | Hit rate: {y_h.mean():.2%}")
    
    # Heuristic for single-class (72h case)
    if len(np.unique(y_h)) < 2:
        print(f"  Warning: Single class detected. Using ranking heuristic.")
        danger = (1 / (test_fe['dist_min_ci_0_5h'] + 1)) * 0.5 + (test_fe['seasonal_hazard'] * 0.5)
        test_preds[h] = 0.96 + (pd.Series(danger).rank(pct=True).values * 0.03) 
        continue

    oof_blended_ranks = np.zeros(len(df_h))
    horizon_test_ranks = np.zeros(len(test))
    
    for seed in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 * seed + 10)
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(df_h, y_h)):
            df_tr, df_val = df_h.iloc[tr_idx], df_h.iloc[val_idx]
            
            # Augment training fold
            df_tr_aug = produce_digital_twins(df_tr, multiplier=12, seed=seed)
            X_tr = df_tr_aug[FEATURES].fillna(0).values
            y_tr_aug = ((df_tr_aug['event'] == 1) & (df_tr_aug['time_to_hit_hours'] <= h)).astype(int).values
            w_tr = df_tr_aug['sample_weight'].values
            
            X_val = df_val[FEATURES].fillna(0).values
            X_ts = test_fe[FEATURES].fillna(0).values
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_ts_s = scaler.transform(X_ts)
            
            # --- Model A: Segmented CatBoost ---
            mask_stat_tr = df_tr_aug['low_temporal_resolution_0_5h'] == 1
            mask_dyn_tr = df_tr_aug['low_temporal_resolution_0_5h'] == 0
            
            preds_cat_val = np.zeros(len(df_val))
            preds_cat_ts = np.zeros(len(test))
            
            # Static fire expert
            if mask_stat_tr.sum() > 5:
                cb_s = CatBoostClassifier(iterations=800, depth=3, learning_rate=0.03, verbose=0, random_seed=42)
                cb_s.fit(X_tr_s[mask_stat_tr], y_tr_aug[mask_stat_tr], sample_weight=w_tr[mask_stat_tr])
                preds_cat_ts += cb_s.predict_proba(X_ts_s)[:, 1]
                preds_cat_val += cb_s.predict_proba(X_val_s)[:, 1]
            
            # Dynamic fire expert
            if mask_dyn_tr.sum() > 5:
                cb_d = CatBoostClassifier(iterations=1200, depth=5, learning_rate=0.02, verbose=0, random_seed=42)
                cb_d.fit(X_tr_s[mask_dyn_tr], y_tr_aug[mask_dyn_tr], sample_weight=w_tr[mask_dyn_tr])
                dyn_ts_mask = test_fe['low_temporal_resolution_0_5h'] == 0
                preds_cat_ts[dyn_ts_mask] = cb_d.predict_proba(X_ts_s[dyn_ts_mask])[:, 1]
                dyn_val_mask = df_val['low_temporal_resolution_0_5h'] == 0
                preds_cat_val[dyn_val_mask] = cb_d.predict_proba(X_val_s[dyn_val_mask])[:, 1]

            # --- Model B: Global LightGBM ---
            lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.025, max_depth=4, num_leaves=12,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
            lgbm.fit(X_tr_s, y_tr_aug, sample_weight=w_tr)
            preds_lgbm_val = lgbm.predict_proba(X_val_s)[:, 1]
            preds_lgbm_ts = lgbm.predict_proba(X_ts_s)[:, 1]
            
            # --- RANK BLENDING (Protects C-index) ---
            rank_cat_val = pd.Series(preds_cat_val).rank(pct=True)
            rank_lgbm_val = pd.Series(preds_lgbm_val).rank(pct=True)
            blended_rank_val = (0.55 * rank_cat_val + 0.45 * rank_lgbm_val)
            
            rank_cat_ts = pd.Series(preds_cat_ts).rank(pct=True)
            rank_lgbm_ts = pd.Series(preds_lgbm_ts).rank(pct=True)
            blended_rank_ts = (0.55 * rank_cat_ts + 0.45 * rank_lgbm_ts)
            
            oof_blended_ranks[val_idx] += blended_rank_val.values / N_SEEDS
            horizon_test_ranks += blended_rank_ts.values / (N_FOLDS * N_SEEDS)
            
    # --- PROPER CALIBRATION (Fixes Brier Score) ---
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(oof_blended_ranks, y_h)
    calibrated = ir.transform(horizon_test_ranks)
    
    # Tiny tie-breaker to ensure 8-decimal precision uniqueness without affecting Brier/AUC
    # This adds 1e-9 scaled by rank to break isotonic ties
    test_preds[h] = calibrated + (horizon_test_ranks * 1e-9)
    print(f"  Done {h}h. Mean calibrated prob: {test_preds[h].mean():.6f}")

# --------------------------------------------------------------------------
# 5. FINAL POST-PROCESSING
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 5. THE STARK LOGIC GATES (Dynamic Physics Post-Processing)
# --------------------------------------------------------------------------
submission = pd.DataFrame({'event_id': test['event_id']})
cols = [f'prob_{h}h' for h in horizons]
for h, col in zip(horizons, cols):
    submission[col] = test_preds[h]

print("\nDeploying The Stark Logic Gates...")

# Stark Constants
V_MAX_BASE = 420.0  # m/h (Theoretical max speed in absolute peak summer)
SAFETY_MARGIN = 1.15 # 15% buffer above calculated reach-time

# Prepare seasonal scaling for test set
test_fe = expert_physics_engine(test)
test_dist = test_fe['dist_min_ci_0_5h'].values
seasonal_hazard = test_fe['seasonal_hazard'].values
closing_speed = test_fe['closing_speed_m_per_h'].values

zero_counts = {}
max_counts = {}

for h, col in zip(horizons, cols):
    # 1. Dynamic V_MAX (Higher hazard = higher potential speed)
    # Scaled such that at hazard 1.0, V_MAX = 420; at hazard 0.0, V_MAX = 84
    v_max_h = V_MAX_BASE * (seasonal_hazard * 0.8 + 0.2)
    
    # 2. Dynamic Zeroing (Impossible Reach Zone)
    limit = v_max_h * h * SAFETY_MARGIN
    mask_zero = test_dist > limit
    
    zero_counts[h] = mask_zero.sum()
    submission.loc[mask_zero, col] = 0.0
    
    # 3. Hard Target Injection (Certain Strike Zone)
    # If fire is < 500m and moving TOWARD us (closing_speed > 0)
    # We set 12h+ to 0.9999 (High certainty hit)
    mask_max = (test_dist < 500) & (closing_speed > 5)
    max_counts[h] = mask_max.sum()
    submission.loc[mask_max, col] = np.maximum(submission.loc[mask_max, col], 0.9999)

print(f"  Zeroed rows: {zero_counts}")
print(f"  Maxed rows: {max_counts}")

# Row-wise Monotonicity Fix (Ensures prob_12h <= prob_24h etc.)
print("Finalizing Monotonicity...")
submission[cols] = np.maximum.accumulate(submission[cols].values, axis=1)

submission[cols] = submission[cols].round(8)
out_path = os.path.join(BASE_DIR, 'wids_ultimate_v7_stark_submission.csv')
submission.to_csv(out_path, index=False)

print(f"\nSUCCESS! Stark submission saved to: {out_path}")
print(submission.head(10))
