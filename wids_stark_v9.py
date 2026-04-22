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
# 2. EXPERT PHYSICS ENGINE
# NEW v9: Added dist_slope interaction features
# --------------------------------------------------------------------------
def expert_physics_engine(df):
    d = df.copy()
    eps = 1e-6

    # --- A. EXTERNAL SEASONAL FEATURES ---
    d['seasonal_hazard'] = d['event_start_month'].map(lambda x: seasonal_data[x][0])
    d['seasonal_temp']   = d['event_start_month'].map(lambda x: seasonal_data[x][1])
    d['seasonal_hum']    = d['event_start_month'].map(lambda x: seasonal_data[x][2])
    d['seasonal_dryness'] = 100 - d['seasonal_hum']

    # --- B. CORE PHYSICS ---
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
    d['is_morning']   = ((d['event_start_hour'] >= 6)  & (d['event_start_hour'] <= 12)).astype(int)
    d['is_afternoon'] = ((d['event_start_hour'] > 12)  & (d['event_start_hour'] <= 18)).astype(int)
    d['is_night']     = ((d['event_start_hour'] >= 22) | (d['event_start_hour'] <= 5)).astype(int)

    # -------------------------------------------------------------------------
    # NEW v9 — DIST_SLOPE INTERACTION FEATURES
    # dist_slope_ci_0_5h > 0 means fire is moving AWAY (retreating)
    # dist_slope_ci_0_5h < 0 means fire is moving TOWARD (approaching)
    # These create hard asymmetric signals the base features miss.
    # -------------------------------------------------------------------------
    d['fire_retreating']  = (d['dist_slope_ci_0_5h'] > 0).astype(int)
    d['fire_approaching'] = (d['dist_slope_ci_0_5h'] < 0).astype(int)

    # How aggressively is it retreating/approaching? (signed slope magnitude)
    d['approach_rate'] = -d['dist_slope_ci_0_5h']  # positive = approaching fast

    # Interaction: retreating fire with high model confidence → dampen signal
    d['retreat_x_dist']  = d['fire_retreating'] * d['dist_min_ci_0_5h']

    # Interaction: approaching fire close to zone → amplify danger signal
    d['approach_x_speed'] = d['fire_approaching'] * d['total_speed']

    # Effective TTI using slope-adjusted approach rate
    effective_speed = np.maximum(d['total_speed'] + d['approach_rate'], eps)
    d['tti_slope_adjusted'] = np.where(
        effective_speed > 0.5,
        dist_to_buffer / effective_speed,
        999
    )
    d['tti_slope_log'] = np.log1p(d['tti_slope_adjusted'])

    return d

train_fe = expert_physics_engine(train)
test_fe  = expert_physics_engine(test)

# NEW v9 features added to FEATURES list
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
    'event_start_hour', 'event_start_month',
    # --- NEW v9 ---
    'fire_retreating', 'fire_approaching', 'approach_rate',
    'retreat_x_dist', 'approach_x_speed',
    'tti_slope_adjusted', 'tti_slope_log',
]

# --------------------------------------------------------------------------
# 3. DIGITAL TWINS AUGMENTATION (unchanged from v8)
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
            twin = expert_physics_engine(twin)
            synthetic_pool.append(twin)

    return pd.concat(synthetic_pool, axis=0).reset_index(drop=True)

# --------------------------------------------------------------------------
# 4. TRAINING: RANK-THEN-CALIBRATE
# NEW v9: Calibrate CatBoost and LightGBM SEPARATELY before blending,
#         so IsotonicRegression sees a cleaner signal per model.
# --------------------------------------------------------------------------
horizons = [12, 24, 48, 72]
N_FOLDS  = 5
N_SEEDS  = 3

test_preds = {h: np.zeros(len(test)) for h in horizons}

print("=== Starting Stark Logic Gates v9 Pipeline ===")

for h in horizons:
    print(f"\n--- Horizon: {h}h ---")

    df_h = train_fe.copy()
    unknown_mask = (df_h['event'] == 0) & (df_h['time_to_hit_hours'] < h)
    df_h = df_h[~unknown_mask].reset_index(drop=True)
    y_h  = ((df_h['event'] == 1) & (df_h['time_to_hit_hours'] <= h)).astype(int).values

    print(f"  Samples: {len(df_h)} | Hit rate: {y_h.mean():.2%}")

    if len(np.unique(y_h)) < 2:
        print(f"  Single class — using ranking heuristic.")
        danger = (1 / (test_fe['dist_min_ci_0_5h'] + 1)) * 0.5 + (test_fe['seasonal_hazard'] * 0.5)
        test_preds[h] = 0.96 + (pd.Series(danger).rank(pct=True).values * 0.03)
        continue

    # -----------------------------------------------------------------------
    # NEW v9: Separate OOF accumulators for CatBoost and LightGBM
    # We will calibrate each independently, then blend calibrated probs.
    # This gives IsotonicRegression a cleaner, model-specific signal.
    # -----------------------------------------------------------------------
    oof_cat_ranks  = np.zeros(len(df_h))
    oof_lgbm_ranks = np.zeros(len(df_h))
    test_cat_ranks  = np.zeros(len(test))
    test_lgbm_ranks = np.zeros(len(test))

    for seed in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 * seed + 10)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(df_h, y_h)):
            df_tr, df_val = df_h.iloc[tr_idx], df_h.iloc[val_idx]

            df_tr_aug = produce_digital_twins(df_tr, multiplier=12, seed=seed)
            X_tr  = df_tr_aug[FEATURES].fillna(0).values
            y_tr_aug = ((df_tr_aug['event'] == 1) & (df_tr_aug['time_to_hit_hours'] <= h)).astype(int).values
            w_tr  = df_tr_aug['sample_weight'].values

            X_val = df_val[FEATURES].fillna(0).values
            X_ts  = test_fe[FEATURES].fillna(0).values

            scaler   = StandardScaler()
            X_tr_s   = scaler.fit_transform(X_tr)
            X_val_s  = scaler.transform(X_val)
            X_ts_s   = scaler.transform(X_ts)

            # --- Model A: Segmented CatBoost ---
            mask_stat_tr = df_tr_aug['low_temporal_resolution_0_5h'] == 1
            mask_dyn_tr  = df_tr_aug['low_temporal_resolution_0_5h'] == 0

            preds_cat_val = np.zeros(len(df_val))
            preds_cat_ts  = np.zeros(len(test))

            if mask_stat_tr.sum() > 5:
                cb_s = CatBoostClassifier(iterations=800, depth=3, learning_rate=0.03,
                                          verbose=0, random_seed=42)
                cb_s.fit(X_tr_s[mask_stat_tr], y_tr_aug[mask_stat_tr],
                         sample_weight=w_tr[mask_stat_tr])
                preds_cat_ts  += cb_s.predict_proba(X_ts_s)[:, 1]
                preds_cat_val += cb_s.predict_proba(X_val_s)[:, 1]

            if mask_dyn_tr.sum() > 5:
                cb_d = CatBoostClassifier(iterations=1200, depth=5, learning_rate=0.02,
                                          verbose=0, random_seed=42)
                cb_d.fit(X_tr_s[mask_dyn_tr], y_tr_aug[mask_dyn_tr],
                         sample_weight=w_tr[mask_dyn_tr])
                dyn_ts_mask  = test_fe['low_temporal_resolution_0_5h'] == 0
                dyn_val_mask = df_val['low_temporal_resolution_0_5h'] == 0
                preds_cat_ts[dyn_ts_mask]   = cb_d.predict_proba(X_ts_s[dyn_ts_mask])[:, 1]
                preds_cat_val[dyn_val_mask] = cb_d.predict_proba(X_val_s[dyn_val_mask])[:, 1]

            # --- Model B: Global LightGBM ---
            lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.025, max_depth=4,
                                  num_leaves=12, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=-1)
            lgbm.fit(X_tr_s, y_tr_aug, sample_weight=w_tr)
            preds_lgbm_val = lgbm.predict_proba(X_val_s)[:, 1]
            preds_lgbm_ts  = lgbm.predict_proba(X_ts_s)[:, 1]

            # Accumulate SEPARATE rank OOFs (NEW v9)
            oof_cat_ranks[val_idx]  += pd.Series(preds_cat_val).rank(pct=True).values  / N_SEEDS
            oof_lgbm_ranks[val_idx] += pd.Series(preds_lgbm_val).rank(pct=True).values / N_SEEDS
            test_cat_ranks  += pd.Series(preds_cat_ts).rank(pct=True).values  / (N_FOLDS * N_SEEDS)
            test_lgbm_ranks += pd.Series(preds_lgbm_ts).rank(pct=True).values / (N_FOLDS * N_SEEDS)

    # -----------------------------------------------------------------------
    # NEW v9: Calibrate each model separately, then blend calibrated probs
    # -----------------------------------------------------------------------
    ir_cat  = IsotonicRegression(out_of_bounds='clip')
    ir_lgbm = IsotonicRegression(out_of_bounds='clip')

    ir_cat.fit(oof_cat_ranks,  y_h)
    ir_lgbm.fit(oof_lgbm_ranks, y_h)

    cal_cat_ts  = ir_cat.transform(test_cat_ranks)
    cal_lgbm_ts = ir_lgbm.transform(test_lgbm_ranks)

    # Blend calibrated probabilities (55/45 — same ratio as before)
    blended = 0.55 * cal_cat_ts + 0.45 * cal_lgbm_ts

    # Tiny tie-breaker (preserves uniqueness without affecting Brier/AUC)
    test_preds[h] = blended + (test_cat_ranks * 1e-9)

    print(f"  Done {h}h | Mean prob: {test_preds[h].mean():.6f} | "
          f"Cat mean: {cal_cat_ts.mean():.4f} | LGBM mean: {cal_lgbm_ts.mean():.4f}")

# --------------------------------------------------------------------------
# 5. POST-PROCESSING: STARK LOGIC GATES v9
# Replaces the aggressive v8 quantum protocol.
# --------------------------------------------------------------------------
print("\n=== Stark Logic Gates v9 Post-Processing ===")

submission = pd.DataFrame({'event_id': test['event_id']})
cols = [f'prob_{h}h' for h in horizons]
for h, col in zip(horizons, cols):
    submission[col] = test_preds[h]

# --- Physics Constants ---
V_MAX_PHYSICS    = 400.0   # m/h — max fire spread (training data: ~380 m/h)
SAFETY_MARGIN    = 1.25    # fire must be 25% beyond reachable zone to be zeroed
NEAR_DIST_THRESH = 400.0   # meters — fire is essentially at the zone boundary
NEAR_SPEED_THRESH = 2.0    # m/h minimum closing speed for strike boost
NEAR_CERTAIN_PROB = 0.97   # NOT 1.0 — preserves calibration for ~3% edge cases

# Seasonal V_max scaling: 0.6 + 0.4 * seasonal_hazard → range [0.62, 0.98]
v_scale      = 0.6 + (0.4 * test_fe['seasonal_hazard'].values)
v_max_eff    = V_MAX_PHYSICS * v_scale
test_dist    = test_fe['dist_min_ci_0_5h'].values
closing_spd  = test_fe['closing_speed_m_per_h'].values

zero_counts   = {}
strike_counts = {}

for h, col in zip(horizons, cols):
    probs = submission[col].values.copy()

    # GATE 1: Dynamic physics zeroing (horizon-aware)
    reachable = v_max_eff * h * SAFETY_MARGIN
    mask_zero = test_dist > reachable
    probs[mask_zero] = 0.0

    # GATE 2: Near-certain strike boost (conservative — 0.97 not 1.0)
    mask_strike = (test_dist < NEAR_DIST_THRESH) & (closing_spd > NEAR_SPEED_THRESH)
    mask_strike_active = mask_strike & (probs > 0.0)
    probs[mask_strike_active] = np.maximum(probs[mask_strike_active], NEAR_CERTAIN_PROB)

    # NO Q_LOW / Q_HIGH clipping (removed from v8 — was destroying calibration)

    submission[col] = probs
    zero_counts[h]   = int((probs == 0.0).sum())
    strike_counts[h] = int(mask_strike_active.sum())

# Monotonicity enforcement (after ALL gates applied)
arr = submission[cols].values.copy()
submission[cols] = np.maximum.accumulate(arr, axis=1)

# Verify
violations = sum(
    (submission[cols[i]] > submission[cols[i+1]]).sum()
    for i in range(len(cols) - 1)
)

# --------------------------------------------------------------------------
# 6. DIAGNOSTICS
# --------------------------------------------------------------------------
print("\n--- Gate Summary ---")
for h in horizons:
    col   = f'prob_{h}h'
    probs = submission[col].values
    n_total = len(probs)
    print(f"  {h:2d}h | Zeroed: {zero_counts[h]:4d} ({100*zero_counts[h]/n_total:.1f}%) "
          f"| Strikes: {strike_counts[h]:3d} "
          f"| Mean(non-zero): {probs[probs>0].mean():.4f} "
          f"| Max: {probs.max():.4f}")

print(f"\n  Monotonicity violations: {violations} (must be 0)")
assert violations == 0, "Monotonicity violated — check gate logic!"
assert zero_counts[72] <= zero_counts[12], \
    "72h has MORE zeros than 12h — physics gate is inverted!"

print("\n--- Sanity: zeroed rows should decrease as horizon increases ---")
for h in horizons:
    print(f"  {h}h zeros: {zero_counts[h]}")

# --------------------------------------------------------------------------
# 7. SAVE
# --------------------------------------------------------------------------
submission[cols] = submission[cols].round(8)
out_path = os.path.join(BASE_DIR, 'wids_stark_v9_submission.csv')
submission.to_csv(out_path, index=False)

print(f"\nSUCCESS — v9 submission saved: {out_path}")
print(submission[cols].describe().round(4))
