# ============================================================
# WiDS 2025 Datathon - Wildfire Survival Analysis (v6C - Calibrated Triage)
# ============================================================
# Upgrades from v6:
# 1. SOFT TRIAGE: Replaces the hard 4700m cliff with a transition zone (4500-5500m).
# 2. CALIBRATED FAR MODEL: Increases the probability floor for far fires to match
#    population-level KM hazards (0.05 to 0.40 range instead of 0.001-0.10).
# 3. ENSEMBLE CLOSE MODEL: Uses GBSA (0.7) and RSF (0.3) for the timing specialist.
# 4. KM-ANCHORED POST-PROCESSING: Matches the mean 72h prob closer to the 0.55 KM estimate.
# ============================================================

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIGURATION
# ============================================================
print("=" * 70)
print("WiDS 2025 WILDFIRE SURVIVAL ANALYSIS  (v6C — Calibrated Triage)")
print("=" * 70)

DATA_DIR = "."
TRIAGE_LOW = 4500
TRIAGE_HIGH = 5500
KM_HORIZONS = {12: 0.22, 24: 0.29, 48: 0.31, 72: 0.55}

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_fixed.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def create_core_features(df):
    df = df.copy()
    # Handle identifies duplicates from EDA
    duplicates_to_drop = ['relative_growth_0_5h', 'projected_advance_m']
    df = df.drop(columns=duplicates_to_drop, errors='ignore')
    
    # Core Log Transforms
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    df['log_num_perimeters'] = np.log1p(df['num_perimeters_0_5h'])
    
    # Bearing Artifact Fix
    df['bearing_is_real'] = (df['spread_bearing_cos'] != 1.0) & (df['num_perimeters_0_5h'] > 1)
    df['spread_bearing_sin_clean'] = df['spread_bearing_sin'] * df['bearing_is_real']
    df['spread_bearing_cos_clean'] = df['spread_bearing_cos'] * df['bearing_is_real']
    return df

train_eng = create_core_features(train_df)
test_eng = create_core_features(test_df)

features_close = [
    'log_dist_min', 'low_temporal_resolution_0_5h', 'log_num_perimeters',
    'log1p_area_first', 'dt_first_last_0_5h', 'event_start_month',
    'alignment_abs', 'dist_slope_ci_0_5h', 'log1p_growth',
    'radial_growth_rate_m_per_h', 'area_growth_rate_ha_per_h',
    'centroid_speed_m_per_h', 'closing_speed_abs_m_per_h',
    'bearing_is_real', 'spread_bearing_sin_clean', 'spread_bearing_cos_clean'
]

# ============================================================
# 2. TRAIN ENSEMBLE "CLOSE MODEL"
# ============================================================
print("\n--- Training Model: The 'Close/Dangerous' Ensemble ---")
# Only use fires < 5500m for the close model to specialize on hits
train_close = train_eng[train_eng['dist_min_ci_0_5h'] <= TRIAGE_HIGH].copy()
X_close_train = train_close[features_close].values
y_close_surv = np.array(list(zip(train_close['event'].astype(bool), train_close['time_to_hit_hours'])),
                        dtype=[('event', bool), ('time', float)])

scaler = StandardScaler()
X_close_scaled = scaler.fit_transform(X_close_train)

# Model 1: Gradient Boosting Specialist
gbsa = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
gbsa.fit(X_close_scaled, y_close_surv)

# Model 2: Random Forest Specialist
rsf = RandomSurvivalForest(n_estimators=200, max_depth=6, min_samples_leaf=3, n_jobs=-1, random_state=42)
rsf.fit(X_close_scaled, y_close_surv)

print(f"  Ensemble trained on {len(train_close)} high-risk samples.")

# ============================================================
# 3. GENERATE PREDICTIONS WITH CALIBRATED TRIAGE ROUTER
# ============================================================
print("\n--- Generating Calibrated Predictions ---")
test_res = test_eng[['event_id', 'dist_min_ci_0_5h', 'log_dist_min']].copy()
X_close_test = scaler.transform(test_eng[features_close].values)

# Get Base Survival Probabilities from Ensemble (Mean of GBSA & RSF)
surv_gbsa = gbsa.predict_survival_function(X_close_test)
surv_rsf = rsf.predict_survival_function(X_close_test)

def get_hit_prob(fn, h):
    try:
        return 1.0 - fn(h)
    except ValueError:
        return 1.0 if h > fn.x[-1] else 0.0

for h in [12, 24, 48, 72]:
    probs_close = [0.7*get_hit_prob(g, h) + 0.3*get_hit_prob(r, h) for g, r in zip(surv_gbsa, surv_rsf)]
    test_res[f'prob_{h}h'] = np.clip(probs_close, 0.01, 0.99)

# --- CALIBRATION: Apply the "Far-Fire" Probability Floor ---
# Even for far fires, P(hit) should match KM population risk (0.05 - 0.40) 
# instead of hitting 0.0, because training had heavy early censoring.
def calibrate_far(row, horizon):
    dist = row['dist_min_ci_0_5h']
    raw_p = row[f'prob_{horizon}h']
    
    # Distance-based weight: 1.0 if close (<4500m), 0.0 if far (>5500m)
    w_close = np.clip((TRIAGE_HIGH - dist) / (TRIAGE_HIGH - TRIAGE_LOW), 0.0, 1.0)
    
    # Baseline floor for far fires at this horizon.
    # We use v4's winning levels: ~0.05 for early horizons, and 1.0 * KM (0.55) for 72h.
    # This reflects the reality that most "far" training fires were just censored early.
    floor_multipliers = {12: 0.15, 24: 0.15, 48: 0.20, 72: 1.00}
    p_floor = KM_HORIZONS[horizon] * floor_multipliers[horizon]
    
    # Blend: Close models use the specialist, Far fires use the probability floor
    return w_close * raw_p + (1.0 - w_close) * p_floor

print("  Applying v4-anchored probability floors and transition zones...")
for h in [12, 24, 48, 72]:
    test_res[f'prob_{h}h'] = test_res.apply(lambda row: calibrate_far(row, h), axis=1)

# ============================================================
# 4. FINAL SUBMISSION & VALIDATION
# ============================================================
submission = test_res[['event_id', 'prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].copy()

# Enforce Monotonicity (ensure P increases over time)
for i in range(len(submission)):
    for j in range(1, 4):
        submission.iloc[i, j+1] = max(submission.iloc[i, j+1], submission.iloc[i, j] + 0.001)

submission_path = "submission_v6_final.csv"
submission.to_csv(submission_path, index=False)
print(f"\n✅ High-impact submission saved → {submission_path}")
print("\nFinal Statistics (v4-Anchored):")
print(submission[[f'prob_{h}h' for h in [12, 24, 48, 72]]].describe().loc['mean'])