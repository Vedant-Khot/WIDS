import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def gold_expert_features(df):
    df = df.copy()

    # 1. THE DOMINANT SIGNAL (Report Insight: 4.8km is the magic boundary)
    # Midpoint of 4674 and 5000 is ~4837. This creates a "Pressure Gradient"
    df['dist_sigmoid'] = 1 / (1 + np.exp((df['dist_min_ci_0_5h'] - 4837) / 200))
    df['log_dist'] = np.log1p(df['dist_min_ci_0_5h'])

    # 2. DYNAMIC PHYSICS (Only for low_temporal_resolution = 0)
    # Total Speed = Centroid approach + Radial expansion
    df['total_approach_speed'] = df['closing_speed_m_per_h'] + df['radial_growth_rate_m_per_h']

    # Time-To-Impact (TTI): (Distance - 5km Buffer) / Speed
    # If the fire is already < 5km, distance left is 0.
    dist_to_buffer = np.maximum(0, df['dist_min_ci_0_5h'] - 5000)
    df['tti_estimate'] = np.where(df['total_approach_speed'] > 0,
                                  dist_to_buffer / (df['total_approach_speed'] + 1e-5),
                                  999)

    # Vector Projection: Is the speed actually aimed at us?
    df['vector_momentum'] = df['total_approach_speed'] * df['alignment_abs']

    # 3. STATIC SIZE IMPACT
    df['log_area'] = np.log1p(df['area_first_ha'])
    df['is_morning'] = df['event_start_hour'].apply(lambda x: 1 if 6 <= x <= 12 else 0)

    features = [
        'low_temporal_resolution_0_5h', 'dist_sigmoid', 'log_dist',
        'tti_estimate', 'vector_momentum', 'log_area',
        'alignment_abs', 'is_morning', 'total_approach_speed'
    ]
    return df[features], df

def produce_digital_twins(df, multiplier=20):
    """
    Synthesizes data for the 60 multi-perimeter fires.
    Keeps the Time-To-Impact constant by scaling distance and speed together.
    """
    dynamic = df[df['low_temporal_resolution_0_5h'] == 0].copy()
    synthetic_pool = [df]

    for _ in range(multiplier):
        twin = dynamic.copy()
        # Jitter physics: if it moves 10% faster, it must be 10% further
        # to preserve the 'personality' of the fire's timing.
        phys_jitter = np.random.uniform(0.9, 1.1, size=len(twin))

        twin['dist_min_ci_0_5h'] *= phys_jitter
        twin['closing_speed_m_per_h'] *= phys_jitter
        twin['radial_growth_rate_m_per_h'] *= phys_jitter
        twin['area_first_ha'] *= np.random.uniform(0.98, 1.02, size=len(twin))

        # Jitter the actual target time slightly to prevent memorization
        twin['time_to_hit_hours'] *= np.random.uniform(0.95, 1.05, size=len(twin))
        twin['time_to_hit_hours'] = twin['time_to_hit_hours'].clip(0, 72)

        synthetic_pool.append(twin)
aug_train = pd.read_csv('/content/drive/MyDrive/Google AI Studio/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Google AI Studio/test.csv')

# 1. Augment and Engineer
# Use the 'aug_train' from Phase 2 (the one with pseudo-labels)
super_train = produce_digital_twins(aug_train, multiplier=20)
X_super, df_super = gold_expert_features(super_train)
X_test_final, _ = gold_expert_features(test)

scaler = StandardScaler()
X_super_scaled = scaler.fit_transform(X_super)
X_test_scaled = scaler.transform(X_test_final)

horizons = [12, 24, 48, 72]
final_098_submission = pd.DataFrame({'event_id': test['event_id']})

for h in horizons:
    print(f"\n>>> Training Experts for {h}h...")
    y = ((df_super['event'] == 1) & (df_super['time_to_hit_hours'] <= h)).astype(int)
    
    # Training Masks
    mask_stat_tr = df_super['low_temporal_resolution_0_5h'] == 1
    mask_dyn_tr = df_super['low_temporal_resolution_0_5h'] == 0
    
    # Test Masks
    mask_stat_ts = test['low_temporal_resolution_0_5h'] == 1
    mask_dyn_ts = test['low_temporal_resolution_0_5h'] == 0
    
    h_preds = np.zeros(len(test))
    
    # Brain A: The Static Specialist (trained on 1-perimeter fires)
    model_stat = CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.01, verbose=0, random_state=42)
    model_stat.fit(X_super_scaled[mask_stat_tr], y[mask_stat_tr])
    h_preds[mask_stat_ts] = model_stat.predict_proba(X_test_scaled[mask_stat_ts])[:, 1]
    
    # Brain B: The Dynamic Specialist (trained on 1,200+ digital twins)
    model_dyn = CatBoostClassifier(iterations=2000, depth=5, l2_leaf_reg=20, learning_rate=0.005, verbose=0, random_state=42)
    model_dyn.fit(X_super_scaled[mask_dyn_tr], y[mask_dyn_tr])
    h_preds[mask_dyn_ts] = model_dyn.predict_proba(X_test_scaled[mask_dyn_ts])[:, 1]
    
    final_098_submission[f'prob_{h}h'] = h_preds
# --- STEP 3: THE 0.98 SOFT-LANDING REFINEMENT ---

cols = [f'prob_{h}h' for h in horizons]
final_098_submission[cols] = np.maximum.accumulate(final_098_submission[cols].values, axis=1)

# Get the distance from the test set for calibration
test_dist = test['dist_min_ci_0_5h']
low_res_mask = (test['low_temporal_resolution_0_5h'] == 1)

for col in cols:
    # 1. THE DANGER ZONE (< 4700m)
    # Instead of hard 1.0, we blend the model with 0.995
    # This protects you if a fire very close somehow dies out
    if col == 'prob_72h':
        final_098_submission.loc[test_dist < 4700, col] = \
            (final_098_submission.loc[test_dist < 4700, col] * 0.2) + 0.795 # Result ~0.98-1.0
    
    # 2. THE GRAY ZONE (4700m to 5500m)
    # We let the model's Physics (Digital Twins) speak here. No changes needed.
    
    # 3. THE SAFE ZONE (> 5500m)
    # Instead of hard 0.0, we use a 'Probability Floor'
    # If the model said 0.0001, we keep it. If the model said 0.1, we pull it down.
    far_mask = test_dist > 5500
    final_098_submission.loc[far_mask, col] = np.minimum(final_098_submission.loc[far_mask, col], 0.005)
    
    # 4. THE STATIONARY PENALTY
    # If the fire has only 1 perimeter AND it's outside the 5km zone, 
    # its probability of hitting is nearly zero because it's not moving.
    static_far = low_res_mask & (test_dist > 5000)
    final_098_submission.loc[static_far, col] = np.minimum(final_098_submission.loc[static_far, col], 0.001)

# --- FINAL SHARPENING (The 'Noiseless' decimals) ---
# We round to 5 decimal places to remove floating point junk
final_098_submission[cols] = final_098_submission[cols].round(5)

# Ensure non-decreasing one last time
final_098_submission[cols] = np.maximum.accumulate(final_098_submission[cols].values, axis=1)

final_098_submission.to_csv('final_soft_16_04.csv', index=False)
print("\n[REFINED]: 0.98 Soft-Expert File Saved.")

    return pd.concat(synthetic_pool, axis=0).reset_index(drop=True)
