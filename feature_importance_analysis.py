"""
Feature Importance Analysis for Wildfire Survival Model
Determines which factors are most prominent in predicting wildfire threat.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

DATA_DIR = r"d:\wids"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

# Raw feature columns (before engineering)
target_cols = ['event_id', 'time_to_hit_hours', 'event']
raw_feature_cols = [c for c in train_df.columns if c not in target_cols]

print("=" * 70)
print("RAW DATASET FEATURES")
print("=" * 70)
for i, col in enumerate(raw_feature_cols):
    print(f"  {i+1:2d}. {col}")

print(f"\nTotal raw features: {len(raw_feature_cols)}")
print(f"Total samples: {len(train_df)}")
print(f"Event rate: {train_df['event'].mean():.2%} ({train_df['event'].sum()} events out of {len(train_df)})")

# Feature engineering (same as model)
def engineer_features(df):
    df = df.copy()
    df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
    df['closing_x_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['urgency_ratio'] = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['eta_hours'] = np.where(df['closing_speed_m_per_h'] > 0, df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 1e-6), 999)
    df['eta_hours'] = df['eta_hours'].clip(0, 999)
    df['log_eta'] = np.log1p(df['eta_hours'])
    df['has_growth'] = (df['area_growth_abs_0_5h'] > 0).astype(int)
    df['growth_x_closing'] = df['area_growth_rate_ha_per_h'] * df['closing_speed_m_per_h']
    df['is_closing'] = (df['closing_speed_m_per_h'] > 0).astype(int)
    df['abs_dist_accel'] = np.abs(df['dist_accel_m_per_h2'])
    df['radial_x_alignment'] = df['radial_growth_rate_m_per_h'] * df['alignment_abs']
    df['is_night'] = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 6)).astype(int)
    df['is_weekend'] = (df['event_start_dayofweek'] >= 5).astype(int)
    df['is_summer'] = df['event_start_month'].isin([6, 7, 8]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['event_start_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['event_start_month'] / 12)
    df['perimeter_rate'] = df['num_perimeters_0_5h'] / (df['dt_first_last_0_5h'] + 0.1)
    df['dist_risk_close'] = (df['dist_min_ci_0_5h'] < 3000).astype(int)
    df['dist_risk_medium'] = ((df['dist_min_ci_0_5h'] >= 3000) & (df['dist_min_ci_0_5h'] < 10000)).astype(int)
    df['dist_risk_far'] = (df['dist_min_ci_0_5h'] >= 50000).astype(int)
    df['speed_x_dist'] = df['centroid_speed_m_per_h'] * df['dist_min_ci_0_5h']
    df['area_x_growth'] = df['area_first_ha'] * df['area_growth_rel_0_5h']
    df['along_track_urgency'] = df['along_track_speed'] / (df['dist_min_ci_0_5h'] + 1)
    df['proj_advance_ratio'] = df['projected_advance_m'] / (df['dist_min_ci_0_5h'] + 1)
    df['dist_min_sq'] = df['dist_min_ci_0_5h'] ** 2 / 1e10
    df['log_area_first'] = np.log1p(df['area_first_ha'])
    df['dist_growth_interaction'] = df['log_dist_min'] * df['has_growth']
    df['radial_to_dist'] = df['radial_growth_m'] / (df['dist_min_ci_0_5h'] + 1)
    df['directional_threat'] = df['alignment_abs'] * df['closing_speed_abs_m_per_h']
    df['intensity_proxy'] = np.log1p(df['area_first_ha'] * df['area_growth_rate_ha_per_h'])
    return df

train_eng = engineer_features(train_df)
feature_cols = [c for c in train_eng.columns if c not in target_cols]

X = train_eng[feature_cols].values
y_time = train_eng['time_to_hit_hours'].values
y_event = train_eng['event'].values.astype(bool)
y_surv = np.array(list(zip(y_event, y_time)), dtype=[('event', bool), ('time', float)])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a single GBSA model for feature importance
print("\n" + "=" * 70)
print("TRAINING GBSA FOR FEATURE IMPORTANCE")
print("=" * 70)

model = GradientBoostingSurvivalAnalysis(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_samples_split=5, subsample=0.8, random_state=42
)
model.fit(X_scaled, y_surv)

# Get feature importances
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE RANKING (Top 30)")
print("=" * 70)
print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Category'}")
print("-" * 80)

# Categorize features
def categorize_feature(name):
    distance_kw = ['dist', 'closing', 'projected_advance', 'along_track', 'radial_to_dist']
    speed_kw = ['speed', 'closing', 'urgency', 'eta', 'accel']
    area_kw = ['area', 'growth', 'intensity', 'perimeter']
    time_kw = ['hour', 'day', 'month', 'night', 'weekend', 'summer', 'temporal']
    direction_kw = ['alignment', 'bearing', 'cross_track', 'directional', 'radial_x']
    
    name_l = name.lower()
    
    # Check categories (order matters for overlapping keywords)
    if any(k in name_l for k in ['alignment', 'bearing', 'cross_track', 'directional_threat', 'radial_x_alignment']):
        return '[DIR] Direction'
    if any(k in name_l for k in ['eta', 'urgency', 'along_track_urgency', 'proj_advance_ratio']):
        return '[URG] Urgency'
    if any(k in name_l for k in ['dist', 'closing_speed', 'closing_x', 'projected_advance']):
        return '[DST] Distance/Closing'
    if any(k in name_l for k in ['speed', 'accel', 'centroid']):
        return '[SPD] Speed'
    if any(k in name_l for k in ['area', 'growth', 'intensity', 'perimeter', 'has_growth']):
        return '[FIR] Fire Size/Growth'
    if any(k in name_l for k in ['hour', 'day', 'month', 'night', 'weekend', 'summer', 'event_start']):
        return '[TMP] Temporal'
    if any(k in name_l for k in ['log1p', 'log_area', 'log_dist', 'log_eta']):
        return '[TRN] Transformed'
    if 'low_temporal' in name_l or 'num_perimeters' in name_l or 'dt_first' in name_l:
        return '[OBS] Observation'
    if 'fit_r2' in name_l:
        return '[FIT] Model Fit'
    return '[OTH] Other'

for rank, idx in enumerate(sorted_idx[:30]):
    cat = categorize_feature(feature_cols[idx])
    bar = '#' * int(importances[idx] / importances[sorted_idx[0]] * 30)
    print(f"  {rank+1:<4} {feature_cols[idx]:<40} {importances[idx]:<12.6f} {cat}")
    print(f"       {bar}")

# Category-level aggregation
print("\n" + "=" * 70)
print("CATEGORY-LEVEL IMPORTANCE (aggregated)")
print("=" * 70)

cat_importance = {}
for idx in range(len(feature_cols)):
    cat = categorize_feature(feature_cols[idx])
    cat_importance[cat] = cat_importance.get(cat, 0) + importances[idx]

total_imp = sum(cat_importance.values())
for cat, imp in sorted(cat_importance.items(), key=lambda x: -x[1]):
    pct = imp / total_imp * 100
    bar = '#' * int(pct / 2)
    print(f"  {cat:<25} {pct:5.1f}%  {bar}")

# Correlation analysis with event
print("\n" + "=" * 70)
print("CORRELATION WITH EVENT (Top 20 by absolute correlation)")
print("=" * 70)

correlations = {}
for col in feature_cols:
    correlations[col] = train_eng[col].corr(train_eng['event'])

sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for rank, (col, corr) in enumerate(sorted_corr[:20]):
    direction = "[+]" if corr > 0 else "[-]"
    bar = '#' * int(abs(corr) * 50)
    print(f"  {rank+1:<4} {col:<40} {corr:+.4f} {direction}")
    print(f"       {bar}")

# Simple stats for key features
print("\n" + "=" * 70)
print("KEY FEATURE STATISTICS (Event=1 vs Event=0)")
print("=" * 70)

key_features = ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_first_ha', 
                'area_growth_rate_ha_per_h', 'alignment_abs', 'radial_growth_rate_m_per_h',
                'centroid_speed_m_per_h', 'num_perimeters_0_5h', 'event_start_month']

events = train_df[train_df['event'] == 1]
no_events = train_df[train_df['event'] == 0]

print(f"{'Feature':<35} {'Event=1 (mean)':<18} {'Event=0 (mean)':<18} {'Ratio':<10}")
print("-" * 85)
for feat in key_features:
    mean_event = events[feat].mean()
    mean_no_event = no_events[feat].mean()
    ratio = mean_event / (mean_no_event + 1e-10)
    print(f"  {feat:<33} {mean_event:<18.2f} {mean_no_event:<18.2f} {ratio:<10.2f}")

print("\n[DONE] Analysis complete!")
