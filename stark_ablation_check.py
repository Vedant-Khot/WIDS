import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('train.csv')

# Use common horizon for test (72h)
H = 72
y = ((train['event'] == 1) & (train['time_to_hit_hours'] <= H)).astype(int).values

# Simple Feature Engineering (subset)
train['seasonal_hazard'] = train['event_start_month'].map({1:0.05, 2:0.06, 3:0.12, 4:0.25, 5:0.45, 6:0.75, 7:0.95, 8:0.92, 9:0.65, 10:0.35, 11:0.15, 12:0.08})
FEATURES = ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'seasonal_hazard', 'event_start_hour']

# Fold 0 Split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, val_idx = next(skf.split(train, y))

X_tr, y_tr = train.iloc[tr_idx][FEATURES], y[tr_idx]
X_val, y_val = train.iloc[val_idx][FEATURES], y[val_idx]
val_orig = train.iloc[val_idx].copy()

# Train simple model
model = CatBoostClassifier(iterations=200, depth=4, verbose=0)
model.fit(X_tr, y_tr)
probs_val = model.predict_proba(X_val)[:, 1]

# Baseline Score
brier_base = brier_score_loss(y_val, probs_val)

# STARK GATES
V_MAX_BASE = 420.0
SAFETY_MARGIN = 1.15
v_max_h = V_MAX_BASE * (val_orig['seasonal_hazard'].values * 0.8 + 0.2)
limit = v_max_h * H * SAFETY_MARGIN

probs_stark = probs_val.copy()
# Zeroing
zero_mask = val_orig['dist_min_ci_0_5h'].values > limit
probs_stark[zero_mask] = 0.0

# Maxing
max_mask = (val_orig['dist_min_ci_0_5h'].values < 500) & (val_orig['closing_speed_m_per_h'].values > 0)
probs_stark[max_mask] = 0.9999

# Post-Gate Score
brier_stark = brier_score_loss(y_val, probs_stark)

print(f"Brier Baseline ({H}h): {brier_base:.6f}")
print(f"Brier Stark    ({H}h): {brier_stark:.6f}")
print(f"Brier Delta: {brier_base - brier_stark:.6f} (Positive is good)")

# Near-fire vs Far-fire split
far_mask = val_orig['dist_min_ci_0_5h'] > limit
near_mask = val_orig['dist_min_ci_0_5h'] < 1000

if far_mask.any():
    print(f"Far-fire Brier Delta: {brier_score_loss(y_val[far_mask], probs_val[far_mask]) - brier_score_loss(y_val[far_mask], probs_stark[far_mask]):.6f}")
if near_mask.any():
    print(f"Near-fire Brier Delta: {brier_score_loss(y_val[near_mask], probs_val[near_mask]) - brier_score_loss(y_val[near_mask], probs_stark[near_mask]):.6f}")
