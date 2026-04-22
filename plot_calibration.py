import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('train.csv')
H = 24
y = ((train['event'] == 1) & (train['time_to_hit_hours'] <= H)).astype(int).values
train['seasonal_hazard'] = train['event_start_month'].map({1:0.05, 2:0.06, 3:0.12, 4:0.25, 5:0.45, 6:0.75, 7:0.95, 8:0.92, 9:0.65, 10:0.35, 11:0.15, 12:0.08})
FEATURES = ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'seasonal_hazard', 'event_start_hour']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, val_idx = next(skf.split(train, y))
X_tr, y_tr = train.iloc[tr_idx][FEATURES], y[tr_idx]
X_val, y_val = train.iloc[val_idx][FEATURES], y[val_idx]
val_orig = train.iloc[val_idx].copy()

# Baseline model
model = CatBoostClassifier(iterations=200, depth=4, verbose=0)
model.fit(X_tr, y_tr)
probs_v6 = model.predict_proba(X_val)[:, 1]

# v7 Logic (Stark Gates)
p_v7 = probs_v6.copy()
V_MAX_BASE = 420.0
v_max_h = V_MAX_BASE * (val_orig['seasonal_hazard'].values * 0.8 + 0.2)
limit = v_max_h * H * 1.15
p_v7[val_orig['dist_min_ci_0_5h'].values > limit] = 0.0
p_v7[(val_orig['dist_min_ci_0_5h'].values < 500) & (val_orig['closing_speed_m_per_h'].values > 5)] = 0.9999

# Calibration curve
prob_true_v6, prob_pred_v6 = calibration_curve(y_val, probs_v6, n_bins=10)
prob_true_v7, prob_pred_v7 = calibration_curve(y_val, p_v7, n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot(prob_pred_v6, prob_true_v6, 's-', label='v6 (Baseline)', color='gray', alpha=0.5)
plt.plot(prob_pred_v7, prob_true_v7, 'o-', label='v7 (Stark Gates)', color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Calibration Curve Comparison ({H}h)')
plt.legend()
plt.grid(True)
plt.savefig('calibration_comparison.png')
print("Calibration plot saved as calibration_comparison.png")
