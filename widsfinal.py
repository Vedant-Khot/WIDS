import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Load your best previous submission (v31) to use its ranking (C-index)
# We trust the RANKING of the old model, but we will fix its CALIBRATION.
v31 = pd.read_csv("submission_v31.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Constants from EDA
CRITICAL_DIST = 5400

def get_group(df):
    # Defining the 4 groups that dictate fire behavior
    mask_close = df['dist_min_ci_0_5h'] < CRITICAL_DIST
    p1 = (df['num_perimeters_0_5h'] == 1) & mask_close
    p2 = (df['num_perimeters_0_5h'] == 2) & mask_close
    p3 = (df['num_perimeters_0_5h'] >= 3) & mask_close
    far = df['dist_min_ci_0_5h'] >= CRITICAL_DIST
    return p1, p2, p3, far

# 1. Calculate Empirical Targets from Train
p1_tr, p2_tr, p3_tr, far_tr = get_group(train)
horizons = [12, 24, 48, 72]
group_targets = { "p1": {}, "p2": {}, "p3": {}, "far": {} }

for h in horizons:
    for name, mask in [("p1", p1_tr), ("p2", p2_tr), ("p3", p3_tr), ("far", far_tr)]:
        # Calculate exactly what % of this group hit by horizon H
        hits = train[mask & (train['event'] == 1) & (train['time_to_hit_hours'] <= h)]
        group_targets[name][h] = len(hits) / mask.sum()

# 2. Apply "Mean Centering" to the Test Set
# We take the probabilities from v31 and scale them so the group mean matches Train
test_p1, test_p2, test_p3, test_far = get_group(test)
v32 = v31.copy()

for h in [12, 24, 48]: # 72h is handled separately
    col = f"prob_{h}h"
    
    for name, mask in [("p1", test_p1), ("p2", test_p2), ("p3", test_p3)]:
        if mask.sum() == 0: continue
        
        target = group_targets[name][h]
        current_preds = v32.loc[mask, col].values
        
        # Rank-Preserving Scaling:
        # This keeps the order of fires (C-index) but shifts the mean (Brier)
        if current_preds.mean() > 0:
            scaling_factor = target / current_preds.mean()
            v32.loc[mask, col] = np.clip(current_preds * scaling_factor, 0, 0.99)

# 3. The Far-Fire Oracle Rule
# Training data shows 0% of Far fires hit by 48h. 
# We set them to absolute 0 (or very close)
far_mask = test['dist_min_ci_0_5h'] >= CRITICAL_DIST
for h in [12, 24, 48]:
    v32.loc[far_mask, f"prob_{h}h"] = 0.0001

# 4. The 72h Horizon Rule (The "WiDS Special")
# Rule: If it's close, it's hitting. If it's far, it's censored.
# Censored fires don't count at 72h. So we predict 1.0 for all close fires.
close_mask = test['dist_min_ci_0_5h'] < CRITICAL_DIST
v32.loc[close_mask, "prob_72h"] = 0.999
v32.loc[far_mask, "prob_72h"] = 0.005 # Stays low just in case

# 5. Physics Consistency (Monotonicity)
# Probabilities must increase over time
iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
for i in range(len(v32)):
    row = v32.iloc[i, 1:].values.astype(float)
    if not np.all(np.diff(row) >= 0):
        v32.iloc[i, 1:] = iso.fit_transform([12, 24, 48, 72], row)

# 6. Final "Squeeze"
# Extremely high/low values help Brier if you are right.
# We trust the distance oracle, so we push probabilities near the edges.
def squeeze(x):
    if x > 0.95: return 0.999
    if x < 0.01: return 0.0001
    return x

for h in horizons:
    v32[f"prob_{h}h"] = v32[f"prob_{h}h"].apply(squeeze)

v32.to_csv("submission_v32_99target.csv", index=False)