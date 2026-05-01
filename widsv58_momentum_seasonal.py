import pandas as pd
import numpy as np
import os

# WiDS Global Datathon 2026 — v58 "Momentum Seasonal"
# Building directly on the 0.97488 SOTA (wids50)

# 1. LOAD DATA
DATA_DIR = "c:/Users/parth/Downloads/WIDS_CHALLENGE/Wids2026"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

# Seasonal Climatology 
seasonal_hazard = {
    1: 0.05, 2: 0.06, 3: 0.12, 4: 0.25, 5: 0.45, 6: 0.75,
    7: 0.95, 8: 0.92, 9: 0.65, 10: 0.35, 11: 0.15, 12: 0.08
}

# Constants 
ORACLE_DIST = 5400
HORIZONS = [12, 24, 48, 72]
FLOOR, CEIL = 1e-15, 0.999999999

# 2. DEFINITIVE PHYSICS STRATA
def get_masks(df):
    d, p = df["dist_min_ci_0_5h"].values, df["num_perimeters_0_5h"].values
    far, close = d >= ORACLE_DIST, d < ORACLE_DIST
    return far, (close & (p==1)), (close & (p==2)), (close & (p>=3))

te_far, te_p1, te_p2, te_p3p = get_masks(test)
tr_masks = get_masks(train)
strata_names = ["far", "p1", "p2", "p3p"]

# 3. ANCHOR TARGETS (Empirical Training Distribution)
targets = {}
for i, name in enumerate(strata_names):
    mask = tr_masks[i]
    if not mask.any():
        targets[name] = {h: 0.0 for h in HORIZONS}
        continue
    targets[name] = {h: (train[mask & (train['event']==1)]['time_to_hit_hours'] <= h).mean() for h in HORIZONS}

# PHYSICS LOCK
targets["p3p"] = {h: CEIL for h in HORIZONS}
targets["p2"][48] = CEIL

# 4. THE RANKING IDENTITY: KINETIC + SEASONAL MOMENTUM
def get_momentum_rank(df):
    log_mass = np.log1p(df['area_first_ha'])
    velocity = (df['closing_speed_m_per_h'] + df['radial_growth_rate_m_per_h']).clip(0.1)
    dist_inv = 1.0 / (df['dist_min_ci_0_5h'].clip(1)**0.5)
    
    # Peak burn multiplier
    solar_factor = np.where(df['event_start_hour'].between(12, 18), 1.25, 1.0)
    
    # V58 UPGRADE: Seasonal Multiplier
    season_factor = df['event_start_month'].map(seasonal_hazard).fillna(0.5)
    # We add 0.5 to the season factor so it acts as a soft dampener/booster (0.55 to 1.45)
    season_multiplier = season_factor + 0.5 
    
    return (log_mass * velocity * dist_inv) * solar_factor * season_multiplier

test['rank_score'] = get_momentum_rank(test)

# 5. INITIAL ASSEMBLY & DYNAMIC POLARIZATION
sub = pd.DataFrame({'event_id': test['event_id']})

def lagrangian_snap(values, target):
    v = np.array(values).copy()
    if len(v) == 0: return v
    for _ in range(100):
        actual = np.mean(v)
        diff = target - actual
        if abs(diff) < 1e-18: break
        v = np.clip(v + diff, FLOOR, CEIL)
    return v

# Dynamic Polarization params: tighter for 12h, wider for 48h
spreads = {12: 0.22, 24: 0.28, 48: 0.35}

for h in [12, 24, 48]:
    col = f"prob_{h}h"
    h_probs = np.zeros(len(test))
    h_probs[te_far] = FLOOR
    
    for name, mask in zip(["p1", "p2", "p3p"], [te_p1, te_p2, te_p3p]):
        if not mask.any(): continue
        t_val = targets[name][h]
        
        scores = test.loc[mask, 'rank_score']
        z = (scores - scores.mean()) / (scores.std() + 1e-9)
        
        # V58 UPGRADE: Dynamic horizon-based spread
        polarized = t_val + (np.sign(z) * np.abs(z)**1.6 * spreads[h])
        
        h_probs[mask] = lagrangian_snap(polarized, t_val)
    sub[col] = h_probs

# 6. THE 72H CENSORSHIP HACK
sub["prob_72h"] = CEIL
sub.loc[te_far, "prob_72h"] = FLOOR * 2
sub.loc[te_far, ["prob_12h", "prob_24h", "prob_48h"]] = FLOOR

# 7. THE 250-CYCLE EQUILIBRIUM SOLVER
for _ in range(250):
    for i in range(len(sub)):
        if te_far[i]: continue
        for j in range(1, 4):
            p_c, c_c = f"prob_{HORIZONS[j-1]}h", f"prob_{HORIZONS[j]}h"
            if sub.loc[i, c_c] < sub.loc[i, p_c]:
                sub.loc[i, c_c] = sub.loc[i, p_c] + 1e-15 
    
    for h in [12, 24, 48]:
        col = f"prob_{h}h"
        for name, mask in zip(["p1", "p2", "p3p"], [te_p1, te_p2, te_p3p]):
            if mask.any():
                sub.loc[mask, col] = lagrangian_snap(sub.loc[mask, col].values, targets[name][h])

# 8. FINAL SAVE
sub.iloc[:, 1:] = sub.iloc[:, 1:].clip(FLOOR, CEIL).round(12)
out_path = f"{DATA_DIR}/submission_v58_momentum_seasonal.csv"
sub.to_csv(out_path, index=False)
print(f"SUCCESS! Submission V58 Ready: {out_path}")
