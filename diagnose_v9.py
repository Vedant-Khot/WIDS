"""
WiDS v9 Diagnostic Script
Run this BEFORE resubmitting to understand what hurt the score.
Requires: train.csv in the same directory.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

seasonal_data = {
    1:  (0.05, 12, 65), 2:  (0.06, 14, 60), 3:  (0.12, 18, 55),
    4:  (0.25, 22, 45), 5:  (0.45, 28, 35), 6:  (0.75, 32, 25),
    7:  (0.95, 35, 20), 8:  (0.92, 34, 22), 9:  (0.65, 29, 30),
    10: (0.35, 23, 40), 11: (0.15, 17, 50), 12: (0.08, 13, 60)
}

# ── Two versions of the feature engine ──────────────────────────────────────

def physics_engine_v8(df):
    d = df.copy()
    eps = 1e-6
    d['seasonal_hazard'] = d['event_start_month'].map(lambda x: seasonal_data[x][0])
    d['seasonal_temp']   = d['event_start_month'].map(lambda x: seasonal_data[x][1])
    d['seasonal_hum']    = d['event_start_month'].map(lambda x: seasonal_data[x][2])
    d['seasonal_dryness'] = 100 - d['seasonal_hum']
    d['dist_sigmoid'] = 1 / (1 + np.exp((d['dist_min_ci_0_5h'] - 4837) / 200))
    d['log_dist'] = np.log1p(d['dist_min_ci_0_5h'])
    d['total_speed'] = d['closing_speed_m_per_h'] + d['radial_growth_rate_m_per_h']
    dist_to_buffer = np.maximum(0, d['dist_min_ci_0_5h'] - 5000)
    d['tti_estimate'] = np.where(d['total_speed'] > 0.5, dist_to_buffer / (d['total_speed'] + eps), 999)
    d['tti_log'] = np.log1p(d['tti_estimate'])
    d['log_area'] = np.log1p(d['area_first_ha'])
    d['danger_momentum'] = d['total_speed'] * d['alignment_abs']
    d['radius_vs_dist'] = d['radial_growth_m'] / (d['dist_min_ci_0_5h'] + eps)
    d['bearing_toward_evac'] = d['alignment_abs'] * d['alignment_cos']
    d['aligned_closing'] = d['alignment_cos'] * d['closing_speed_m_per_h']
    d['is_morning']   = ((d['event_start_hour'] >= 6)  & (d['event_start_hour'] <= 12)).astype(int)
    d['is_afternoon'] = ((d['event_start_hour'] > 12)  & (d['event_start_hour'] <= 18)).astype(int)
    d['is_night']     = ((d['event_start_hour'] >= 22) | (d['event_start_hour'] <= 5)).astype(int)
    return d

def physics_engine_v9(df):
    d = physics_engine_v8(df)
    eps = 1e-6
    dist_to_buffer = np.maximum(0, d['dist_min_ci_0_5h'] - 5000)
    d['fire_retreating']  = (d['dist_slope_ci_0_5h'] > 0).astype(int)
    d['fire_approaching'] = (d['dist_slope_ci_0_5h'] < 0).astype(int)
    d['approach_rate']    = -d['dist_slope_ci_0_5h']
    d['retreat_x_dist']   = d['fire_retreating'] * d['dist_min_ci_0_5h']
    d['approach_x_speed'] = d['fire_approaching'] * d['total_speed']
    effective_speed = np.maximum(d['total_speed'] + d['approach_rate'], eps)
    d['tti_slope_adjusted'] = np.where(effective_speed > 0.5, dist_to_buffer / effective_speed, 999)
    d['tti_slope_log'] = np.log1p(d['tti_slope_adjusted'])
    return d

FEATURES_V8 = [
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

FEATURES_V9 = FEATURES_V8 + [
    'fire_retreating', 'fire_approaching', 'approach_rate',
    'retreat_x_dist', 'approach_x_speed',
    'tti_slope_adjusted', 'tti_slope_log',
]

# ── Quick CV to compare approaches ──────────────────────────────────────────

def quick_cv_brier(train_fe, features, horizon, n_folds=3, seed=0,
                   separate_calibration=True, label=""):
    df_h = train_fe.copy()
    unknown_mask = (df_h['event'] == 0) & (df_h['time_to_hit_hours'] < horizon)
    df_h = df_h[~unknown_mask].reset_index(drop=True)
    y_h = ((df_h['event'] == 1) & (df_h['time_to_hit_hours'] <= horizon)).astype(int).values

    if len(np.unique(y_h)) < 2:
        print(f"  [{label}] {horizon}h: single class, skipping")
        return np.nan

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(df_h))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_h, y_h)):
        df_tr, df_val = df_h.iloc[tr_idx], df_h.iloc[val_idx]
        X_tr  = df_tr[features].fillna(0).values
        y_tr  = y_h[tr_idx]
        X_val = df_val[features].fillna(0).values

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        cb = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03,
                                verbose=0, random_seed=seed)
        cb.fit(X_tr_s, y_tr)
        p_cat = cb.predict_proba(X_val_s)[:, 1]

        lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4,
                              num_leaves=12, random_state=seed, verbosity=-1)
        lgbm.fit(X_tr_s, y_tr)
        p_lgbm = lgbm.predict_proba(X_val_s)[:, 1]

        if separate_calibration:
            ir_c = IsotonicRegression(out_of_bounds='clip')
            ir_l = IsotonicRegression(out_of_bounds='clip')
            r_cat_tr  = pd.Series(cb.predict_proba(X_tr_s)[:, 1]).rank(pct=True).values
            r_lgbm_tr = pd.Series(lgbm.predict_proba(X_tr_s)[:, 1]).rank(pct=True).values
            ir_c.fit(r_cat_tr, y_tr)
            ir_l.fit(r_lgbm_tr, y_tr)
            r_cat_val  = pd.Series(p_cat).rank(pct=True).values
            r_lgbm_val = pd.Series(p_lgbm).rank(pct=True).values
            blended = 0.55 * ir_c.transform(r_cat_val) + 0.45 * ir_l.transform(r_lgbm_val)
        else:
            r_cat   = pd.Series(p_cat).rank(pct=True).values
            r_lgbm  = pd.Series(p_lgbm).rank(pct=True).values
            blended_rank = 0.55 * r_cat + 0.45 * r_lgbm
            r_blend_tr = (
                0.55 * pd.Series(cb.predict_proba(X_tr_s)[:, 1]).rank(pct=True).values
              + 0.45 * pd.Series(lgbm.predict_proba(X_tr_s)[:, 1]).rank(pct=True).values
            )
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(r_blend_tr, y_tr)
            blended = ir.transform(blended_rank)

        oof_probs[val_idx] = blended

    bs = brier_score_loss(y_h, oof_probs)
    print(f"  [{label}] {horizon}h | Brier: {bs:.5f} | Hit rate: {y_h.mean():.2%}")
    return bs


# ── Run diagnostics ──────────────────────────────────────────────────────────

print("=" * 60)
print("DIAGNOSTIC: Isolating what changed between v8 and v9")
print("=" * 60)

train_v8 = physics_engine_v8(train)
train_v9 = physics_engine_v9(train)

print("\n[1] dist_slope feature quality check")
col = 'dist_slope_ci_0_5h'
if col in train.columns:
    v = train[col].dropna()
    pct_positive = (v > 0).mean()
    pct_zero     = (v == 0).mean()
    print(f"  mean={v.mean():.4f}  std={v.std():.4f}")
    print(f"  pct retreating (>0): {pct_positive:.1%}")
    print(f"  pct exactly zero:    {pct_zero:.1%}")
    if pct_zero > 0.4:
        print("  ⚠️  >40% zero — likely a low-resolution feature adding noise.")

print("\n[2] Correlation of new v9 features with 12h label")
y_12 = ((train['event'] == 1) & (train['time_to_hit_hours'] <= 12)).astype(int)
new_feats = ['fire_retreating','fire_approaching','approach_rate',
             'retreat_x_dist','approach_x_speed','tti_slope_adjusted','tti_slope_log']
for f in new_feats:
    if f in train_v9.columns:
        corr = train_v9[f].fillna(0).corr(y_12)
        flag = "⚠️  weak" if abs(corr) < 0.05 else "✅ useful"
        print(f"  {f:30s}: corr = {corr:+.4f}  {flag}")

print("\n[3] CV Brier — 4 combinations across all horizons")
print("    Lower Brier = better score")
print("-" * 60)

results = {}
for h in [12, 24, 48, 72]:
    print(f"\n  Horizon {h}h:")
    bs_v8      = quick_cv_brier(train_v8, FEATURES_V8, h, separate_calibration=False, label="v8-feats v8-cal")
    bs_new_f   = quick_cv_brier(train_v9, FEATURES_V9, h, separate_calibration=False, label="v9-feats v8-cal")
    bs_new_c   = quick_cv_brier(train_v8, FEATURES_V8, h, separate_calibration=True,  label="v8-feats v9-cal")
    bs_new_fc  = quick_cv_brier(train_v9, FEATURES_V9, h, separate_calibration=True,  label="v9-feats v9-cal")
    results[h] = dict(v8=bs_v8, new_feats=bs_new_f, new_cal=bs_new_c, both=bs_new_fc)

print("\n" + "=" * 60)
print("SUMMARY TABLE  (▼ = better than v8 baseline, ▲ = worse)")
print("=" * 60)
print(f"{'H':>4} | {'v8 base':>9} | {'v9 feats':>9} | {'v9 cal':>9} | {'v9 both':>9}")
print("-" * 52)
for h, r in results.items():
    base = r['v8']
    def fmt(v):
        if np.isnan(v): return "   n/a   "
        d = v - base
        sym = "▼" if d < 0 else "▲"
        return f"{v:.5f}{sym}"
    print(f"{h:>3}h | {base:.5f}  | {fmt(r['new_feats']):>9} | {fmt(r['new_cal']):>9} | {fmt(r['both']):>9}")

print("\nAction:")
print("  If v9-feats ▲ → remove dist_slope interaction features.")
print("  If v9-cal   ▲ → revert to single blended-rank calibration.")
print("  Use whichever combo gives ▼ on the most horizons.")
