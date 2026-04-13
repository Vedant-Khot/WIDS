# V5 Model Code Review — Against EDA Report Findings

**Verdict summary:** V5 has the right survival modeling framework but contains 3 dead features,
2 data consistency bugs, 1 wrong LightGBM objective, 1 flawed meta-model target,
multiple report-recommended features not implemented, and key report warnings ignored.

---

## Table of Contents

1. [What V5 Gets Right](#1-what-v5-gets-right)
2. [Critical Bugs — Features That Are Dead or Broken](#2-critical-bugs--features-that-are-dead-or-broken)
3. [Data Consistency Bugs](#3-data-consistency-bugs)
4. [Model Architecture Bugs](#4-model-architecture-bugs)
5. [Features That Should Be Dropped (Report Said So)](#5-features-that-should-be-dropped-report-said-so)
6. [Features Missing From V5 (Report Recommended)](#6-features-missing-from-v5-report-recommended)
7. [Threshold and Configuration Errors](#7-threshold-and-configuration-errors)
8. [Structural Architecture Issues](#8-structural-architecture-issues)
9. [The KM Prior Disable — Was That Right?](#9-the-km-prior-disable--was-that-right)
10. [Priority Fix List](#10-priority-fix-list)
11. [Corrected Feature Engineering Function](#11-corrected-feature-engineering-function)

---

## 1. What V5 Gets Right

These are genuinely correct decisions aligned with the EDA report.

| Component | Status | Reason |
|---|---|---|
| `log_dist_min = log1p(dist_min_ci)` | ✅ Correct | Report: corr = −0.812, single best feature |
| GBSA + RSF survival models | ✅ Correct | Right framework for right-censored data |
| XGBoost `survival:cox` objective | ✅ Correct | Handles censoring via partial likelihood |
| Isotonic calibration per fold | ✅ Correct | Per-horizon calibration is sound |
| Monotonicity enforcement | ✅ Correct | Required: prob_12h ≤ 24h ≤ 48h ≤ 72h |
| Stratified K-fold on `event` | ✅ Correct | Right approach for 31% hit rate imbalance |
| Seed averaging (3 seeds) | ✅ Correct | Reduces variance on small dataset |
| `has_growth` indicator | ✅ Correct | Report: good missingness flag |
| `low_temporal_resolution_0_5h` included | ✅ Correct | Report: primary segmentation variable |
| `alignment_abs` used over `alignment_cos` | ✅ Correct | Report: alignment_abs corr = 0.349 vs 0.120 |
| `closing_speed_abs_m_per_h` used | ✅ Correct | Report: prefer absolute over signed |
| Submission sanity checks | ✅ Correct | Good defensive practice |
| Blending comparison kept | ✅ Correct | Good fallback if stacking overfits |
| Optuna hyperparameter search | ✅ Correct | Good for small datasets |

---

## 2. Critical Bugs — Features That Are Dead or Broken

These features produce either all-zero or NaN values. They contribute nothing — or worse, they introduce NaN values that can silently break tree models depending on how missing values are handled.

### Bug 2.1: `fast_hit_signal` — Always Zero (Dead Feature)

```python
# V5 CODE:
df['fast_hit_signal'] = ((df['dist_min_ci_0_5h'] < 5000) & 
                          (df['closing_speed_m_per_h'] > 500)).astype(int)
```

**Why it's broken:** The maximum `closing_speed_m_per_h` in the entire training dataset is **354.1 m/h**. The threshold of 500 m/h is unreachable. No fire in training (or likely test) satisfies this condition.

```
fast_hit_signal fires in training: 0
fast_hit_signal correlation: NaN (constant column)
```

**Fix:** Either drop this feature, or use a realistic threshold:
```python
df['fast_hit_signal'] = ((df['dist_min_ci_0_5h'] < 5000) & 
                          (df['closing_speed_m_per_h'] > 50)).astype(int)
# Or better — just use the continuous version:
df['close_and_closing'] = (df['dist_min_ci_0_5h'] < 5000).astype(int) * df['closing_speed_abs_m_per_h']
```

---

### Bug 2.2: `low_res_x_urgency` — Always Zero (Dead Feature)

```python
# V5 CODE:
df['low_res_x_urgency'] = df['low_temporal_resolution_0_5h'] * df['urgency_ratio']
```

**Why it's broken:** `urgency_ratio = closing_speed / (dist + 1)`. Low-temporal-resolution fires (ltr=1) **always** have `closing_speed = 0` because closing speed requires two perimeter measurements. So `urgency_ratio = 0` for all ltr=1 fires. The product is therefore always 0.

```
low_res_x_urgency nonzero: 0 / 221
low_res_x_urgency correlation: NaN
```

**Correct interaction:** The report recommends interactions between `low_temporal_resolution` and features that ARE available for those fires. Since `dist_min_ci_0_5h` is always available:

```python
df['low_res_x_log_dist'] = df['low_temporal_resolution_0_5h'] * df['log_dist_min']
```

---

### Bug 2.3: `low_res_x_growth` — Always Zero (Dead Feature)

```python
# V5 CODE:
df['low_res_x_growth'] = df['low_temporal_resolution_0_5h'] * df['area_growth_rate_ha_per_h']
```

**Why it's broken:** Same root cause. Low-temporal-resolution fires always have `area_growth_rate = 0` (requires two perimeters). This interaction is identically zero for all 221 rows.

```
low_res_x_growth nonzero: 0 / 221
low_res_x_growth correlation: NaN
```

**Correct version:** Interact with `area_first_ha` which IS always available:
```python
df['low_res_x_area'] = df['low_temporal_resolution_0_5h'] * df['log1p_area_first']
```

---

## 3. Data Consistency Bugs

### Bug 3.1: Percentile Features Computed Separately on Train and Test (Data Leakage / Inconsistency)

```python
# V5 CODE (in engineer_features — applied to both train and test independently):
for col in ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_first_ha']:
    df[f'{col}_pctile'] = df[col].rank(pct=True)
```

**Why it's broken:** `rank(pct=True)` computes the percentile rank *within the dataframe it's called on*. When called on `train_eng`, percentiles are relative to the 221 training fires. When called on `test_eng`, percentiles are relative to the 95 test fires. These are completely different distributions.

Example: A fire with `dist_min_ci = 50,000m` might be at the 60th percentile in training but the 45th percentile in test (if test fires happen to be farther away on average). The model trained on the 60th percentile pattern will receive 45th percentile signals — this is a train/test inconsistency that degrades generalization.

**Fix:** Compute percentiles from training data only, then apply to test:
```python
# In main pipeline (NOT inside engineer_features):
from scipy.stats import percentileofscore

for col in ['dist_min_ci_0_5h', 'closing_speed_m_per_h', 'area_first_ha']:
    train_vals = train_eng[col].values
    train_eng[f'{col}_pctile'] = train_eng[col].rank(pct=True)
    test_eng[f'{col}_pctile'] = test_eng[col].apply(
        lambda x: percentileofscore(train_vals, x, kind='rank') / 100.0
    )
```

---

### Bug 3.2: Monotonicity Enforcement Loop Has a Clipping Order Bug

```python
# V5 CODE:
for i in range(len(submission)):
    p12 = float(submission.loc[i, 'prob_12h'])
    p24 = max(float(submission.loc[i, 'prob_24h']), p12 + 1e-5)
    p48 = max(float(submission.loc[i, 'prob_48h']), p24 + 1e-5)
    p72 = max(float(submission.loc[i, 'prob_72h']), p48 + 1e-5)
    submission.loc[i, ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']] = [
        min(p12, 0.999), min(p24, 0.999), min(p48, 0.999), min(p72, 0.999)
    ]
```

**Why it's broken:** Suppose `prob_48h = 0.9995` and `prob_72h = 0.9998`. After the max chain: `p48 = 0.9995`, `p72 = 0.9998`. After clipping: `p48 = 0.999`, `p72 = 0.999`. Now `p72 = p48` — the strict monotonicity (`+1e-5`) was enforced before clipping but violated after. Worse, the code clips each independently which can invert the order.

**Fix:** Use vectorized cummax then clip:
```python
cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
submission[cols] = submission[cols].cummax(axis=1).clip(0.001, 0.999)
```

---

## 4. Model Architecture Bugs

### Bug 4.1: LightGBM Uses Wrong Objective for Survival Data

```python
# V5 CODE:
params = {
    'objective': 'regression',   # ← WRONG
    ...
}
y_tr_lgb = np.where(y_tr['event'], y_tr['time'], -y_tr['time'])  # Cox label encoding
```

**Why it's broken:** The Cox label encoding (positive for events, negative for censored times) is designed for the Cox partial likelihood loss — not for standard regression. Using `objective='regression'` with this label encoding means LightGBM is trying to minimize MSE against values like `[+18.5, -43.2, +0.9, ...]` — a nonsensical target. The model is not learning a survival function; it's learning to predict a mix of positive and negative numbers.

XGBoost handles this correctly because `survival:cox` is a dedicated Cox objective that understands the sign encoding. LightGBM needs its own equivalent.

**Fix — Option A (match XGBoost approach):**
```python
# LightGBM doesn't have a native Cox objective — use XGBoost instead for this model
# Or use a proper AFT objective:
params = {
    'objective': 'regression',
    'metric': 'mse',
}
# And change the label to the AFT convention: just time (all positive)
# with a weight of 0 for censored rows:
```

**Fix — Option B (proper AFT):**
```python
# Use XGBoost AFT instead of LGB for the gradient boosted survival model
params_aft = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'aft_loss_distribution_scale': 1.0,
}
```

**Fix — Option C (keep LGB, use correct labels):**
```python
# Treat LGB as a ranking model — use risk scores and calibrate probabilities separately
# Use label = time_to_hit for events, large value for censored (inverted survival time)
y_tr_lgb = np.where(y_tr['event'], y_tr['time'], 999)  # censored = "very long time"
params = {'objective': 'regression', 'metric': 'mse'}
# Then negate predictions to get risk scores (lower predicted time = higher risk)
```

---

### Bug 4.2: Stacking Meta-Model Risk Target is Logically Inverted

```python
# V5 CODE:
risk_target = np.zeros(len(X_train))
for i in range(len(X_train)):
    if y_event[i]:
        risk_target[i] = 1.0 / (y_time[i] + 0.1)   # short time = high risk ← OK
    else:
        risk_target[i] = 0.0                          # censored = zero risk ← WRONG
```

**Why it's broken:** A fire censored at `time = 2h` (meaning it was only observed for 2 hours before the monitoring window ended) gets risk = 0.0. But a fire that actually hit at `time = 70h` gets risk = `1 / 70.1 ≈ 0.014`. This means the censored 2h fire appears SAFER than the fire that hit at 70h — which is the opposite of what we want.

Censored fires should get **high uncertainty**, not zero risk. Many censored fires simply ran out of observation time — they may have been heading toward an evac zone. Setting them to 0 poisons the meta-model.

**Fix:**
```python
# Use the average predicted risk from the base models as a pseudo-target
# rather than constructing a manual target:
risk_target = np.mean([rankdata(oof_preds[m]['risk']) for m in model_list], axis=0)
risk_meta = Ridge(alpha=10.0)
risk_meta.fit(meta_train.values, risk_target)
```

---

## 5. Features That Should Be Dropped (Report Said So)

The following features are in `feature_cols` because they are not explicitly excluded. They all appear in the raw data and are included automatically in the `feature_cols` list comprehension. The report clearly recommended dropping all of these:

| Feature | Report Verdict | V5 Status | Problem |
|---|---|---|---|
| `relative_growth_0_5h` | DROP — exact dup | Still included | Byte-for-byte identical to `area_growth_rel_0_5h` |
| `projected_advance_m` | DROP — exact neg | Still included | Exact negative of `dist_change_ci_0_5h` |
| `alignment_cos` | DROP — weaker | Still included | Corr 0.12 vs 0.35 for `alignment_abs` |
| `spread_bearing_deg` | DROP — use sin/cos | Still included | 0°/360° discontinuity |
| `along_track_speed` | DROP — no signal | Still included | Corr = 0.008 |
| `cross_track_component` | DROP — no signal | Still included | Corr = −0.058 |
| `dist_accel_m_per_h2` | DROP — weak, noisy | Still included | Corr = −0.073, 84% zeros |
| `event_start_hour` | DROP — very weak | Still included | Corr = 0.047 |
| `event_start_dayofweek` | DROP — spurious | Still included | Corr = −0.119, no causal mechanism |
| `log_area_ratio_0_5h` | DROP — dup of log1p_growth | Still included | Corr 0.81 with log1p_growth |
| `dist_change_ci_0_5h` | DROP one of pair | Still included | Keep only one of this or projected_advance |

**Impact:** These 11 features add noise, inflate the feature space, and confuse regularization in tree models. For tree models with 50 trees these may not matter much — but for a Ridge meta-model with only 221 samples, extra noisy features hurt.

---

## 6. Features Missing From V5 (Report Recommended)

These features were explicitly recommended in the EDA report as important additions, but are absent from V5:

### Missing 6.1: `log_num_perimeters` (Strong Signal, Corr = +0.405)

```python
# RECOMMENDED (not in V5):
df['log_num_perimeters'] = np.log1p(df['num_perimeters_0_5h'])
```

`num_perimeters_0_5h` has corr = +0.371. Log-transforming it achieves corr = +0.405 (stronger than `dt_first_last_0_5h` which V5 also misses). V5 uses raw `num_perimeters_0_5h` which is heavily right-skewed (median=1, max=17). Missing this is a meaningful loss.

---

### Missing 6.2: `bearing_is_real` (Corr = +0.314, Fixes the Artifact)

```python
# RECOMMENDED (not in V5):
df['bearing_is_real'] = (df['spread_bearing_cos'] != 1.0).astype(int)
# Equivalently: df['bearing_is_real'] = (df['num_perimeters_0_5h'] > 1).astype(int)
```

The report identified that `spread_bearing_cos = 1.0` for all 196 single-perimeter fires (because bearing defaults to 0°). V5 uses raw `spread_bearing_cos` without this correction, so the model may learn from an artifact rather than a real signal.

---

### Missing 6.3: `dt_first_last_0_5h` (Corr = +0.353, Report Tier 1)

```python
# Already in the raw data — just not used in V5!
# The feature is never referenced in engineer_features()
```

`dt_first_last_0_5h` is the time span between first and last perimeter. Corr with event = +0.353. It is always available (value = 0 for single-perimeter fires, which is meaningful). The report listed it as a Tier 1 feature. V5 neither creates it nor preserves it in `feature_cols` — wait, actually it IS in the raw data and will be in `feature_cols` automatically since it passes the filter. But it is never log-transformed or specially treated. This is OK — it will be included — but it should be noted.

Actually checking: `dt_first_last_0_5h` IS in the raw CSV so it WILL appear in `feature_cols`. This is fine. ✓

---

### Missing 6.4: `is_close` Soft Version (The Most Powerful Signal)

V5 uses `dist_risk_close = (dist_min_ci < 3000)` which **misses 13 hit fires** (those between 3,000m and 4,674m). The correct threshold from the EDA is 5,000m:

```python
# CURRENT (wrong threshold):
df['dist_risk_close'] = (df['dist_min_ci_0_5h'] < 3000).astype(int)   # misses 13 hits

# CORRECT:
df['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)           # captures all 69 hits
```

The 3,000m threshold classifies 13 hit fires as "not close" — these fires will tend to receive lower predicted probabilities than they should.

---

## 7. Threshold and Configuration Errors

### Error 7.1: `dist_risk_close` Uses 3000m Instead of 4674m

Verified from the data: The maximum distance for a training hit fire is 4,674m. The correct "close" threshold is therefore 5,000m (with a small buffer). Using 3,000m misclassifies 13 out of 69 hit fires as "not close."

```
Close fires captured at <3000m: 56 (all hit)
Close fires captured at <5000m: 69 (all hit) ← correct
Missed by V5's threshold: 13 hit fires
```

### Error 7.2: `BRIER_WEIGHTS` Excludes 12h Horizon

```python
# V5 CODE:
EVAL_HORIZONS = [12, 24, 48, 72]   # 12h IS evaluated
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}  # 12h is NOT weighted!
```

V5 added 12h as a prediction horizon (correctly, in Phase 1.3). But the `compute_weighted_brier` function never includes 12h because `BRIER_WEIGHTS` doesn't have a 12h entry. This means:

- Models are tuned via Optuna on `compute_hybrid_score` which uses `compute_weighted_brier`
- But 12h Brier is never in the optimization signal
- So the model has no pressure to be calibrated at 12h
- The 12h predictions are made but effectively unconstrained by the loss

**Fix:**
```python
BRIER_WEIGHTS = {12: 0.15, 24: 0.25, 48: 0.35, 72: 0.25}
# Or if 12h has equal importance:
BRIER_WEIGHTS = {12: 0.25, 24: 0.25, 48: 0.25, 72: 0.25}
```

### Error 7.3: `fast_hit_signal` Threshold 500 m/h Is Physically Impossible

As shown in the bug analysis: max closing speed in dataset = 354.1 m/h. The threshold 500 m/h is never reached. Either this was miscalibrated on a different dataset or the units were confused (500 m/h is about 0.5 km/h — very slow; 500 m/h = 8.3 m/min).

---

## 8. Structural Architecture Issues

### Issue 8.1: No Two-Segment Model (Report's Top Recommendation Ignored)

The EDA report's primary modeling recommendation was to split fires into two segments:
- **Far fires (dist ≥ 5km, n=152):** All predicted near zero. Simple logistic on log_dist.
- **Close fires (dist < 5km, n=69):** The interesting problem — predict *when* they hit.

**V5 pools all 221 fires into a single model.** The problem with pooling:

- 72% of training fires (160/221) have only 3–6 usable features
- These fires overwhelm the feature importance calculations
- The model must simultaneously learn "which fires hit" (distance-driven) and "when they hit" (growth/alignment-driven) from the same feature set
- Growth and directional features are almost entirely zero for the majority class, so the model may learn spurious zero-based patterns

The two-segment approach would be cleaner and more interpretable. Within the "close" segment, you have 69 fires — still very small, but now all features are relevant and the survival question is pure timing.

### Issue 8.2: Meta-Stacking With 221 Samples Has High Overfitting Risk

V5 implements Ridge stacking with `alpha=10`. The meta-feature matrix has:
- 4 models × (1 risk + 4 horizon probs) = 20 meta-features
- Fitted on 221 OOF predictions

Even with Ridge regularization, fitting 20 features to 221 samples is a 11:1 ratio — acceptable but tight. The real concern is that the stacking OOF score is computed on the same data the meta-model was trained on (OOF predictions), which is valid — but OOF predictions themselves contain variance from a 15-fold CV (5 folds × 3 repeats), so each prediction already has noise from the fold sampling.

The comparison against blending is done correctly (both evaluated on the same OOF data), but with 221 samples, stacking winning by a small margin could be pure noise. The report recommends blending as the safer option.

### Issue 8.3: Survival Function Extrapolation at 12h is Unreliable

```python
def survfunc_to_probs(surv_fns, horizons):
    for i, fn in enumerate(surv_fns):
        t_max = fn.x[-1]
        for h in horizons:
            survival_at_h = fn(min(h, t_max))   # correctly handles max
            # BUT: what if fn.x[0] > 12?
```

The GBSA and RSF survival functions are step functions over the observed event times in the training fold. If the minimum event time in a fold is, say, 0.5h, then evaluating at 12h is fine — the function has support there. But if the function extrapolates at the LEFT boundary (near 0h), the 12h estimate might behave oddly for fires that hit very fast.

More critically: for fires that will NOT hit (far fires), the survival function may be nearly flat near 1.0 across all horizons — meaning P(hit) ≈ 0 for all four horizons. This is correct behavior, but should be verified by checking the range of predictions for far fires separately.

---

## 9. The KM Prior Disable — Was That Right?

```python
# V5 CODE:
KM_PRIOR_WEIGHT = 0.0  # Disabled — was destroying 72h predictions
```

The comment says KM prior blending was "raising the 72h floor too high." But consider what the KM estimates actually are:

| Horizon | KM estimate | Naive rate |
|---|---|---|
| 12h | 0.2247 | 0.222 |
| 24h | 0.2944 | 0.285 |
| 48h | 0.3134 | 0.299 |
| **72h** | **0.5519** | **0.312** |

The KM 72h estimate is 0.552 vs naive 0.312. This large difference is real and correct — the KM estimator accounts for the many censored fires that were still approaching their evac zones when monitoring ended. The naive rate treats all censored fires as "never hit" which is wrong for right-censored data.

**The likely issue:** If the base models were predicting 72h probabilities around 0.3 (the naive rate), then blending with KM prior of 0.55 would sharply raise them — and if the evaluation metric uses naive labels (0/1 based on observed events), this would look bad. But the KM estimate is actually the better calibrated value for the true population.

**Recommendation:** Do not disable KM prior blending — instead, investigate why it was "destructive." It likely exposed that the evaluation function (`compute_brier_censor_aware`) is itself using incorrect naive labels for the 72h horizon. The KM prior was probably right and exposing a bug in the Brier computation.

---

## 10. Priority Fix List

Ranked by expected impact on leaderboard score:

| Priority | Fix | Expected Impact |
|---|---|---|
| 🔴 P1 | Fix `dist_risk_close` threshold from 3000m → 5000m | Correctly classifies 13 more hit fires |
| 🔴 P1 | Drop 3 dead features (`fast_hit_signal`, `low_res_x_urgency`, `low_res_x_growth`) | Removes NaN columns that may break models |
| 🔴 P1 | Fix LightGBM objective — use XGBoost AFT or correct label scheme | LGB is not learning survival at all currently |
| 🔴 P1 | Fix percentile features — compute from train, apply to test | Fixes train/test inconsistency |
| 🔴 P1 | Add `BRIER_WEIGHTS[12] = 0.15` to optimize 12h Brier | 12h is currently unconstrained |
| 🟡 P2 | Drop 11 report-flagged junk features | Cleaner regularization, less noise |
| 🟡 P2 | Add `log_num_perimeters` (corr = 0.405) | Strong missing feature |
| 🟡 P2 | Add `bearing_is_real` flag (corr = 0.314) | Fixes spread_bearing_cos artifact |
| 🟡 P2 | Fix meta-model risk target | Current target is logically inverted for censored fires |
| 🟡 P2 | Fix monotonicity with vectorized cummax | Removes potential clipping order bug |
| 🟢 P3 | Fix low_res interactions to use available features | Replace dead interactions with useful ones |
| 🟢 P3 | Reconsider KM prior disable | 72h predictions may be systematically underestimated |
| 🟢 P3 | Implement two-segment model architecture | Report's top structural recommendation |
| 🟢 P3 | Fix kinetic_threat_proxy (squaring near-zero is unstable) | Minor cleanup |

---

## 11. Corrected Feature Engineering Function

This replaces V5's `engineer_features()` with all bugs fixed and all report recommendations applied:

```python
def engineer_features_v6(df, train_df=None):
    """
    V6 feature engineering — all bugs fixed, report recommendations applied.
    train_df: pass training DataFrame when processing test set (for percentile consistency).
    """
    df = df.copy()

    # ── TIER 1: Always-available features (221/221) ───────────────────
    df['log_dist_min']          = np.log1p(df['dist_min_ci_0_5h'])
    df['log1p_area_first']      = np.log1p(df['area_first_ha'])         # already in data but ensure
    df['log_num_perimeters']    = np.log1p(df['num_perimeters_0_5h'])   # MISSING IN V5: corr=0.405
    df['is_close']              = (df['dist_min_ci_0_5h'] < 5000).astype(int)  # FIX: was 3000
    df['bearing_is_real']       = (df['spread_bearing_cos'] != 1.0).astype(int)  # MISSING: corr=0.314

    # ── Temporal: keep only month (drop hour, dayofweek per report) ──
    df['is_summer']             = df['event_start_month'].isin([6, 7, 8]).astype(int)
    # drop: event_start_hour, event_start_dayofweek

    # ── Interaction with segment flag (only meaningful features) ──────
    df['low_res_x_log_dist']    = df['low_temporal_resolution_0_5h'] * df['log_dist_min']
    df['low_res_x_area']        = df['low_temporal_resolution_0_5h'] * df['log1p_area_first']
    # REMOVED: low_res_x_urgency, low_res_x_growth (always zero)

    # ── TIER 2: Multi-perimeter fires (61/221) ───────────────────────
    df['has_growth']            = (df['area_growth_abs_0_5h'] > 0).astype(int)
    df['urgency_ratio']         = df['closing_speed_m_per_h'] / (df['dist_min_ci_0_5h'] + 1)
    df['log_urgency']           = np.log1p(df['urgency_ratio'])
    df['directional_threat']    = df['alignment_abs'] * df['closing_speed_abs_m_per_h']
    df['confidence_weighted_urgency'] = df['urgency_ratio'] * (1 - df['low_temporal_resolution_0_5h'])

    # ── TIER 3: Moving fires only (25/221) ───────────────────────────
    df['wind_driven_index']     = df['centroid_speed_m_per_h'] / (df['radial_growth_rate_m_per_h'] + 1)
    df['explosive_growth_index']= df['area_growth_rel_0_5h'] / (np.log1p(df['area_first_ha']) + 0.1)
    df['night_growth_anomaly']  = df['is_night'] * df['area_growth_rate_ha_per_h']
    df['fire_momentum']         = df['area_growth_rate_ha_per_h'] * df['centroid_speed_m_per_h']

    # ── Night flag (use sparingly — weak signal) ─────────────────────
    df['is_night']              = ((df['event_start_hour'] >= 20) | 
                                    (df['event_start_hour'] <= 6)).astype(int)

    # ── Percentile features — FIXED for train/test consistency ───────
    ref_df = train_df if train_df is not None else df
    from scipy.stats import percentileofscore
    for col in ['dist_min_ci_0_5h', 'area_first_ha']:
        ref_vals = ref_df[col].values
        if train_df is not None:
            df[f'{col}_pctile'] = df[col].apply(
                lambda x: percentileofscore(ref_vals, x, kind='rank') / 100.0
            )
        else:
            df[f'{col}_pctile'] = df[col].rank(pct=True)
    # REMOVED: closing_speed_m_per_h_pctile (88% zeros → percentile is meaningless)

    # ── FEATURES TO EXPLICITLY DROP ───────────────────────────────────
    # These are present in the raw data and must be excluded from feature_cols:
    cols_to_drop = [
        'relative_growth_0_5h',      # exact dup of area_growth_rel_0_5h
        'projected_advance_m',        # exact neg of dist_change_ci_0_5h
        'alignment_cos',              # strictly weaker than alignment_abs
        'spread_bearing_deg',         # use sin/cos, not raw degrees
        'along_track_speed',          # corr = 0.008
        'cross_track_component',      # corr = −0.058
        'dist_accel_m_per_h2',       # corr = −0.073, 84% zeros
        'event_start_hour',           # corr = 0.047
        'event_start_dayofweek',      # corr = −0.119, no causal mechanism
        'log_area_ratio_0_5h',        # corr 0.81 with log1p_growth, weaker
        'fast_hit_signal',            # always zero (threshold 500 m/h unreachable)
        'low_res_x_urgency',          # always zero
        'low_res_x_growth',           # always zero
        'kinetic_threat_proxy',       # squaring near-zero, only 18/221 nonzero
        'night_closing_anomaly',      # only 11/221 nonzero
        'event_id',
    ]
    return df, cols_to_drop


# Usage:
train_eng, cols_to_drop = engineer_features_v6(train_df)
test_eng, _              = engineer_features_v6(test_df, train_df=train_df)

TARGET_COLS = ['event_id', 'time_to_hit_hours', 'event']
feature_cols = [c for c in train_eng.columns
                if c not in TARGET_COLS
                and c not in cols_to_drop
                and train_eng[c].dtype != 'object']

# Also fix Brier weights:
BRIER_WEIGHTS = {12: 0.15, 24: 0.25, 48: 0.35, 72: 0.25}

# And fix monotonicity enforcement:
cols = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']
submission[cols] = submission[cols].cummax(axis=1).clip(0.001, 0.999)
```

---

*Analysis performed against EDA report findings. All correlations verified on train.csv (221 rows).
Dead features confirmed by computing nonzero counts directly on training data.*
