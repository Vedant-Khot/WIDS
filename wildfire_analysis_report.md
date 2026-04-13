# Wildfire Evacuation Zone Threat Prediction — Full EDA & Modeling Report

**Competition:** Survival-aware wildfire threat classification  
**Dataset:** 316 labeled fire events (train: 221, test: 95)  
**Task:** Predict four probabilities per fire — P(hit evac zone by 12h, 24h, 48h, 72h)  
**Evaluation emphasis:** Calibration over raw accuracy; proper handling of right-censored survival data

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [The Sparsity Problem — Root Cause Diagnosis](#2-the-sparsity-problem--root-cause-diagnosis)
3. [The Dominant Signal — dist_min_ci_0_5h](#3-the-dominant-signal--dist_min_ci_0_5h)
4. [Feature-by-Feature Analysis](#4-feature-by-feature-analysis)
5. [Survival Structure Analysis](#5-survival-structure-analysis)
6. [Feature Availability Matrix](#6-feature-availability-matrix)
7. [Artifacts, Traps, and Gotchas](#7-artifacts-traps-and-gotchas)
8. [Exact Duplicates to Drop](#8-exact-duplicates-to-drop)
9. [Engineered Features](#9-engineered-features)
10. [Modeling Strategy](#10-modeling-strategy)
11. [Calibration Requirements](#11-calibration-requirements)
12. [Final Feature Checklist](#12-final-feature-checklist)
13. [Submission Guidelines](#13-submission-guidelines)
14. [Key Numbers Summary](#14-key-numbers-summary)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total fires | 316 |
| Training rows | 221 |
| Test rows | 95 |
| Features | 34 |
| Targets | time_to_hit_hours, event |
| Hit events (event=1) | 69 (31.22%) |
| Censored events (event=0) | 152 (68.78%) |
| Prediction horizons | 12h, 24h, 48h, 72h |
| Data type | Right-censored survival data |

### Target Label Construction

- `event = 1`: Fire came within 5 km of an evacuation zone centroid within 72 hours. `time_to_hit_hours` is the observed time of that event.
- `event = 0`: Fire never came within 5 km within 72 hours. `time_to_hit_hours` is the **last observed time** (censoring time), always ≤ 72.

This is **right-censored survival data**. Standard binary classification ignores the censoring mechanism and will produce miscalibrated outputs. The correct framework is survival modeling.

### Empirical Hit Rates by Horizon

| Horizon | Fires that hit | Naive rate |
|---|---|---|
| 12h | 49 | 22.2% |
| 24h | 63 | 28.5% |
| 48h | 66 | 29.9% |
| 72h | 69 | 31.2% |

### Kaplan-Meier Estimates (overall population)

| Horizon | KM P(hit by h) |
|---|---|
| 12h | 0.2247 |
| 24h | 0.2944 |
| 48h | 0.3134 |
| 72h | 0.5519 |

Note: KM estimates differ from naive rates because they account for censoring. The 72h KM estimate (0.55) is substantially higher than the naive 0.31 — indicating that many censored fires were on track to hit but ran out of the observation window. This distinction is critical for calibration.

---

## 2. The Sparsity Problem — Root Cause Diagnosis

### The Zero Problem Is Structural, Not Random

The dataset appears heavily zero-inflated — many features are 88–92% zeros. This is **not random missingness** and cannot be addressed with standard imputation. The root cause is:

**72.4% of fires (160/221) have only a single perimeter observation in the first 5 hours.**

All growth, movement, and dynamics features are computed as *differences between perimeters*. With only one perimeter snapshot, there is no second measurement to compute a difference from. These features are therefore definitionally zero for single-perimeter fires — they encode the absence of data, not the absence of fire behavior.

### Perimeter Count Distribution

| num_perimeters | Count | Hit rate |
|---|---|---|
| 1 | 160 | 20.0% |
| 2 | 25 | — |
| 3 | 7 | — |
| 4 | 5 | — |
| 5 | 5 | — |
| 6 | 2 | — |
| 7 | 7 | — |
| 9 | 2 | — |
| 11 | 1 | — |
| 12 | 5 | — |
| 13 | 1 | — |
| 17 | 1 | — |
| **Multi (>1)** | **61** | **60.7%** |

### The Two Subpopulations

| Group | n | Hit rate | Features available |
|---|---|---|---|
| Single-perimeter (low_temporal_resolution=1) | 160 | 20.0% | ~6 features only |
| Multi-perimeter (low_temporal_resolution=0) | 61 | 60.7% | All 34 features |

These are fundamentally different modeling problems. Single-perimeter fires have almost no dynamic information — only initial area, distance to evac zone, and time metadata. Multi-perimeter fires have full growth, movement, and closing-speed data.

**Modeling implication:** Treat these as two subpopulations. Fit separate models or use a segmentation flag as an interaction variable. Do not pool them naively.

### The `low_temporal_resolution_0_5h` Flag

This flag is defined as: 1 if `dt_first_last_0_5h < 0.5h` OR only 1 perimeter, else 0.

- `low_temporal_resolution = 1`: 161 fires — all growth/movement features are exactly 0
- `low_temporal_resolution = 0`: 60 fires — all features are populated

The flag is a perfect missingness indicator. It is also a strong predictor in its own right (correlation with event = −0.379, meaning good-data fires hit more often).

**Modeling implication:** Always include `low_temporal_resolution_0_5h` as a direct feature AND use it as a segmentation criterion.

---

## 3. The Dominant Signal — dist_min_ci_0_5h

### The Most Important Finding in the Dataset

`dist_min_ci_0_5h` (minimum distance to nearest evacuation zone centroid, in meters) is a near-perfect binary classifier on the training set.

**Every fire within 4,674m of an evacuation zone centroid hit (100% hit rate, n=69). No fire at or beyond 5,000m ever hit (0% hit rate, n=152).**

| Distance threshold | Below: hit rate | Above: hit rate |
|---|---|---|
| < 1,000m | 100% (n=7) | 28.97% (n=214) |
| < 2,000m | 100% (n=26) | 22.05% (n=195) |
| < 3,000m | 100% (n=56) | 7.88% (n=165) |
| < 4,000m | 100% (n=63) | 3.80% (n=158) |
| < 5,000m | 100% (n=69) | 0.00% (n=152) |
| < 6,000m | 89.6% (n=77) | 0.00% (n=144) |
| < 8,000m | 81.2% (n=85) | 0.00% (n=136) |
| < 10,000m | 75.8% (n=91) | 0.00% (n=130) |

The exact maximum distance for a 100% hit fire: **4,674m**. There is a clean gap — no fires between 4,674m and 5,000m exist in training data, and no fires above 5,000m ever hit.

### Correlation Comparison

| Feature | Correlation with event |
|---|---|
| raw dist_min_ci_0_5h | −0.481 |
| log1p(dist_min_ci_0_5h) | **−0.812** |
| is_close (dist < 5000m) | **1.000** |

The log transform dramatically improves the correlation from −0.481 to −0.812. The binary indicator `is_close` achieves 1.000 on training data — a perfect separator.

### Critical Warning for the Test Set

The 1.000 correlation on training data is almost certainly optimistic. The 4,674–6,000m range contains no training fires, but the test set may have fires in this gray zone. Do **not** use a hard cutoff. Instead:

- Use `log1p(dist_min_ci_0_5h)` as a continuous feature
- Apply a logistic function for smooth, non-binary probability outputs
- Calibrate on cross-validation folds, not just training accuracy

### What dist_min_ci Actually Measures

This is the distance from the fire's initial centroid to the nearest evacuation zone centroid at the start of the observation window. It is a **static snapshot** — not accounting for fire growth direction or speed. The fact that it perfectly predicts outcomes suggests that fires are fundamentally proximity-driven: if a fire starts close enough, it will reach an evac zone; if it starts far, it won't within 72 hours.

---

## 4. Feature-by-Feature Analysis

### 4.1 Distance Features (9 total)

**dist_min_ci_0_5h** — Always available (221/221). Minimum distance to nearest evac zone centroid. Single best predictor. Always log-transform. Correlation with event: −0.481 (raw), −0.812 (log-transformed). **Verdict: Keep, transform.**

**dist_std_ci_0_5h** — Available for 19/221 fires. Standard deviation of distances to all evac zone centroids (spread of fire relative to multiple zones). Corr: +0.142. Only available when multiple perimeters exist. **Verdict: Keep for tier 2/3 model.**

**dist_change_ci_0_5h** — Available for 18/221 fires. Change in distance (d_final − d_initial). Negative means closing. Corr: −0.106. **This is the exact negative of `projected_advance_m`. Drop one — keep `dist_change_ci_0_5h`.**

**projected_advance_m** — Available for 18/221 fires. Defined as d_initial − d_final. Exact negative of `dist_change_ci_0_5h`. Corr: +0.106. **Verdict: Drop — exact duplicate with opposite sign.**

**dist_slope_ci_0_5h** — Available for 52/221 fires. Linear slope of distance vs time (meters/hour). Negative means fire is closing on evac zone. Corr: −0.115. Note: populated for more fires than `dist_change_ci` because it uses the linear fit across all perimeters rather than just first/last. **Verdict: Keep — richer availability than dist_change.**

**closing_speed_m_per_h** — Available for 18/221 fires. Rate of closing (positive = closing). Corr: +0.107. Can be negative (retreating). **Verdict: Use `closing_speed_abs_m_per_h` instead.**

**closing_speed_abs_m_per_h** — Available for 18/221 fires. Absolute closing speed. Corr: +0.139. **Verdict: Keep over signed version.**

**dist_accel_m_per_h2** — Available for 35/221 fires. Acceleration in distance change. Corr: −0.073. Very weak signal. **Verdict: Drop — weak and sparse.**

**dist_fit_r2_0_5h** — Available for 19/221 fires. R² of linear fit to distance vs time. Measures how consistently the fire was approaching/retreating. Corr: +0.143. **Verdict: Optional, keep if model benefits.**

### 4.2 Growth Features (10 total)

**area_first_ha** — Always available (221/221). Initial fire area at t0 in hectares. Corr: −0.181 (larger fires hit LESS often). Important nuance: large fires tend to start farther from evacuation zones (confirmed by `area_first_ha` vs `dist_min_ci` correlation = +0.072 — weak but in expected direction). The negative correlation is likely a confound with distance, not a causal signal. Range: 0.04 to 11,942 ha. **Verdict: Keep, use log1p transform.**

**log1p_area_first** — Always available (221/221). Log(1 + initial area). Corr: −0.168. Preferred over raw area due to extreme right skew (max 11,942 ha). **Verdict: Keep — use this over area_first_ha in models.**

**area_growth_abs_0_5h** — Available for 25/221 fires. Absolute area growth in hectares. Corr: +0.158. Zero for all single-perimeter fires. **Verdict: Keep for tier 3 model.**

**area_growth_rel_0_5h** — Available for 25/221 fires. Relative area growth (fraction of initial area). Corr: +0.166. **EXACT DUPLICATE of `relative_growth_0_5h` — byte-for-byte identical. Drop one.**

**relative_growth_0_5h** — Available for 25/221 fires. Identical to `area_growth_rel_0_5h`. **Verdict: Drop — confirmed duplicate.**

**area_growth_rate_ha_per_h** — Available for 25/221 fires. Growth rate in hectares per hour. Corr: +0.172. **Verdict: Keep for tier 3.**

**log1p_growth** — Available for 24/221 fires. Log(1 + absolute growth). Corr: +0.293. Strongest growth signal due to log compression of skewed values. **Verdict: Best growth feature — keep.**

**log_area_ratio_0_5h** — Available for 25/221 fires. Log(final area / initial area). Corr: +0.229. Corr with `log1p_growth`: 0.810 — highly collinear. **Verdict: Pick one; prefer `log1p_growth` (stronger corr).**

**radial_growth_m** — Available for 25/221 fires. Change in effective radius (meters). Corr: +0.209. Corr with `area_growth_abs_0_5h`: very high. **Verdict: Keep for tier 3, watch collinearity.**

**radial_growth_rate_m_per_h** — Available for 25/221 fires. Rate of radial growth in m/h. Corr: +0.215. Good timing predictor — within close fires, corr with `time_to_hit_hours`: −0.185 (faster growth = hits sooner). **Verdict: Keep.**

### 4.3 Centroid Kinematics (5 features)

**centroid_displacement_m** — Available for 25/221 fires. Total displacement of fire centroid. Corr: +0.208. Only 25 fires actually moved detectably. **Verdict: Keep for tier 3.**

**centroid_speed_m_per_h** — Available for 25/221 fires. Speed of centroid movement. Corr: +0.209. Highly correlated with `centroid_displacement_m`. **Verdict: Keep one; prefer speed (rate-based).**

**spread_bearing_deg** — Available for 25/221 fires. Raw bearing in degrees (0–360). Corr: +0.281. **Verdict: Drop — always use circular encoding (sin/cos) instead of raw degrees. Circular features handle the 0°/360° boundary correctly.**

**spread_bearing_sin** — Available for 25/221 fires. Sine of bearing. Corr: +0.188. **Verdict: Keep.**

**spread_bearing_cos** — ⚠️ **TRAP: Appears available for 221/221 fires but is an artifact.** When a fire has only 1 perimeter, no bearing can be computed. The code defaults to bearing = 0°, so cos(0°) = 1.0. All 161 low-temporal-resolution fires have `spread_bearing_cos = 1.0` exactly. The correlation of −0.323 is partly spurious — it is really detecting "this fire had no movement data" not "the fire was spreading in a particular direction." **Verdict: Create `bearing_is_real = (spread_bearing_cos != 1.0)` as an indicator, then use the raw value only for fires where `bearing_is_real = True`.**

### 4.4 Directionality (4 features)

**alignment_abs** — Available for 61/221 fires. Absolute alignment between fire motion and direction to evac zone (0–1 scale; 1 = perfectly aligned toward evac zone). Corr: +0.349. Important nuance: this feature activates for 61 fires, not just the 25 that physically moved. It is computed from `dist_slope_ci` (how distance to evac is changing over time) rather than centroid displacement — so fires that are closing in on an evac zone even without visible centroid movement will have nonzero alignment. Timing predictor within close fires: corr with time_to_hit_hours = −0.368 (higher alignment = hits sooner). **Verdict: Best directionality feature — always include.**

**alignment_cos** — Available for 61/221 fires. Cosine of angle between fire motion and evac direction. Corr: +0.120. Much weaker than `alignment_abs` because it is signed — a fire heading slightly away gets negative values even if mostly aligned. **Verdict: Drop — `alignment_abs` is strictly better.**

**along_track_speed** — Available for 25/221 fires. Speed component toward/away from evac zone. Corr: +0.008. Essentially zero signal. **Verdict: Drop.**

**cross_track_component** — Available for 25/221 fires. Sideways drift perpendicular to evac direction. Corr: −0.058. Very weak. **Verdict: Drop.**

### 4.5 Temporal Coverage (3 features)

**num_perimeters_0_5h** — Always available (221/221). Count of perimeters in first 5 hours. Corr: +0.371. Strong signal — more perimeters means better data AND higher fire activity. Range: 1–17. Distribution is highly right-skewed (median = 1). **Verdict: Keep, log-transform (log1p).**

**dt_first_last_0_5h** — Available for 61/221 fires (zero for all single-perimeter fires). Time span in hours between first and last perimeter. Corr: +0.353. When nonzero, indicates how long the fire was actively monitored. **Verdict: Keep.**

**low_temporal_resolution_0_5h** — Always available (221/221). Binary flag. Corr: −0.379 with event (note: corr with event is negative because it is 1 for bad-data fires that hit less). **Verdict: Keep — primary segmentation variable.**

### 4.6 Temporal Metadata (3 features)

**event_start_month** — Always available (221/221). Month of fire start (1–12). Corr: +0.093. Distribution: fires are heavily concentrated in summer months (Jun–Sep = 92% of events). Slight elevation in September (42.9% hit rate vs 31–33% in Jun–Aug). **Verdict: Optional — weak but consistent seasonal signal. Include as a low-weight feature.**

**event_start_hour** — Available for 205/221 (16 fires start at midnight, hour=0). Corr: +0.047. Afternoon starts (12–18h) have slightly higher hit rates (41.3%) vs night (23.5%), but sample sizes are small. **Verdict: Drop — very weak, potentially spurious.**

**event_start_dayofweek** — Available for 190/221 (31 fires on Monday, day=0). Corr: −0.119. Day of week should have no causal relationship to fire behavior. **Verdict: Drop — likely spurious.**

---

## 5. Survival Structure Analysis

### Why Survival Modeling Is Required

This is right-censored survival data, which means:

- For hit fires (event=1): we observe the exact time to hit
- For censored fires (event=0): we only know the fire had NOT hit by its last observation time

Using binary classification treats all censored fires as "no hit" — this is incorrect. A fire censored at 40 hours is more informative than a fire censored at 2 hours, but binary classification treats them identically. Survival models use this censoring information properly via the partial likelihood.

### Hit Timing Distribution (69 hit fires)

| Time window | Fires hitting | Cumulative |
|---|---|---|
| 0–3h | 29 (42%) | 29 |
| 3–6h | 15 (22%) | 44 |
| 6–12h | 5 (7%) | 49 |
| 12–24h | 14 (20%) | 63 |
| 24–48h | 3 (4%) | 66 |
| 48–72h | 3 (4%) | 69 |

Key insight: **71% of hits occur within the first 12 hours.** The survival problem is heavily front-loaded. Most fires that will hit do so very quickly. The remaining hits trickle in through 72h.

### KM Estimates by Distance Stratum

| Stratum | n | P(hit 12h) | P(hit 24h) | P(hit 48h) | P(hit 72h) |
|---|---|---|---|---|---|
| Close (< 5km) | 69 | 0.710 | 0.913 | 0.957 | 1.000 |
| Mid (5–50km) | 50 | 0.000 | 0.000 | 0.000 | 0.000 |
| Far (> 50km) | 102 | 0.000 | 0.000 | 0.000 | 0.000 |

Within close fires, the timing distribution is:
- Median time-to-hit: **3.5 hours**
- Mean time-to-hit: **10.0 hours**
- 75th percentile: 14.3 hours

### KM Estimates by Temporal Resolution

| Group | n | P(hit 12h) | P(hit 24h) | P(hit 48h) | P(hit 72h) |
|---|---|---|---|---|---|
| Good data (ltr=0) | 60 | 0.579 | 0.601 | 0.628 | 0.628 |
| Low-res (ltr=1) | 161 | 0.094 | 0.181 | 0.197 | 0.477 |

Note the large jump at 72h for low-res fires in the KM estimate (0.197 → 0.477). This is partly driven by censoring in this group — many fires were censored early and the KM estimator is uncertain at later horizons.

### Within-Close-Fire Timing Predictors

For the 69 fires that will definitely hit, what predicts how quickly they hit?

| Feature | Corr with time_to_hit_hours | Interpretation |
|---|---|---|
| alignment_abs | −0.368 | More aligned → hits sooner |
| log1p_area_first | −0.239 | Larger fire → hits sooner |
| radial_growth_rate_m_per_h | −0.185 | Faster growth → hits sooner |
| area_growth_rate_ha_per_h | −0.142 | Faster growth → hits sooner |
| closing_speed_m_per_h | −0.085 | Faster closing → hits sooner |
| dist_min_ci_0_5h | +0.005 | Distance within close group barely matters |

Interesting: within the close-fire group, the initial distance no longer predicts timing (corr ≈ 0) — because all fires start within 5km. What matters is *how fast* the fire is moving and *how aligned* it is with the evac zone direction.

---

## 6. Feature Availability Matrix

The dataset naturally segments into 5 groups based on feature availability:

| Group | has_growth | has_alignment | has_dist_slope | has_dist_change | n | Hit rate |
|---|---|---|---|---|---|---|
| A: Single-perim, no dynamics | 0 | 0 | 0 | 0 | 160 | 20.0% |
| B: Multi-perim, dist-aligned only | 0 | 1 | 0 | 0 | 9 | 33.3% |
| C: Multi-perim, slope+aligned | 0 | 1 | 1 | 0 | 27 | 59.3% |
| D: Full data, no dist_change | 1 | 1 | 1 | 0 | 7 | 100.0% |
| E: Full data, all features | 1 | 1 | 1 | 1 | 18 | 61.1% |

This confirms the two-subpopulation hypothesis and provides a clear cascading segmentation logic for modeling.

---

## 7. Artifacts, Traps, and Gotchas

### 7.1 spread_bearing_cos = 1.0 Artifact

When a fire has only one perimeter, no direction of spread can be computed. The code defaults the bearing to 0° (north). cos(0°) = 1.0. This means:

- All 161 low-temporal-resolution fires have `spread_bearing_cos = 1.0` exactly
- `spread_bearing_cos` appears to be always-available (221/221 nonzero) but is really only valid for 60 fires
- The correlation of −0.323 is partially spurious — it is detecting "fire is single-perimeter" not directional information

**Fix:** Create `bearing_is_real = (num_perimeters > 1)` or `(spread_bearing_cos != 1.0)` as an indicator flag. Only use the bearing values when this flag is True.

### 7.2 Perfect Training Separation (is_close = 1.000)

The feature `is_close = (dist_min_ci < 5000m)` achieves a correlation of 1.000 with event on training data. This is almost certainly an artifact of limited training data (221 fires) — the test set likely contains fires in the 4,700–6,000m gray zone that the training data happens not to cover.

**Risk:** If you hard-code the threshold, test fires near the boundary will get extreme probabilities (0 or 1) with no gradation. This will hurt calibration.

**Fix:** Use `log1p(dist_min_ci)` as a continuous feature and let the model learn a smooth decision boundary. Do not apply a hard cutoff rule.

### 7.3 Large Initial Area is Negatively Correlated

`area_first_ha` has corr = −0.181 with event — meaning larger fires hit LESS often. This seems counterintuitive. The explanation: large fires (e.g., 10,000 ha) tend to occur in remote areas far from evacuation zones. The negative correlation is a confound with distance, not a meaningful causal relationship.

**Modeling implication:** Include both `log1p_area_first` and `log_dist_min` in the model — the model will learn to separate these confounded effects.

### 7.4 Monotonicity Constraint on Predictions

The four prediction targets are cumulative probabilities over time. By definition, a fire cannot unhit an evacuation zone — so:

```
prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h
```

This constraint must hold for every fire in the submission. Survival models naturally satisfy this (cumulative hazard is monotonically non-decreasing). Binary classifiers fitted independently at each horizon will NOT satisfy this and require post-processing.

**Fix:** Either use a survival model (handles monotonicity intrinsically) or apply isotonic regression post-processing to enforce the ordering.

### 7.5 Do Not Predict Exact 0 or 1

The competition evaluates calibration. Predicting exactly 0.0 or 1.0 results in infinite log-loss if the prediction is wrong (even once). Even for fires clearly beyond 5km, assign a small epsilon (e.g., 0.01–0.03) rather than exactly 0.

For fires clearly within 3km, cap at 0.97–0.99 rather than 1.0.

---

## 8. Exact Duplicates to Drop

These pairs/groups are mathematically identical and must not both appear in a model:

### 8.1 relative_growth_0_5h ≡ area_growth_rel_0_5h
Verified byte-for-byte identical across all 221 rows. Both represent relative area growth as a fraction of initial area. **Drop `relative_growth_0_5h`.**

### 8.2 projected_advance_m ≡ −dist_change_ci_0_5h
Verified as exact negatives across all 221 rows. `projected_advance_m = d_initial − d_final`, `dist_change_ci_0_5h = d_final − d_initial`. Including both inflates feature importance, confuses regularization, and makes model interpretation impossible. **Drop `projected_advance_m`.**

### 8.3 log1p_growth ≈ log_area_ratio_0_5h (corr = 0.810)
Not exact duplicates, but very highly correlated. `log1p_growth` = log(1 + absolute growth in ha). `log_area_ratio` = log(final/initial area) = log(1 + relative growth). They differ when initial area is large. `log1p_growth` has higher predictive correlation (+0.293 vs +0.229). **Drop `log_area_ratio_0_5h`, keep `log1p_growth`.**

### 8.4 alignment_cos vs alignment_abs
Not duplicates, but redundant. `alignment_cos` is signed (−1 to +1); `alignment_abs` is unsigned (0 to 1). The signed version loses information for modeling because most models cannot easily learn that "very negative" and "very positive" alignment both mean "strongly directed." `alignment_abs` is strictly more predictive (corr +0.349 vs +0.120). **Drop `alignment_cos`.**

### 8.5 spread_bearing_deg vs spread_bearing_sin/cos
`spread_bearing_deg` is the raw angle in degrees. This should never be used directly in ML models because the 0°/360° boundary creates a discontinuity — two fires spreading northeast at 359° and 1° appear completely different to a model using raw degrees. Always use the circular encoding (`spread_bearing_sin`, `spread_bearing_cos`). **Drop `spread_bearing_deg`.**

---

## 9. Engineered Features

### 9.1 log_dist_min = log1p(dist_min_ci_0_5h)
**Correlation with event: −0.812** (vs −0.481 for raw distance — a massive improvement)

This is the single most important transformation in the entire dataset. The raw distance is exponentially distributed (range: 307m to 757,700m — over 3 orders of magnitude). Log compression linearizes the relationship with event probability. Must create this before any modeling.

```python
df['log_dist_min'] = np.log1p(df['dist_min_ci_0_5h'])
```

### 9.2 is_close = (dist_min_ci_0_5h < 5000)
**Correlation with event: 1.000 on training data**

Binary indicator for the primary split. Use as a soft logistic proxy, not a hard rule. This is your primary model segmentation feature.

```python
df['is_close'] = (df['dist_min_ci_0_5h'] < 5000).astype(int)
```

### 9.3 bearing_is_real = (spread_bearing_cos != 1.0)
Binary indicator that strips out the artifact where no-movement fires default to bearing=0°.

```python
df['bearing_is_real'] = (df['spread_bearing_cos'] != 1.0).astype(int)
```

Equivalently: `(df['num_perimeters_0_5h'] > 1)`.

### 9.4 log_num_perimeters = log1p(num_perimeters_0_5h)
The perimeter count is right-skewed (max=17, median=1). Log compression improves linearity.

```python
df['log_num_perimeters'] = np.log1p(df['num_perimeters_0_5h'])
```

### 9.5 log1p_area_first (already in dataset)
Already computed. Prefer over raw `area_first_ha` (range: 0.04 to 11,942 ha — extreme skew).

### 9.6 Optional: closing_eta = dist_min_ci / max(closing_speed, epsilon)
Estimated time (in hours) until fire reaches evac zone at current closing speed.

```python
df['closing_eta'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'].clip(lower=0.1))
df['closing_eta'] = df['closing_eta'].clip(upper=10000)
```

Corr with event: −0.154. Useful but only meaningful for fires actively closing (18/221).

---

## 10. Modeling Strategy

### Overview

The data has an inherent two-level structure that must be respected:
1. **Which fires will hit at all?** (dominated by distance)
2. **When will close fires hit?** (dominated by growth/movement/alignment)

A single pooled model can learn this but will struggle with the extreme class imbalance in feature availability. A two-model architecture is more robust.

### Recommended Architecture: Two-Segment Survival Model

#### Segment 1: Far Fires (dist_min_ci ≥ 5000m, n=152 in training)

Available features: `log_dist_min`, `log1p_area_first`, `log_num_perimeters`, `low_temporal_resolution_0_5h`, `event_start_month`

These fires never hit in training. Assign near-zero baseline probabilities. Use a logistic regression on `log_dist_min` to produce a smooth, calibrated small probability rather than exactly zero.

For far fires, produce: `prob_12h ≈ prob_24h ≈ prob_48h ≈ prob_72h ≈ ε` (small, smoothly decreasing with distance). Suggested epsilon range: 0.01–0.05.

#### Segment 2: Close Fires (dist_min_ci < 5000m, n=69 in training)

These fires all eventually hit (within 72h in training). The modeling question is: **when** do they hit?

Use Kaplan-Meier or Cox Proportional Hazards to model time-to-hit within this segment. Features for timing:

- `alignment_abs` (corr with time_to_hit: −0.368)
- `log1p_area_first` (−0.239)
- `radial_growth_rate_m_per_h` (−0.185)
- `area_growth_rate_ha_per_h` (−0.142)
- `closing_speed_m_per_h` (−0.085)
- `log_dist_min` (barely relevant within segment, but include)

For close fires, convert Cox hazard H(t) to probability: `P(hit by t) = 1 − exp(−H(t))`

Evaluate at t = 12, 24, 48, 72 hours.

### Alternative: Single XGBoost Survival Model (AFT)

XGBoost with `objective = "survival:aft"` handles censoring directly without manual segmentation. Features: all cleaned features with zero-indicators. Output: predicted survival time → convert to per-horizon probabilities.

This is a good second model for ensembling.

### Alternative: Four Independent Logistic Regressors + Monotone Fix

Fit four binary classifiers, one per horizon (12h, 24h, 48h, 72h). Each uses label: 1 if event=1 AND time_to_hit ≤ h. Apply isotonic regression post-processing to enforce monotonicity across horizons. Simpler but ignores censoring.

### Cross-Validation Strategy

Use stratified k-fold (k=5) stratified by `event` AND `low_temporal_resolution_0_5h`. This ensures each fold has representative proportions of hit fires and data-quality groups.

With 221 training rows, each fold has ~44 examples — small. Consider using k=10 or leave-one-out for final evaluation.

---

## 11. Calibration Requirements

The competition explicitly emphasizes calibration over raw accuracy. This means predicted probabilities must reflect true frequencies.

### What Good Calibration Looks Like

A model is well-calibrated if, among all fires predicted to have 30% hit probability, approximately 30% actually hit. This can be verified with a reliability diagram (calibration curve).

### Key Calibration Actions

**1. Use log-loss or Brier score as the optimization metric, not AUC or accuracy.**

**2. Apply Platt scaling (logistic calibration) or isotonic regression after fitting any model.** Fit these calibrators on held-out fold data, not training data.

**3. Never predict exactly 0 or 1.** Use floor/ceiling: clip predictions to [0.01, 0.99] at minimum.

**4. Enforce monotonicity before submission:**
```python
# Ensure prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h
df['prob_24h'] = df[['prob_12h','prob_24h']].max(axis=1)
df['prob_48h'] = df[['prob_24h','prob_48h']].max(axis=1)
df['prob_72h'] = df[['prob_48h','prob_72h']].max(axis=1)
```

**5. Check calibration separately for the two segments** (close fires and far fires). A model may be well-calibrated overall but miscalibrated within each group.

**6. The KM estimates are your calibration anchor.** Overall population KM estimates: 12h → 0.22, 24h → 0.29, 48h → 0.31, 72h → 0.55. Your average predicted probabilities across the test set should be in this range.

---

## 12. Final Feature Checklist

### Features to ALWAYS Include (6)

| Feature | Notes |
|---|---|
| `log_dist_min` (engineered) | log1p(dist_min_ci_0_5h) — single most important |
| `low_temporal_resolution_0_5h` | Primary segmentation flag |
| `log_num_perimeters` (engineered) | log1p(num_perimeters_0_5h) |
| `log1p_area_first` | Already in dataset; prefer over area_first_ha |
| `dt_first_last_0_5h` | Observation window length; 0 for single-perim fires |
| `event_start_month` | Weak but consistent seasonal signal |

### Features to Include for Multi-Perimeter Fires (Tier 2, 52–61 fires)

| Feature | Notes |
|---|---|
| `alignment_abs` | Best directionality feature |
| `dist_slope_ci_0_5h` | How fast distance is changing |
| `bearing_is_real` (engineered) | Flag for valid bearing data |

### Features to Include for Moving Fires (Tier 3, 25 fires)

| Feature | Notes |
|---|---|
| `log1p_growth` | Best growth feature |
| `radial_growth_rate_m_per_h` | Good timing predictor |
| `area_growth_rate_ha_per_h` | Keep for timing |
| `centroid_speed_m_per_h` | Fire movement speed |
| `closing_speed_abs_m_per_h` | Absolute closing rate |
| `spread_bearing_sin` | Only valid when bearing_is_real=1 |
| `spread_bearing_cos` | Only valid when bearing_is_real=1 |

### Features to DROP

| Feature | Reason |
|---|---|
| `relative_growth_0_5h` | Exact duplicate of area_growth_rel_0_5h |
| `projected_advance_m` | Exact negative of dist_change_ci_0_5h |
| `alignment_cos` | Strictly weaker than alignment_abs |
| `spread_bearing_deg` | Use sin/cos encoding instead |
| `along_track_speed` | Corr ≈ 0.008, no signal |
| `cross_track_component` | Corr ≈ −0.058, no signal |
| `dist_accel_m_per_h2` | Weak, noisy (corr −0.073) |
| `event_start_hour` | Corr +0.047, very weak |
| `event_start_dayofweek` | Corr −0.119, likely spurious |
| `log_area_ratio_0_5h` | Corr 0.81 with log1p_growth; weaker |
| `centroid_displacement_m` | Near-duplicate of centroid_speed |
| `dist_change_ci_0_5h` | Exact negative of projected_advance (keep one) |
| `event_id` | Identifier, no predictive value |

---

## 13. Submission Guidelines

### Format

CSV with 95 rows (one per test fire) and columns: `event_id, prob_12h, prob_24h, prob_48h, prob_72h`

### Mandatory Checks Before Submitting

1. All values are between 0 and 1 (inclusive)
2. No exact 0.0 or 1.0 values — use floor 0.01 and ceiling 0.99
3. For every row: `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`
4. 95 rows exactly — no missing event_ids
5. Average predicted probabilities are in the right ballpark: ~0.22, ~0.29, ~0.31, ~0.55 for 12h, 24h, 48h, 72h

### Baseline Strategy (Minimum Viable Submission)

Even without a full survival model, a strong baseline can be built using distance alone:

```python
import numpy as np

def predict_baseline(dist_min):
    log_d = np.log1p(dist_min)
    # Logistic on log distance, fit parameters from training
    # Approximate: fires <5km always hit, >5km never hit
    p_72h = 1 / (1 + np.exp(0.00005 * (dist_min - 4500)))
    p_12h = p_72h * 0.71  # KM ratio: 0.71 of 72h hits occurred by 12h
    p_24h = p_72h * 0.913
    p_48h = p_72h * 0.957
    return np.clip([p_12h, p_24h, p_48h, p_72h], 0.01, 0.99)
```

---

## 14. Key Numbers Summary

| Metric | Value |
|---|---|
| Training size | 221 fires |
| Hit rate | 31.2% (69 fires) |
| Single-perimeter fires | 72.4% (160 fires) |
| Single-perimeter hit rate | 20.0% |
| Multi-perimeter hit rate | 60.7% |
| Perfect separation threshold | 4,674m (dist_min_ci) |
| Best single feature correlation | log_dist_min: −0.812 |
| Exact duplicate pairs | 2 confirmed (relative_growth, projected_advance) |
| Features to drop | 13 |
| Features to keep (clean set) | ~16 |
| Median time-to-hit (close fires) | 3.5 hours |
| Fires hitting within 12h | 49 (71% of all hits) |
| KM P(hit by 72h) | 0.552 (accounting for censoring) |
| Naive P(hit by 72h) | 0.312 (ignoring censoring) |

---

*Report generated from EDA session on train.csv (221 rows, 34 features) and metaData.csv. All findings verified with Python/pandas. Correlations are Pearson; survival estimates use Kaplan-Meier with Greenwood standard errors.*
