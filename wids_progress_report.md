# WiDS 2025 Wildfire Survival Model — Full Progress Report

> **Competition:** WiDS 2025 Datathon — Wildfire Evacuation Zone Hit Prediction  
> **Metric:** `0.3 × C-index + 0.7 × (1 − Weighted Brier Score)`  
> **Best Score Achieved:** **0.9599** (submission_v4.csv)  
> **Dataset:** 221 train fires (69 hits, 31.2%), 95 test fires

---

## Leaderboard History

| Submission | Score | Notes |
|---|---|---|
| `submission_v4.csv` | **0.9599** ✅ Best | GBSA + RSF + XGB-Cox + XGB-AFT ensemble |
| `submission_claude_v1.csv` | Not yet submitted | claude_v1 model, OOF H=0.971 |
| `submission_v6_final.csv` | **0.9577** | Triage + v4-anchored floor |
| `submission_v8.csv` | Not yet submitted | Isotonic calibration ensemble |
| `submission_v7.csv` | **0.92** | Broken heuristic probability formula |
| `submission_v6.csv` | ~0.92 | Pure triage, too conservative |
| `submission_model_1_imputated_advanced.csv` | Not yet submitted | 72h collapse bug |

---

## Phase 1 — EDA & Data Understanding

### Key Findings from `wildfire_analysis_report.md`

- **Distance is the oracle signal:** `dist_min_ci_0_5h` alone separates hits from non-hits almost perfectly. No fire > 4,674m ever hit in training.
- **Structural sparsity:** 72% of fires are "low temporal resolution" (only 1 perimeter snapshot) — most dynamic features are zero for them.
- **Bearing artifact:** `spread_bearing_cos = 1.0` is a default/placeholder for 160 of 221 fires. Using raw bearing corrupts the model.
- **Duplicate columns confirmed:** `relative_growth_0_5h` = `area_growth_rel_0_5h`, `projected_advance_m` = negative of `dist_change_ci_0_5h`.
- **KM estimates vs. naive rates:**

| Horizon | Naive Hit Rate | KM Estimate |
|---|---|---|
| 12h | 22.2% | ~22% |
| 24h | 28.5% | ~29% |
| 48h | 29.9% | ~31% |
| 72h | **31.2%** | **~55%** ← key gap |

> The 72h KM estimate (0.55) being way above the naive rate (0.31) is because most "far" fires were censored early — they never had a chance to hit within the observation window.

---

## Phase 2 — Bug Fixes Across Versions

### V5 Bugs (from `v5_run_report.md`)
- **Catastrophic collapse:** Final isotonic re-calibration saw all 69 close fires as events → output 0.999 for everyone at 72h
- **KM prior blending bug:** Weighted average pulled all 72h predictions to 0.999
- **Lesson:** Per-fold calibration is safer than post-hoc global re-calibration

### V6 Bugs (Fixed in this session)
| Bug | Fix |
|---|---|
| `features_cloe` typo → crash | Fixed to `features_close` |
| `LogisticRegression` on all-zero labels → crash | Replaced with distance-decay formula |
| Flat probabilities across horizons for far fires | Applied KM-anchored ratios per horizon |
| No CV evaluation for close model | Added 5-fold CV → OOF C-index: 0.6714 |
| `RSF.feature_importances_` not implemented | Removed that block |
| `import seaborn` not installed | Removed unused import |

### V7 Bug (Major — caused 0.92 score)
- Used `1 - exp(-0.02 * risk * h/24)` heuristic instead of real isotonic calibration
- This formula doesn't map to actual hit rates — v8 reverted to isotonic calibration

### claude_new_model_after_eda_v1.py Bugs Fixed
| Bug | Fix |
|---|---|
| `corrwith()` on scalar → AttributeError | Changed to `pd.isna(col.corr(event))` |
| `OPTUNA_TRIALS=50, N_REPEATS=3` → 2-3 hour runtime | Reduced to 20 trials, 1 repeat |
| `KM_BLEND[72]=0.15` too conservative | Increased to 0.40 |
| Wrong data file path (`train_model_1_imputed_advanced.csv`) | User corrected to `train_imputed_advanced.csv` |

---

## Phase 3 — Model Architecture Evolution

### V3 (Baseline Ensemble)
- **Models:** XGBoost (Cox) + LightGBM + GBSA + RSF  
- **Issue:** LightGBM used `regression` objective (wrong for survival), not `cox`  
- **Calibration:** Isotonic per fold  

### V4 (Best Score: 0.9599)
- **Models:** GBSA + RSF + XGB-Cox + XGB-AFT (4-model ensemble)
- **Key feature:** Per-fold Isotonic Calibration maps risk scores to real probabilities
- **72h handling:** Hard `max(prob_72h, 0.55)` floor applied post-processing
- **OOF Hybrid:** ~0.97

### V5 (Broken — Score Not Public)
- Attempted stacking meta-model + KM blending
- **FATAL:** Post-hoc global isotonic calibration collapsed 72h to 0.999
- Never submitted due to collapse detection

### V6 (Triage Specialist — 0.9577)
- **Architecture:** Hard 4700m threshold splits fires into two groups
  - Far fires (>4700m) → Logistic decay formula (distance-based)
  - Close fires (≤4700m) → RSF timing specialist
- **Problem:** Too conservative at 72h (mean=0.29 vs KM=0.55)

### V6C / V6_final (Calibrated Triage — 0.9577)
- Added transition zone (4500–5500m) instead of hard threshold
- Added GBSA/RSF ensemble for close fires
- **v4-anchored floor:** KM ratios applied to far fires
- Still slightly under v4 due to missing ensemble breadth

### V7 (Failed — 0.92)
- Rebuilt from scratch with GBSA + LGB ensemble
- **Fatal mistake:** Used exponential heuristic instead of isotonic calibration
- Proved definitively: isotonic calibration is mandatory for this competition

### V8 (Precision Ensemble — Not Yet Submitted)
- Restored isotonic calibration (v4's secret)
- Clean features from EDA (dropped duplicates + artifacts)
- GBSA + LGB ensemble with per-fold Optuna tuning
- 72h mean: 0.67 (v4 was 0.70), correlation with v4: 0.96

### claude_new_model_after_eda_v1.py (In Progress)
- **Most sophisticated pipeline yet**
- 4 models: GBSA, RSF, XGB-Cox (survival:cox), XGB-AFT (survival:aft)
- 2 seeds × 5 folds with Optuna (20 trials/model)
- Blend grid search + Ridge stacking comparison
- KM prior blend at 72h = 0.40 (v4-anchored)
- OOF Hybrid scores seen so far: **H=0.984 (Fold 4 GBSA)**

---

## Phase 4 — Feature Engineering

### Always Drop (Confirmed Junk)
```
relative_growth_0_5h  → exact dup of area_growth_rel_0_5h
projected_advance_m   → exact negative of dist_change_ci_0_5h  
alignment_cos         → strictly weaker than alignment_abs
spread_bearing_deg    → use sin/cos encoding only
along_track_speed     → corr = 0.008
cross_track_component → corr = -0.058
dist_accel_m_per_h2  → 84% zeros, corr = -0.073
```

### Key Engineered Features Added
| Feature | Correlation | Notes |
|---|---|---|
| `log_dist_min` | −0.55 | Primary signal |
| `log_num_perimeters` | +0.41 | ADD-1: was missing in V5 |
| `bearing_is_real` | +0.31 | ADD-2: fixes bearing artifact |
| `low_res_x_log_dist` | — | ADD-3: replaces dead interactions |
| `close_and_closing` | — | ADD-4: meaningful interaction |
| `dist_x_alignment` | — | ADD-4: new meaningful interaction |

### Percentile Features (FIX-5)
- Computed from training distribution, applied to test
- Previously computed separately → data leakage risk

---

## Phase 5 — Calibration Strategy

### What Works
1. **Per-fold Isotonic Calibration** (v4, v8, claude_v1): Map raw risk scores to actual hit-by-H rates using censoring-aware labels. This is the #1 driver of Brier score quality.
2. **KM Prior Blending at 72h** (0.40 weight): Pulls predictions toward the population-level 0.55 rate. Prevents systematic underestimation of 72h risk.
3. **Vectorized cummax Monotonicity**: Ensures `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h` without loop bugs.

### What Doesn't Work
1. **Global post-hoc isotonic re-calibration** (v5 fatal bug): Sees all close fires as events → collapses everything to 1.0
2. **Heuristic exponential formula** (v7 fatal bug): `1 - exp(-r × h)` doesn't match actual hit rates
3. **Single-class LogisticRegression** (v6 bug): Far fires are all non-events → crashes or predicts 0 for everything

---

## Phase 6 — Competitive Intelligence

### From the "0.98+ Score Mystery" Discussion
- **Poster at 0.972** confirmed: C-index ≈ 0.94, WBS ≈ 0.014 (same as our OOF stats)
- **Our C-index is competitive:** claude_v1 shows GBSA C=0.9615, XGB-Cox C=0.9716
- **The gap is WBS:** We're at ~0.017, need ~0.006 for 0.98+
- **External data hypothesis:** Top scorers (0.98+) likely use FIRMS/GOES satellite fire data with location coordinates that we don't have access to
- **Our realistic ceiling without external data:** ~0.970

### External Data Evaluated
| Source | Decision | Reason |
|---|---|---|
| "WiDS 2026 External Features" CSV | ❌ Rejected | Synthetic data (1881–present), no join key, global scope |
| CALFIRE perimeters | Not tried | Would need lat/lon to join |
| FIRMS satellite | Not tried | Requires location decoding |

---

## Current Status & Next Steps

### Immediate (In Progress)
- `claude_new_model_after_eda_v1.py` running on `train_imputed_advanced.csv`
  - Using imputed data (from `model_1.py` preprocessing)
  - Will output `submission_claude_v1_imputed_data.csv`

### Fix Needed: model_1.py 72h Collapse
The `submission_model_1_imputated_advanced.csv` has:
```
prob_72h: mean=0.9990, std=0.0000 ← ALL fires = 0.999
```
Root cause: at 72h, all 69 valid samples are events → isotonic calibration returns 1.0

**Fix:** Skip isotonic at 72h and use raw GBSA survival function probabilities instead.

### Priority Queue
1. ✅ Run `claude_new_model_after_eda_v1.py` on imputed data → submit
2. 🔧 Fix 72h collapse in `model_1.py`
3. 📊 If claude_v1 > 0.9599: analyze what drove the improvement
4. 🔁 Increase `KM_BLEND[48]` to 0.10 in claude_v1 (WBS weight is 0.35 at 48h)
5. 🔁 Try `N_SEEDS=4` for final submission

---

## File Map

| File | Purpose | Status |
|---|---|---|
| `train_fixed.csv` | Training data (72h sat bug fixed) | ✅ Primary |
| `train_imputed_advanced.csv` | Training data with imputed NaN features | ✅ Used in claude_v1 |
| `test_imputed_advanced.csv` | Test data with imputations | ✅ Used in claude_v1 |
| `ai_model_v4.py` | Best-scoring model (0.9599) | ✅ Reference |
| `claude_new_model_after_eda_v1.py` | Most sophisticated pipeline | 🔄 Running |
| `models/model_1.py` | GBSA+RSF ensemble | ⚠️ 72h collapse bug |
| `models/model_3_optuna.py` | Optuna-tuned variant | 📋 Not evaluated |
| `wildfire_analysis_report.md` | Full EDA findings | ✅ Reference |
| `mistake_notes.md` | Known bugs and pitfalls | ✅ Reference |
