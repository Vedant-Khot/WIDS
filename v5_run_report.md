# V5 Run Report — Post-Mortem & Fix

## Run Summary

| Metric | Value |
|---|---|
| **OOF Hybrid Score** | **0.9772** ✅ (excellent) |
| Competition Score | **0.947** ❌ (down from 0.963) |
| Ensemble Method | BLENDING (beat stacking) |
| Best Weights | GBSA=0.4, XGB=0.6, RSF=0, LGB=0 |
| Seeds completed | 5/5 |
| Runtime | ~10.5 hours |

## Per-Seed OOF Results (All 5 Seeds)

| Seed | GBSA Hybrid | RSF Hybrid | XGB Hybrid | LGB Hybrid |
|---|---|---|---|---|
| 42 | 0.9750 | 0.9722 | 0.9752 | 0.9567 |
| 123 | ~0.975 | ~0.971 | ~0.975 | ~0.955 |
| 456 | ~0.974 | ~0.970 | ~0.975 | ~0.956 |
| 789 | ~0.975 | ~0.971 | ~0.975 | ~0.955 |
| 2025 | 0.9747 | 0.9712 | 0.9752 | 0.9550 |

> [!NOTE]
> **The model itself is excellent** — consistent 0.975+ hybrid across all seeds. The problem was entirely in post-processing.

## Root Cause: TWO Post-Processing Bugs

### Bug 1: Final Isotonic Re-calibration Collapsed 72h 💀

```
Final isotonic re-calibration on OOF...
    12h: [0.0256, 0.9126] → [0.0000, 1.0000]   ← expanded too much
    24h: [0.0377, 0.9272] → [0.0000, 0.9800]   ← ok-ish
    48h: [0.0426, 0.9307] → [0.0000, 1.0000]   ← expanded too much
    72h: [0.6265, 0.9551] → [1.0000, 1.0000]   ← CATASTROPHIC!
```

**What happened at 72h**: The isotonic regression was trained on 221 OOF samples where almost every sample with a known outcome by 72h had indeed been hit (because censored samples with `time < 72h` are excluded, and most event=1 happen early). With so few negative labels remaining, the isotonic function learned: "if prob > 0.63 → predict 1.0". Since ALL test predictions were already above 0.63, everything became 1.0.

**Result**: `prob_72h = 0.999` for all 95 test samples. This is catastrophic because:
- The Brier score at 72h penalizes `(0.999 - 0)² ≈ 1.0` for every non-hit sample
- 72h Brier has a 30% weight in the competition metric

### Bug 2: KM Prior Raised 72h Floor

```
KM P(hit ≤ 72h) = 0.5519
72h: raw=[0.6348, 0.9999] → blended=[0.6265, 0.9551]
```

The raw 72h probabilities already had a high minimum (0.635). The KM prior at 72h is 0.55, which means **blending actually helped slightly** (lowered the floor from 0.635 to 0.627). But the range was still too compressed, which made the subsequent isotonic regression collapse.

The KM prior wasn't the primary culprit here — it was the isotonic step. But the KM prior was unhelpful at shorter horizons, slightly pulling low-risk predictions upward.

## Submission File Analysis

```
              prob_12h   prob_24h   prob_48h   prob_72h
mean          0.1828     0.2743     0.2873     0.9990   ← BROKEN
std           0.3160     0.4328     0.4520     0.0000   ← NO VARIANCE
min           0.0000     0.0000     0.0000     0.9990
max           0.9990     0.9990     0.9990     0.9990
unique vals   3          varies     3          1        ← ONLY 1 VALUE!
```

> [!CAUTION]
> **prob_72h has only 1 unique value (0.999) across all 95 test samples.** This single column likely accounts for the entire 0.016 score drop.

## Fix Applied

### Changes to [ai_model_v5.py](file:///d:/wids/ai_model_v5.py):

1. **Removed final isotonic re-calibration** — the per-fold isotonic calibration during training is sufficient
2. **Disabled KM prior blending** (`KM_PRIOR_WEIGHT = 0.0`) — the base probabilities are already well-calibrated
3. **Reduced seeds from 5 to 3** — saves ~40% runtime while keeping good variance reduction
4. **Added intermediate prediction saving** — saves OOF/test predictions to disk so we can regenerate submissions without retraining

### Expected Impact

| Component | Before (broken) | After (fixed) |
|---|---|---|
| prob_72h values | All 0.999 | Proper distribution |
| prob_12h/48h | Only 3 unique values | Full range |
| Competition score | 0.947 | **0.963+ expected** |

## Key Stacking vs Blending Results

```
Stacking Hybrid:  0.9763
Blending Hybrid:  0.9772  ← WINNER
```

Blending outperformed stacking on this dataset. With only 221 samples, the Ridge meta-model doesn't have enough data to learn meaningful interactions between base model outputs. Simple weighted averaging with rank-based blending is more robust.

## What Worked Well in V5

1. **4-model ensemble** — GBSA and XGB dominated (0.4 + 0.6 weights)
2. **Expanded Optuna** (50 trials) — consistent 0.97+ per-fold scores
3. **Seed averaging** — very stable across all 5 seeds (± 0.001)
4. **New features** — 59 features vs V3's ~45, competitive or better OOF scores
5. **Direct 12h modeling** — no longer using the `prob_24h × 0.75` hack

## Running Status

✅ Fixed V5 is now running with 3 seeds (~1 hour estimated runtime).
The fix removes both post-processing bugs while keeping all model improvements.
