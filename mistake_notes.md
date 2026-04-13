# WiDS 2025 — Mistake Notes & Lessons Learned

## Mistake #1: Final Isotonic Re-calibration on Tiny Data 💀

### What happened
Added a "final isotonic re-calibration" step that re-fit isotonic regression on OOF predictions to adjust test probabilities. At 72h horizon, this mapped **every** prediction to 1.0.

### Why it failed
- Only **221 training samples**. After excluding censored samples before 72h, only ~100-150 samples remained for fitting.
- At 72h, most surviving samples had `label = 1` (events that hit) because censored-before-72h samples were excluded. The isotonic function learned: "everything → 1.0"
- The OOF probabilities were already well-calibrated from per-fold isotonic regression during training. Double-calibrating overfits on small data.

### Lesson
> **Never add a post-processing step that re-fits a model on the same OOF data without cross-validating it too.** On small datasets, any additional fitting layer can collapse. If you must re-calibrate, use nested CV or at minimum check that the output has variance.

### Rule
```
✅ DO: Calibrate within each CV fold (as we did for XGB/LGB)
❌ DON'T: Add another calibration layer on top of already-calibrated OOF predictions
❌ DON'T: Fit isotonic regression when one class has < 20 samples
```

---

## Mistake #2: Not Sanity-Checking the Submission File 🔍

### What happened
The script printed submission statistics at the end, which clearly showed `prob_72h: std ≈ 0`, but we didn't catch it before submitting.

### The warning signs we missed
```
prob_72h: mean=0.999, std=0.000, min=0.999, max=0.999
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         This screams "something is wrong"
```

### Lesson
> **Always run automated sanity checks BEFORE saving the submission.** A single assert would have caught this.

### Rule — Add to every future submission script
```python
# SUBMISSION SANITY CHECKS (add before saving)
for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
    assert submission[col].nunique() > 5, f"ALERT: {col} has only {submission[col].nunique()} unique values!"
    assert submission[col].std() > 0.01, f"ALERT: {col} has near-zero std={submission[col].std():.6f}"
    assert submission[col].min() < 0.5, f"ALERT: {col} min is too high: {submission[col].min():.4f}"
    assert submission[col].max() > 0.5, f"ALERT: {col} max is too low: {submission[col].max():.4f}"
print("✅ All sanity checks passed!")
```

---

## Mistake #3: Adding Too Many Changes at Once 🎯

### What happened
V5 added **10+ changes** simultaneously over V3:
1. New features (6 features)
2. Fixed 12h probability  
3. Stacking meta-model
4. KM prior blending
5. Seed averaging
6. Isotonic re-calibration
7. Expanded Optuna
8. Low-res interaction features
9. Advanced features
10. Intermediate prediction saving

When the score dropped, we couldn't immediately tell WHICH change caused it.

### Lesson
> **Change one thing at a time. Measure. Then change the next thing.** This is the single most important rule in competitive ML. The competitor notebooks we studied emphasized "Change Logs" — each experiment changes ONE variable.

### Rule
```
1. Start from your best submission
2. Change ONE thing
3. Run, submit, record score
4. If better → keep. If worse → revert.
5. Repeat
```

---

## Mistake #4: KM Prior Blending Without Validation ⚖️

### What happened
Added Kaplan-Meier prior blending (`10% KM + 90% model`). The KM prior at 72h is 0.5519, which means even a sample with near-zero fire risk gets pulled toward a 55% hit probability.

### Why it was problematic
- The KM prior represents the **population average**, not individual risk. Blending it uniformly adds bias to extreme predictions.
- For low-risk samples: pulls probability UP (bad)
- For high-risk samples: pulls probability DOWN (slightly bad)
- Net effect: compresses the prediction range, reduces discrimination

### Lesson
> **Population-level statistics should not be blended uniformly into individual predictions.** If you want to use a prior, apply it differentially (e.g., only to uncertain predictions) or validate that it actually improves your metric on OOF.

---

## Mistake #5: Not Saving Intermediate Predictions 💾

### What happened
The 10+ hour training run produced excellent OOF and test predictions. But when we discovered the post-processing bug, we had NO WAY to regenerate a fixed submission without re-training from scratch.

### Lesson  
> **Always save intermediate predictions to disk.** The training loop is the expensive part. Post-processing is cheap and can be iterated quickly.

### Rule
```python
# Add IMMEDIATELY after the CV loop finishes:
import pickle
pickle.dump({
    'oof_preds': oof_preds,
    'test_preds': test_preds,
    'y_event': y_event,
    'y_time': y_time,
}, open('intermediate_preds.pkl', 'wb'))
```

---

## Mistake #6: Stacking on 221 Samples Adds Complexity Without Benefit 📊

### What happened
Built a Ridge stacking meta-model using 20 meta-features (4 models × 5 columns each). Stacking scored 0.9763 vs blending's 0.9772 — **stacking lost**.

### Why
- With 221 samples and 20 meta-features, the Ridge model doesn't have enough data to learn useful interactions
- Simple weighted blending is more robust — it has only 3 free parameters (4 weights summing to 1)
- Stacking shines with 10K+ samples where the meta-model can learn complex patterns

### Lesson
> **On small datasets, simpler ensemble methods (blending, averaging) beat more complex ones (stacking, learned combiners).** Only add stacking if you have >2000 samples.

---

## Golden Rules for Future Iterations

### Before Running
- [ ] **One change at a time** — never bundle multiple experiments
- [ ] Review post-processing steps — do they need their own CV?
- [ ] Set `N_SEEDS=1` for quick tests, increase only for final submissions

### After Running  
- [ ] **Check submission stats** — std > 0, min < max, reasonable ranges
- [ ] Compare with previous best submission side-by-side
- [ ] Save intermediate predictions to disk

### Before Submitting
- [ ] Run sanity checks (unique values, range, monotonicity)
- [ ] Eyeball 10 random rows — do the probabilities make intuitive sense?
- [ ] Compare with sample_submission format exactly

### Experiment Log Template
```
| Version | Change Made | OOF Hybrid | Competition Score | Notes |
|---------|-------------|-----------|-------------------|-------|
| V3      | baseline    | 0.970     | 0.963             | best  |
| V5      | +10 changes | 0.977     | 0.947             | broken 72h |
| V5-fix  | removed postproc | ???  | ???               | running |
```
