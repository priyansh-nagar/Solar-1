---
name: Module 4 lead-time evaluation
description: Root-cause of lead_time=0 on TEST and the evaluate.py fix applied
---

## The bug

`evaluate_model()` passed `opt_t` (F1-optimal threshold, ~0.79) to `compute_lead_times()`.
But lead time should be measured at the same operating point as TPR@FAR — the FAR-based threshold.
For TEST the FAR threshold is 0.782; the pre-flare p60 signal peaks at 0.44 → no alerts at either threshold.

## The genuine training limitation

TEST day 2024-02-09 X3.4 has two onsets:
- idx=0: day starts already in M+ (X3.4 onset is near start of observations). No preceding windows → unmeasurable. This is not a bug.
- idx=721: real C→M transition. Pre-onset windows (691-720) output p60≈0.37-0.44. FAR threshold=0.782. Signal is elevated vs quiet background but insufficient to cross the operating threshold.

Lead time=0 on TEST is genuine at this operating point. It is NOT purely an evaluation bug.

**Why:**
The 60-min head is poorly calibrated on TEST (ECE=0.288 raw). TRAIN prior (36% M+) vs TEST prior (21.8% M+) mismatch inflates the model's baseline output, raising the FAR threshold needed for 0.5 FA/hr. This squeezes out the pre-onset signal.

## The fix applied to evaluate.py

1. Added `_far_operating_threshold()` — derives threshold from FAR budget (same logic as `tpr_at_far()`).
2. `evaluate_model()` now passes `far_op_t` (not `opt_t`) to `compute_lead_times()`.
3. `compute_lead_times()` now:
   - Skips onset at idx=0 explicitly with a count field
   - Returns `min_threshold_for_lead` (max pre-onset p) for diagnostics
   - Verbose output prints the gap: "Pre-onset signal max p=X vs FAR threshold Y"

## How to fix lead time for real

Options in priority order:
1. Add another TRAIN day with a slow-rise X-class precursor visible in SDD2 30 min before peak. This is the only thing that directly teaches the model the pre-flare pattern.
2. Apply calibrated probabilities for lead time (isotonic calibrator reduces ECE to 0.097). After calibration, FAR threshold may drop enough that p60=0.44 clears it.

## Current corrected metrics (after evaluate.py fix)

- TEST TPR@FAR0.5: 0.721
- TEST X-recall: 0.856
- TEST ECE raw: 0.288 / calibrated: 0.097
- TEST Lead time: 0.0 min (1/1 measurable onset missed; pre-onset max p=0.44 < FAR thr 0.78)
- VAL Lead time: 30.0 min median (3/6 measurable events)
