---
name: Module 3 detector design decisions
description: Two-channel voter architecture; why Band C not hardness_ratio
---

## Two-channel voter
- Soft channel: `excess_A > adaptive_threshold(rolling_stats_of_excess_A_itself)`
- Hard channel: `band_C (>3 keV cts/s) > adaptive_threshold(rolling_stats_of_band_C)`
- Both must agree for ≥30 s (sustained trigger)
- 5-min min gap between distinct events (merge closer ones)

## Why Band C, not hardness_ratio
hardness_ratio = Band C / Band A. Band A rises ~60× during flares, Band C only ~10×,
so the ratio DECREASES during flares — wrong sign. Band C in raw cts/s is zero at quiet
and clearly non-zero at flares. This is the correct two-channel discriminant.

**Why:** Discovered by running diagnostics on X6.4 day; hardness_ratio never triggered
because the floor (0.05) exceeded the actual peak ratio (~0.021). Band C directly
fixes this — quiet=0 cts/s, flare=441 cts/s, easily separated.

## Adaptive CFAR threshold design
- Threshold = rolling_mean(signal) + σ × rolling_std(signal); floor applied
- Rolling stats computed on the SIGNAL ITSELF (excess_A or band_C), not raw counts
- excess_A window: 300s (5-min); band_C window: 900s (15-min)
- Default σ=3.0 for production; 2.0 works for clean synthetic tests (CFAR window
  includes flare rise edge in synthetic data, elevating rolling mean)

## Evaluation note
- TPR is cadence-level (fraction of GOES flare cadences detected), not event-level
- Cadence-level TPR looks low (0.001-0.003) because events cover ~60-140s but
  GOES labels thousands of cadences; event-level precision is 0.67-1.0
- Lead time (onset→peak) is the headline metric, not TPR
