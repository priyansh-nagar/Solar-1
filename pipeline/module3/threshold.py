"""
threshold.py — Adaptive detection threshold
============================================
The threshold at each cadence is:

    thr_excess_A[t] = max(FLOOR, roll_mean_excess_A[t] + sigma × roll_std_excess_A[t])

Design rationale
----------------
• Fixed thresholds fail because solar background varies across the day, the
  solar cycle, and between quiet / active periods.

• We compute rolling statistics on excess_A ITSELF (not on raw count rates),
  so the threshold is in the same dimensionless unit as the signal.  Using
  rolling_std_5min (which is in raw cts/s) against excess_A (a ratio) produces
  a threshold 10,000× too high — do NOT repeat that mistake.

• The rolling mean acts as a slowly-moving baseline; sigma × rolling_std sets
  the noise floor above it.  A genuine flare onset will push excess_A many
  sigma above this baseline.

• FLOOR prevents the threshold from collapsing to near-zero on ultra-quiet
  days, which would make cosmic-ray single-cadence hits look like flares.

Hard channel (Band C counts)
------------------------------
SoLEXS Band C (>3 keV, harder X-rays) is essentially zero during quiet periods
and rises sharply during real flares.  It is used as the confirmation channel
in the two-channel voter.

  thr_band_C[t] = max(FLOOR_C, roll_mean_C[t] + sigma_C × roll_std_C[t])

Why Band C, not hardness_ratio (Band C / Band A)?
  • During a flare, Band A rises ~60×, Band C rises ~10×.
  • The ratio Band C / Band A therefore DECREASES during a flare — wrong sign.
  • Band C in absolute counts/s is zero at quiet and non-zero during flares.
  • A cosmic ray hits Band C for a single cadence; a flare rises for ≥30 s.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Low-level rolling stats (O(n), no pandas dependency)
# ---------------------------------------------------------------------------

def _rolling_mean_std(
    x: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-looking rolling mean and std over *window* samples.

    NaN inputs are treated as 0 for the purpose of the rolling window
    (they do not propagate into the output).
    """
    x_safe = np.where(np.isnan(x), 0.0, x)
    cs     = np.concatenate([[0.0], np.cumsum(x_safe)])
    cs_sq  = np.concatenate([[0.0], np.cumsum(x_safe ** 2)])

    ends   = np.arange(1, len(x) + 1)
    starts = np.maximum(0, ends - window)
    counts = (ends - starts).astype(float)

    roll_mean = (cs[ends] - cs[starts]) / counts
    roll_var  = (cs_sq[ends] - cs_sq[starts]) / counts - roll_mean ** 2
    roll_std  = np.sqrt(np.maximum(roll_var, 0.0))

    return roll_mean, roll_std


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def adaptive_excess_threshold(
    excess_A: np.ndarray,
    window_s: int = 300,
    cadence_s: float = 1.0,
    base_sigma: float = 3.0,
    floor: float = 1.0,
) -> np.ndarray:
    """
    Compute per-cadence adaptive threshold for excess_A.

    The threshold at time t is:
        max(floor, roll_mean[t] + base_sigma × roll_std[t])

    computed over a backward window of *window_s* seconds.

    Parameters
    ----------
    excess_A  : (n_times,) float — dimensionless excess_A values
    window_s  : rolling window in seconds (default 300 = 5 min)
    cadence_s : seconds per sample
    base_sigma: σ multiplier above rolling mean (default 3.0)
    floor     : absolute minimum threshold (default 1.0 ≈ B-class proxy)

    Returns
    -------
    threshold : (n_times,) float, same shape as excess_A
    """
    w = max(1, int(window_s / cadence_s))
    roll_mean, roll_std = _rolling_mean_std(excess_A, w)
    thr = roll_mean + base_sigma * roll_std
    return np.maximum(thr, floor)


def adaptive_band_C_threshold(
    band_C: np.ndarray,
    window_s: int = 900,
    cadence_s: float = 1.0,
    sigma: float = 3.0,
    floor: float = 1.0,
) -> np.ndarray:
    """
    Compute per-cadence adaptive threshold for the Band C count rate.

    During quiet periods Band C ≈ 0 cts/s; during flares it rises sharply.
    A cadence is "hard-channel ON" when band_C[t] > thr_band_C[t].

    Parameters
    ----------
    band_C    : (n_times,) float — Band C count rate in cts/s
    window_s  : rolling window in seconds (default 900 = 15 min)
    cadence_s : seconds per sample
    sigma     : σ multiplier (default 3.0)
    floor     : minimum threshold in cts/s (default 1.0)

    Returns
    -------
    thr_band_C : (n_times,) float
    """
    w = max(1, int(window_s / cadence_s))
    roll_mean, roll_std = _rolling_mean_std(band_C, w)
    thr = roll_mean + sigma * roll_std
    return np.maximum(thr, floor)


def compute_all_thresholds(
    preprocess_ds,
    prefix: str = "solexs_sdd2",
    base_sigma: float = 3.0,
    floor_excess_A: float = 1.0,
    band_C_sigma: float = 3.0,
    band_C_floor: float = 1.0,
    cadence_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Extract threshold arrays from a preprocessed Dataset.

    Returns
    -------
    dict with keys:
      "excess_A"       : (n,) adaptive threshold for excess_A
      "band_C"         : (n,) adaptive threshold for Band C (hard channel)
      "raw_excess_A"   : (n,) raw excess_A values
      "raw_band_C"     : (n,) raw Band C smooth values (hard channel signal)
      "raw_hr"         : (n,) raw hardness_ratio (kept for reference)
      "raw_deriv"      : (n,) raw derivative_1s values
      "quality"        : (n,) uint8 quality flags
    """
    def _get(name: str) -> np.ndarray | None:
        key = f"{prefix}_{name}"
        if key in preprocess_ds.data_vars:
            return preprocess_ds[key].values.copy()
        return None

    excess_A = _get("excess_A")
    deriv    = _get("derivative_1s")
    quality  = _get("quality")
    hr       = _get("hardness_ratio")   # kept for logging / reference

    if excess_A is None:
        raise KeyError(
            f"Variable '{prefix}_excess_A' not found in Dataset. "
            "Ensure preprocess_day() has been run first."
        )

    # Adaptive threshold: computed from rolling stats of excess_A itself
    thr_excess = adaptive_excess_threshold(
        excess_A, cadence_s=cadence_s,
        base_sigma=base_sigma, floor=floor_excess_A,
    )

    # Hard channel: Band C smooth (>3 keV, zero at quiet, non-zero at flares)
    # Try band_C_smooth, then band_C, then residual_C (in priority order)
    # NOTE: use explicit None checks — numpy arrays have ambiguous truth values
    band_C_signal = None
    for cand in ["band_C_smooth", "band_C", "residual_C"]:
        v = _get(cand)
        if v is not None:
            band_C_signal = v
            break

    thr_band_C = None
    if band_C_signal is not None:
        band_C_nonneg = np.maximum(band_C_signal, 0.0)
        thr_band_C = adaptive_band_C_threshold(
            band_C_nonneg, cadence_s=cadence_s,
            sigma=band_C_sigma, floor=band_C_floor,
        )

    return {
        "excess_A":     thr_excess,
        "band_C":       thr_band_C,
        "raw_excess_A": excess_A,
        "raw_band_C":   band_C_signal,
        "raw_hr":       hr,
        "raw_deriv":    deriv,
        "quality":      quality,
    }
