"""
background.py — SNIP background estimation
===========================================
SNIP = Statistics-sensitive Non-linear Iterative Peak-clipping

Reference: M. Morháč et al., Nuclear Instruments and Methods in Physics
Research A 401 (1997) 113–132.  The algorithm is the de-facto standard in
nuclear and X-ray spectroscopy for background subtraction.

Why SNIP for solar X-ray time series?
--------------------------------------
The solar soft X-ray background has two slowly varying components:
  (a) The quiet-sun thermal emission (timescale: hours–days).
  (b) Gradual coronal heating trends during active periods (timescale: minutes).

A simple polynomial or running-median underestimates (a) and overfits (b).
SNIP does neither: it iteratively clips upward fluctuations (peaks = flares)
while letting the slowly varying floor converge to the true background.

The sqrt-space transform
------------------------
SNIP applied directly to counts over-clips low-count cadences because small
Poisson fluctuations appear large relative to the background.  Working in
sqrt(counts) space linearises the Poisson noise (variance → constant), giving
equal weight to quiet and active periods.  After SNIP, we square back.

Implementation detail
---------------------
For a window half-width m, the update at position i is:

    v[i] ← min(v[i],  (v[i-m] + v[i+m]) / 2)

Iterating m from 1 to M (increasing window) progressively clips broader and
broader peaks.  M should be chosen to be wider than the longest expected flare
duration so that even long-duration events are removed from the background.
M = 300 (= 5 minutes at 1-s cadence) is suitable for SoLEXS Band A.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# Default SNIP parameters for SoLEXS 1-s light curves
DEFAULT_M         = 300    # max window half-width (300 s = 5 min)
DEFAULT_CLIPPING_DIRECTION = "increasing"   # standard SNIP


def snip_background(
    values: np.ndarray,
    M: int = DEFAULT_M,
    sqrt_space: bool = True,
    min_floor: float = 1e-6,
) -> np.ndarray:
    """
    Estimate the slowly varying background using the SNIP algorithm.

    Parameters
    ----------
    values      : (n_times,) float array.  NaN positions are handled by
                  forward/backward fill before SNIP and restored after.
    M           : maximum window half-width in samples.  Should be ≥ the
                  longest expected flare duration in samples.
                  M=300 → 5 min at 1-s cadence (covers M- and X-class flares).
    sqrt_space  : if True (default), apply SNIP in sqrt(counts) space to
                  properly weight Poisson statistics.
    min_floor   : clip values below this before sqrt to avoid sqrt(negative).

    Returns
    -------
    background : (n_times,) float array, same length as *values*.
                 NaN positions from the input remain NaN.
    """
    nan_mask = np.isnan(values)
    n = len(values)

    if nan_mask.all():
        return np.full(n, np.nan)

    # ── Fill NaN positions for the algorithm (forward then backward fill) ──
    filled = values.copy()
    filled = _fill_nan(filled)

    # ── Floor at min_floor so sqrt is safe ──
    filled = np.maximum(filled, min_floor)

    # ── Transform to sqrt space ──
    if sqrt_space:
        v = np.sqrt(filled)
    else:
        v = filled.copy()

    # ── SNIP iterations (increasing window half-width 1 → M) ──
    for m in range(1, M + 1):
        if 2 * m >= n:
            break   # window larger than array — stop early

        # Vectorised: clip centre values to average of left/right neighbours
        # Positions i ∈ [m, n-m-1]
        left  = v[: n - 2 * m]    # v[i-m] for i = m..n-m-1
        right = v[2 * m :]        # v[i+m] for i = m..n-m-1
        avg   = (left + right) * 0.5

        # Only clip downward (peak removal), never push values up
        v[m : n - m] = np.minimum(v[m : n - m], avg)

    # ── Back-transform ──
    background = (v ** 2) if sqrt_space else v

    # ── Restore NaN ──
    background[nan_mask] = np.nan

    return background


def subtract_background(
    values: np.ndarray,
    background: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute background-subtracted residual and normalised excess.

    Parameters
    ----------
    values      : smoothed light curve (n_times,)
    background  : SNIP background estimate (n_times,)
    epsilon     : small floor added to background before division to prevent
                  division-by-zero in very quiet periods

    Returns
    -------
    residual        : values - background
    normalised_excess: residual / (background + epsilon)
                       A value of 1.0 means the signal doubled above background.
                       Values ≥ ~5 are typical for C-class flares.
    """
    residual = values - background
    normalised_excess = residual / (np.abs(background) + epsilon)
    return residual, normalised_excess


def snip_background_bands(
    band_arrays: dict,
    M: int = DEFAULT_M,
    sqrt_space: bool = True,
    min_floor: float = 1e-6,
) -> dict:
    """
    Apply snip_background to every band in a {band_name: array} dict.
    """
    return {
        name: snip_background(arr, M, sqrt_space, min_floor)
        for name, arr in band_arrays.items()
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaN values by forward-fill then backward-fill.
    If the entire array is NaN, return as-is.
    """
    out = arr.copy()
    # Forward fill
    mask = np.isnan(out)
    idx  = np.where(~mask, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out[mask] = out[idx[mask]]
    # Backward fill for leading NaNs
    mask = np.isnan(out)
    if mask.any():
        idx2 = np.where(~mask, np.arange(len(out)), len(out) - 1)
        idx2_rev = len(out) - 1 - np.minimum.accumulate((len(out) - 1 - idx2)[::-1])[::-1]
        out[mask] = out[idx2_rev[mask]]
    return out
