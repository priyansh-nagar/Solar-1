"""
gaps.py — Gap detection and interpolation strategy
====================================================
"Gaps" are contiguous runs of unusable cadences (NaN spectrum, outside GTI,
or saturated) in an otherwise continuous time series.

Gap interpolation is intentionally conservative:
  • Gaps ≤ max_gap_s: linear interpolation over total-flux and each band
  • Gaps  > max_gap_s: left as NaN — never fabricate physics over long gaps
  • Interpolated cadences are flagged with QFlag.GAP_FILL

Design note:  The interpolation is applied to scalar light curves (total flux,
energy bands), NOT to the full (n_times, n_channels) spectrum array.  Filling
340-channel spectra would manufacture spectral shape information.  Module 2
(background subtraction) expects NaN channels to remain NaN; only the band
light curves are interpolated for continuity in Module 3/4.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .quality import QFlag


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def find_gaps(
    usable: np.ndarray,        # boolean mask: True = usable cadence
    time_array: np.ndarray,    # (n_times,) seconds
    min_gap_s: float = 1.0,    # minimum gap size to report (seconds)
) -> List[Tuple[int, int, float]]:
    """
    Find contiguous runs of unusable cadences.

    Returns a list of (idx_start, idx_end, duration_s) tuples where
    idx_start..idx_end is the inclusive index range of the gap.
    duration_s is measured as time_array[idx_end] - time_array[idx_start].
    """
    gaps: List[Tuple[int, int, float]] = []
    in_gap = False
    gap_start = 0

    for i, good in enumerate(usable):
        if not good and not in_gap:
            gap_start = i
            in_gap = True
        elif good and in_gap:
            gap_end = i - 1
            duration = float(time_array[gap_end] - time_array[gap_start])
            if duration >= min_gap_s:
                gaps.append((gap_start, gap_end, duration))
            in_gap = False

    # Handle gap that extends to the end of the array
    if in_gap:
        gap_end = len(usable) - 1
        duration = float(time_array[gap_end] - time_array[gap_start])
        if duration >= min_gap_s:
            gaps.append((gap_start, gap_end, duration))

    return gaps


def gap_summary(gaps: List[Tuple[int, int, float]]) -> dict:
    """Human-readable gap statistics."""
    if not gaps:
        return {"n_gaps": 0, "total_gap_s": 0.0, "max_gap_s": 0.0}
    durations = [g[2] for g in gaps]
    return {
        "n_gaps"      : len(gaps),
        "total_gap_s" : float(sum(durations)),
        "max_gap_s"   : float(max(durations)),
        "min_gap_s"   : float(min(durations)),
        "median_gap_s": float(np.median(durations)),
    }


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(
    values: np.ndarray,        # (n_times,) float — NaN where usable=False
    usable: np.ndarray,        # boolean mask
    quality: np.ndarray,       # uint8 quality flags (modified in-place)
    max_gap_s: float = 10.0,   # do not interpolate gaps longer than this
    time_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Linearly interpolate short gaps in a scalar light curve.

    Parameters
    ----------
    values     : 1-D array of physical values (total flux or band counts).
                 NaN marks unusable cadences.
    usable     : boolean mask from quality.is_usable()
    quality    : uint8 quality flag array (QFlag.GAP_FILL is added for
                 filled cadences — modifies the array in-place)
    max_gap_s  : maximum gap length (seconds) to fill; longer gaps stay NaN
    time_array : timestamps; if None, cadences are assumed to be index-spaced

    Returns
    -------
    filled : copy of *values* with short gaps linearly interpolated
    """
    if time_array is None:
        time_array = np.arange(len(values), dtype=float)

    filled = values.copy()

    gaps = find_gaps(usable, time_array)
    for (i_start, i_end, duration) in gaps:
        if duration > max_gap_s:
            # Gap too long — leave as NaN
            continue

        # Find anchor points just outside the gap
        anchor_lo = i_start - 1
        anchor_hi = i_end + 1

        if anchor_lo < 0 or anchor_hi >= len(values):
            # Gap at edge of array — cannot interpolate
            continue

        t_lo = time_array[anchor_lo]
        t_hi = time_array[anchor_hi]
        v_lo = values[anchor_lo]
        v_hi = values[anchor_hi]

        if np.isnan(v_lo) or np.isnan(v_hi):
            # Anchors are themselves bad — skip
            continue

        # Linear interpolation
        t_range = t_hi - t_lo
        for idx in range(i_start, i_end + 1):
            frac = (time_array[idx] - t_lo) / t_range
            filled[idx] = v_lo + frac * (v_hi - v_lo)
            quality[idx] |= QFlag.GAP_FILL

    return filled


def interpolate_all_bands(
    band_arrays: dict,            # {band_name: (n_times,) array}
    usable: np.ndarray,
    quality: np.ndarray,          # modified in-place
    max_gap_s: float = 10.0,
    time_array: Optional[np.ndarray] = None,
) -> dict:
    """
    Apply interpolate_gaps() to every band in *band_arrays*.

    Returns a new dict with the same keys and filled arrays.
    """
    return {
        band: interpolate_gaps(
            arr, usable, quality, max_gap_s, time_array
        )
        for band, arr in band_arrays.items()
    }
