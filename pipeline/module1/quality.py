"""
quality.py — Quality flag filtering and GTI application
=========================================================
Provides functions to:
  • Parse GTI (Good Time Interval) tables from .gti FITS extensions
  • Build a per-second boolean quality mask from GTI intervals + NaN detection
  • Combine GTI mask, NaN mask, and saturation mask into a single QUALITY array
  • Encode flag bits so downstream modules can trace why a cadence was rejected

QUALITY FLAG BIT DEFINITIONS
─────────────────────────────
  Bit 0  (0x01) IN_GTI     — cadence is inside a GTI interval (good)
  Bit 1  (0x02) NAN_ROW    — all spectrum channels are NaN (missing readout)
  Bit 2  (0x04) SATURATED  — saturation/pile-up detected (see channels.py)
  Bit 3  (0x08) GAP_FILL   — cadence was filled in by interpolation (module gaps.py)
  Bit 4  (0x10) LOW_EXPOSURE— exposure < expected cadence duration (partial readout)

A cadence is considered USABLE if bit 0 is set (in_gti) AND bits 1,2 are clear.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quality flag bit masks
# ---------------------------------------------------------------------------

class QFlag:
    IN_GTI       = 0x01
    NAN_ROW      = 0x02
    SATURATED    = 0x04
    GAP_FILL     = 0x08
    LOW_EXPOSURE = 0x10


def is_usable(quality: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask: True where a cadence is usable for science.

    A cadence is usable if:
      • It falls inside a GTI interval (IN_GTI bit set), AND
      • Its spectrum is not all-NaN (NAN_ROW bit clear), AND
      • It is not flagged as saturated (SATURATED bit clear).

    Gap-filled cadences (GAP_FILL bit) are accepted by default because
    Module 2 (background subtraction) will handle them separately.
    """
    in_gti    = (quality & QFlag.IN_GTI)   != 0
    not_nan   = (quality & QFlag.NAN_ROW)  == 0
    not_sat   = (quality & QFlag.SATURATED)== 0
    return in_gti & not_nan & not_sat


# ---------------------------------------------------------------------------
# GTI parsing
# ---------------------------------------------------------------------------

def parse_gti(hdul) -> List[Tuple[float, float]]:
    """
    Extract Good Time Intervals from an open HDU list.

    Returns a list of (t_start, t_stop) tuples in the same time system as
    the FITS file (seconds since MJD 40587 = Unix epoch).

    Handles both the SDD2 case (5 intervals, float64 columns) and the SDD1
    case (0 rows / empty table) gracefully.
    """
    try:
        gti_hdu = hdul["GTI"]
    except KeyError:
        # Try HDU index 1 if extension name lookup fails
        if len(hdul) > 1:
            gti_hdu = hdul[1]
        else:
            return []

    data = gti_hdu.data
    if data is None or len(data) == 0:
        return []

    try:
        starts = data["START"].astype(float)
        stops  = data["STOP"].astype(float)
    except (KeyError, ValueError):
        return []

    # Sanity check: stop must be after start
    valid = stops > starts
    return list(zip(starts[valid], stops[valid]))


def gti_mask(
    time_array: np.ndarray,
    gti_intervals: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Return a boolean array (len = len(time_array)) that is True wherever
    *time_array* falls inside any GTI interval [t_start, t_stop].

    Uses a vectorised approach: O(n_times × n_intervals).  For typical
    SoLEXS day files (86400 × 5 intervals) this is fast.
    """
    mask = np.zeros(len(time_array), dtype=bool)
    for t_start, t_stop in gti_intervals:
        mask |= (time_array >= t_start) & (time_array <= t_stop)
    return mask


# ---------------------------------------------------------------------------
# Combined quality flag array
# ---------------------------------------------------------------------------

def build_quality_flags(
    time_array: np.ndarray,           # (n_times,) seconds
    counts_spectrum: np.ndarray,      # (n_times, n_channels)  may be None
    gti_intervals: List[Tuple[float, float]],
    saturation_mask: Optional[np.ndarray] = None,   # (n_times,) bool
    exposure: Optional[np.ndarray] = None,          # (n_times,) seconds
    cadence_s: float = 1.0,
) -> np.ndarray:
    """
    Build an integer quality flag array for every cadence in *time_array*.

    Parameters
    ----------
    time_array       : 1-D array of timestamps (Unix seconds, 1-s cadence)
    counts_spectrum  : full spectrum array or None (if only .lc is available)
    gti_intervals    : list of (start, stop) from parse_gti()
    saturation_mask  : optional boolean array from channels.flag_saturated_rows()
    exposure         : optional per-cadence exposure time (seconds)
    cadence_s        : expected cadence duration (default 1.0 s)

    Returns
    -------
    quality : (n_times,) uint8 array with QFlag bits set
    """
    n = len(time_array)
    quality = np.zeros(n, dtype=np.uint8)

    # Bit 0: IN_GTI
    if gti_intervals:
        quality[gti_mask(time_array, gti_intervals)] |= QFlag.IN_GTI

    # Bit 1: NAN_ROW — all spectrum channels are NaN
    if counts_spectrum is not None:
        nan_rows = np.all(np.isnan(counts_spectrum), axis=1)
        quality[nan_rows] |= QFlag.NAN_ROW

    # Bit 2: SATURATED
    if saturation_mask is not None:
        quality[saturation_mask] |= QFlag.SATURATED

    # Bit 4: LOW_EXPOSURE — exposure meaningfully shorter than cadence
    if exposure is not None:
        low_exp = exposure < (cadence_s * 0.95)   # allow 5% tolerance
        quality[low_exp] |= QFlag.LOW_EXPOSURE

    return quality


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def quality_summary(quality: np.ndarray) -> dict:
    """
    Return a human-readable summary dict of quality flag counts.
    Useful for logging and sanity checks.
    """
    n = len(quality)
    return {
        "total_cadences"   : n,
        "in_gti"           : int(np.sum((quality & QFlag.IN_GTI)   != 0)),
        "nan_row"          : int(np.sum((quality & QFlag.NAN_ROW)  != 0)),
        "saturated"        : int(np.sum((quality & QFlag.SATURATED)!= 0)),
        "gap_fill"         : int(np.sum((quality & QFlag.GAP_FILL) != 0)),
        "low_exposure"     : int(np.sum((quality & QFlag.LOW_EXPOSURE)!=0)),
        "usable"           : int(np.sum(is_usable(quality))),
        "usable_fraction"  : float(np.sum(is_usable(quality))) / n if n else 0.0,
    }
