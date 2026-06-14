"""
align.py — Temporal alignment between instruments and detectors
================================================================
Aligns SoLEXS SDD1/SDD2 and HEL1OS to a common reference time grid with
millisecond-precision handling.

The Aditya-L1 instruments share the satellite clock so timestamps are already
in the same reference frame (MJD 40587 / Unix epoch).  However, in practice:

  • The two SoLEXS detectors (SDD1, SDD2) may have slightly different start
    times and cadences due to independent FPGA readout cycles.
  • HEL1OS time bins may differ in cadence (it has a faster readout: ~0.1 s).
  • Instrument resets create sub-second time offsets.

Strategy
--------
1. Build a master time grid at the highest common cadence (1 s for SoLEXS).
2. For each instrument/detector, find the nearest grid point for every
   timestamp using searchsorted (< 0.5-cadence tolerance).
3. Store the alignment residual so downstream modules can assess quality.
4. Rows that cannot be aligned within tolerance are masked as NaN.

The output is a consistent (time,) index that maps each instrument's data
to the shared grid — the ingest.py assembler uses this to build the merged
xarray Dataset.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Master grid construction
# ---------------------------------------------------------------------------

def build_master_grid(
    t_start: float,
    t_stop: float,
    cadence_s: float = 1.0,
) -> np.ndarray:
    """
    Build a regular time grid from *t_start* to *t_stop* inclusive.

    Parameters
    ----------
    t_start, t_stop : boundary times in seconds (Unix / MJD40587)
    cadence_s       : grid spacing in seconds

    Returns
    -------
    1-D float64 array of evenly-spaced times.
    """
    n = int(round((t_stop - t_start) / cadence_s)) + 1
    return t_start + np.arange(n) * cadence_s


def align_to_grid(
    source_times: np.ndarray,
    grid: np.ndarray,
    tolerance_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map *source_times* to indices in *grid* using nearest-neighbour lookup.

    Parameters
    ----------
    source_times : (n_source,) array of input timestamps
    grid         : (n_grid,) regular master time grid
    tolerance_s  : max allowed residual in seconds.  Defaults to half the
                   grid cadence.  Timestamps outside tolerance → index = -1.

    Returns
    -------
    grid_indices : (n_source,) int64 — index into *grid* for each source time,
                   or -1 if outside tolerance
    residuals_ms : (n_source,) float64 — alignment residual in milliseconds
                   (positive = source is later than grid point)
    valid_mask   : (n_source,) bool — True where alignment is within tolerance
    """
    if len(grid) < 2:
        raise ValueError("Master grid must have at least 2 points")

    cadence = float(grid[1] - grid[0])
    if tolerance_s is None:
        tolerance_s = cadence * 0.5

    # searchsorted finds the insertion point; we want nearest neighbour
    idx = np.searchsorted(grid, source_times, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)

    # Compare with left and right neighbours
    left_idx  = np.clip(idx - 1, 0, len(grid) - 1)
    right_idx = idx

    left_res  = np.abs(source_times - grid[left_idx])
    right_res = np.abs(source_times - grid[right_idx])

    # Pick the closer neighbour
    use_left = left_res <= right_res
    best_idx = np.where(use_left, left_idx, right_idx)
    best_res = np.where(use_left, source_times - grid[left_idx],
                                  source_times - grid[right_idx])

    valid_mask    = np.abs(best_res) <= tolerance_s
    grid_indices  = np.where(valid_mask, best_idx, -1).astype(np.int64)
    residuals_ms  = best_res * 1000.0   # convert to ms

    return grid_indices, residuals_ms, valid_mask


# ---------------------------------------------------------------------------
# Multi-detector alignment
# ---------------------------------------------------------------------------

def align_detectors(
    detector_times: Dict[str, np.ndarray],
    grid: np.ndarray,
    tolerance_s: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Align multiple detectors to a single master grid.

    Parameters
    ----------
    detector_times : {detector_label: time_array} for each detector
    grid           : master time grid from build_master_grid()
    tolerance_s    : per-timestamp tolerance (default: half-cadence)

    Returns
    -------
    {detector_label: {
        "grid_indices"  : (n_source,) int64,
        "residuals_ms"  : (n_source,) float64,
        "valid_mask"    : (n_source,) bool,
        "coverage_frac" : float,         # fraction of grid covered
        "max_residual_ms": float,
    }}
    """
    results = {}
    for label, times in detector_times.items():
        g_idx, res_ms, valid = align_to_grid(times, grid, tolerance_s)
        # Coverage: what fraction of the master grid has a valid measurement?
        covered = np.zeros(len(grid), dtype=bool)
        covered[g_idx[valid]] = True
        results[label] = {
            "grid_indices"    : g_idx,
            "residuals_ms"    : res_ms,
            "valid_mask"      : valid,
            "coverage_frac"   : float(covered.sum()) / len(grid),
            "max_residual_ms" : float(np.abs(res_ms[valid]).max()) if valid.any() else 0.0,
        }
    return results


def reindex_to_grid(
    values: np.ndarray,
    grid_indices: np.ndarray,
    valid_mask: np.ndarray,
    grid_size: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Scatter *values* from source positions into a grid-sized array.

    Parameters
    ----------
    values      : (n_source, ...) array of data (1-D or 2-D)
    grid_indices: (n_source,) int64 — index in grid for each source row
    valid_mask  : (n_source,) bool
    grid_size   : length of the output array
    fill_value  : value for grid positions with no matching source row

    Returns
    -------
    (grid_size, ...) array with values placed at their grid positions.
    Positions with no data contain *fill_value*.
    """
    if values.ndim == 1:
        out = np.full(grid_size, fill_value, dtype=float)
        out[grid_indices[valid_mask]] = values[valid_mask].astype(float)
    elif values.ndim == 2:
        n_cols = values.shape[1]
        out = np.full((grid_size, n_cols), fill_value, dtype=float)
        out[grid_indices[valid_mask]] = values[valid_mask].astype(float)
    else:
        raise ValueError("values must be 1-D or 2-D")
    return out
