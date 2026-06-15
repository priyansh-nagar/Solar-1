"""
goes_crossmatch.py — High-level GOES label builder
====================================================
Wraps fetch_and_crossmatch() and adds a per-cadence `goes_class` variable
directly to the preprocessed Dataset.  This is the interface Module 4
expects:

    from pipeline.module3.goes_crossmatch import build_goes_labels

    pds = build_goes_labels(pds, date=datetime(2024, 2, 22))
    # pds now has pds["goes_class"] — (n_times,) int8
    # 0=A, 1=B, 2=C, 3=M, 4=X, -1=unknown

Fallback behaviour
------------------
When GOES download fails (no network, future date, NOAA outage):
  • Falls back to empirical excess_A thresholds from memory:
      C threshold: excess_A ≥ 0.15
      M threshold: excess_A ≥ 0.99
      X threshold: excess_A ≥ 3.00
    These come from the Module 3 GOES cross-match calibration and replace
    the wrong first-principles values (C=5.0, M=15.0, X=50.0) that were
    15-100x too high.
  • The fallback still produces *correct* X-class labels for saturated
    cadences (excess_A is masked on saturation, so those cadences get -1,
    which is then replaced by the label from adjacent unsaturated cadences
    in build_windows() — the GOES cross-match is the only source of truth
    for truly saturated events like the 2024-05-06 X4.5).

What Module 4 needs from this module
--------------------------------------
  • pds["goes_class"] — per-cadence int8 label added to the Dataset
  • Empirical thresholds stored in pds.attrs["goes_thresholds"] — used by
    Module 4 output layer calibration (temperature scaling)
  • pds.attrs["goes_label_source"] — "ngdc_netcdf"/"cached"/"excess_A_fallback"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Empirical thresholds (calibrated against GOES on real Aditya-L1 data)
# Source: Module 3 cross-match across 5 active days
# Replace first-principles values — those were 15-100x wrong.
# ---------------------------------------------------------------------------
EMPIRICAL_THRESHOLDS_EXCESS_A = {
    "B_p50": 0.07,
    "C_p50": 0.15,
    "M_p50": 0.99,
    "X_p50": 3.00,
}

# GOES class integer mapping (consistent with goes.py)
GOES_CLASS_INT = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_goes_labels(
    preprocess_ds,
    date: Optional[Union[str, datetime]] = None,
    prefix: str = "solexs_sdd2",
    empirical_thresholds: Optional[dict] = None,
    cache_dir: Optional[str] = "/tmp/goes_cache",
    timeout_s: int = 25,
) -> object:  # returns xr.Dataset
    """
    Add per-cadence ``goes_class`` variable to *preprocess_ds*.

    Parameters
    ----------
    preprocess_ds : xr.Dataset from module2.preprocess_day()
    date          : YYYYMMDD string, YYYY-MM-DD string, or datetime.
                    Pass None to use excess_A fallback only.
    prefix        : detector variable prefix (default "solexs_sdd2")
    empirical_thresholds : dict from a previous CrossMatchResult.thresholds.
                    If None, uses the calibrated defaults from Module 3.
    cache_dir     : directory for caching downloaded GOES files
    timeout_s     : HTTP timeout for GOES download

    Returns
    -------
    xr.Dataset  — same as preprocess_ds but with added variables:
        goes_class   : (time,) int8  — 0=A, 1=B, 2=C, 3=M, 4=X, -1=unknown
    And attrs:
        goes_label_source   : str
        goes_thresholds     : dict (empirical excess_A thresholds for Module 4)
    """
    import xarray as xr

    n = len(preprocess_ds.coords["time"])
    thr = empirical_thresholds or EMPIRICAL_THRESHOLDS_EXCESS_A.copy()

    goes_class_1s = None
    source = "excess_A_fallback"

    # ── Attempt real GOES download ────────────────────────────────────────
    if date is not None:
        date_str = _to_date_str(date)
        try:
            from pipeline.module3.goes import fetch_and_crossmatch
            xm = fetch_and_crossmatch(
                preprocess_ds,
                date=date_str,
                prefix=prefix,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
            )
            if xm.n_crossmatched > 0:
                goes_class_1s = xm.goes_class_1s
                source = xm.source
                # Merge downloaded thresholds into our calibrated set
                thr = {**thr, **xm.thresholds}
                logger.info(
                    "GOES labels: %d crossmatched cadences, source=%s",
                    xm.n_crossmatched, source,
                )
            else:
                logger.warning(
                    "GOES crossmatch returned 0 cadences for %s — "
                    "falling back to excess_A thresholds",
                    date_str,
                )
        except Exception as exc:
            logger.warning(
                "GOES download failed for %s (%s) — falling back to excess_A",
                date_str, exc,
            )

    # ── Fallback: derive labels from excess_A + empirical thresholds ─────
    if goes_class_1s is None:
        goes_class_1s = _labels_from_excess_A(preprocess_ds, prefix, thr)
        source = "excess_A_fallback"
        logger.info(
            "goes_class derived from excess_A (fallback).  "
            "Thresholds: C≥%.2f M≥%.2f X≥%.2f",
            thr["C_p50"], thr["M_p50"], thr["X_p50"],
        )

    # ── Summary ────────────────────────────────────────────────────────────
    unique, counts = np.unique(goes_class_1s[goes_class_1s >= 0], return_counts=True)
    cls_map = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}
    dist = {cls_map.get(int(u), "?"): int(c) for u, c in zip(unique, counts)}
    n_unknown = int((goes_class_1s < 0).sum())
    logger.info("goes_class distribution: %s  unknown=%d", dist, n_unknown)

    # ── Attach to Dataset ─────────────────────────────────────────────────
    ds_out = preprocess_ds.assign({
        "goes_class": xr.DataArray(
            goes_class_1s.astype(np.int8), dims=["time"]
        )
    })
    ds_out.attrs["goes_label_source"]  = source
    ds_out.attrs["goes_thresholds"]    = thr
    ds_out.attrs["goes_class_dist"]    = dist

    return ds_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date_str(date: Union[str, datetime]) -> str:
    """Normalise date to YYYYMMDD string."""
    if isinstance(date, datetime):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "")


def _labels_from_excess_A(
    preprocess_ds,
    prefix: str,
    thresholds: dict,
) -> np.ndarray:
    """
    Derive per-cadence GOES class integers from excess_A using empirical thresholds.

    Thresholds applied (p50 values from Module 3 calibration):
        excess_A ≥ X_p50 (3.00) → X (4)
        excess_A ≥ M_p50 (0.99) → M (3)
        excess_A ≥ C_p50 (0.15) → C (2)
        excess_A ≥ B_p50 (0.07) → B (1)
        else                    → A (0)
        NaN (saturated/GTI bad) → -1 (unknown)

    NOTE: This produces CORRECT X-class labels for unsaturated high-flux
    cadences.  Saturated cadences (where excess_A is NaN due to quality masking)
    remain -1; they are handled in build_windows() by the horizon-max logic:
    the window's label comes from the horizon maximum, which will pick up
    the unsaturated flanks of the flare.
    """
    from pipeline.module1.quality import is_usable, QFlag

    n = len(preprocess_ds.coords["time"])
    goes_class = np.full(n, -1, dtype=np.int8)

    excess_key = f"{prefix}_excess_A"
    if excess_key not in preprocess_ds.data_vars:
        logger.warning(
            "%s not found in Dataset — all cadences labelled unknown (-1)", excess_key
        )
        return goes_class

    excess_A = preprocess_ds[excess_key].values.astype(float)

    # Quality mask: only usable, non-saturated cadences get labels
    q_key = f"{prefix}_quality"
    if q_key in preprocess_ds.data_vars:
        quality = preprocess_ds[q_key].values.astype(np.uint8)
        usable  = is_usable(quality)
        not_sat = (quality & QFlag.SATURATED) == 0
        valid   = usable & not_sat & ~np.isnan(excess_A)
    else:
        valid = ~np.isnan(excess_A)

    x_thr = float(thresholds.get("X_p50", 3.00))
    m_thr = float(thresholds.get("M_p50", 0.99))
    c_thr = float(thresholds.get("C_p50", 0.15))
    b_thr = float(thresholds.get("B_p50", 0.07))

    # Walk from high to low to avoid multiple assignments
    goes_class[valid & (excess_A >= x_thr)] = 4   # X
    goes_class[valid & (excess_A >= m_thr) & (goes_class < 0)] = 3   # M
    goes_class[valid & (excess_A >= c_thr) & (goes_class < 0)] = 2   # C
    goes_class[valid & (excess_A >= b_thr) & (goes_class < 0)] = 1   # B
    goes_class[valid & (goes_class < 0)]                        = 0   # A

    return goes_class
