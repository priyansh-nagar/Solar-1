"""
goes.py — GOES XRS download, alignment, and empirical threshold calibration
============================================================================

Three responsibilities:
  1. Download GOES-16 XRS 1-minute averages from NOAA NGDC for a given date.
  2. Align the 1-min GOES flux to the SoLEXS 1-s time grid by forward-fill
     within each 60-s bin (no extrapolation).
  3. Compute the empirical distribution of SoLEXS excess_A per GOES class
     so that detection thresholds come from data, not assumptions.

GOES classification (1-8 Å / 0.1-0.8 nm channel, W/m²)
---------------------------------------------------------
  A < 1e-7   B [1e-7, 1e-6)   C [1e-6, 1e-5)   M [1e-5, 1e-4)   X ≥ 1e-4

NOAA NGDC NetCDF URL pattern (GOES-16)
---------------------------------------
  https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/
  goes/goes16/l2/data/xrsf-l2-avg1m_science/{YYYY}/{MM}/
  sci_xrsf-l2-avg1m_g16_d{YYYYMMDD}_v2-2-0.nc

Important caveats
-----------------
• SoLEXS saturates on very bright flares; those cadences are quality-flagged.
  The cross-match assigns GOES labels to ALL cadences (including saturated ones),
  so Module 4 can train on GOES labels even when SoLEXS excess_A is masked.

• GOES has its own data gaps.  NaN-fill is used where GOES data is missing.

• Empirical thresholds are stored per-class and should REPLACE the first-
  principles thresholds in windows.py before Module 4 training.
"""

from __future__ import annotations

import io
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GOES flux class boundaries (W/m², 1-8 Å channel)
# ---------------------------------------------------------------------------
GOES_BOUNDARIES = {          # class → (lo, hi)  in W/m²
    "A": (0.0,   1e-7),
    "B": (1e-7,  1e-6),
    "C": (1e-6,  1e-5),
    "M": (1e-5,  1e-4),
    "X": (1e-4,  np.inf),
}
GOES_CLASS_ORDER = ["A", "B", "C", "M", "X"]
GOES_CLASS_INT   = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4, "?": -1}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CrossMatchResult:
    """
    Output of fetch_and_crossmatch().

    Attributes
    ----------
    goes_flux_1s   : (n_times,) float64  — GOES B-channel flux (W/m²) on
                     SoLEXS 1-s grid, NaN where GOES data is missing.
    goes_class_1s  : (n_times,) int8     — GOES class integer (0=A,1=B,…,4=X,
                     -1=unknown) per SoLEXS cadence.
    thresholds     : dict — empirical excess_A thresholds per GOES class.
                     Keys: "B_p50", "B_p95", "C_p50", "C_p95", …, "X_p50", "X_p95"
                     Also: "detection_p50", "detection_p95" (C+ floor).
    goes_df        : optional raw 1-min DataFrame (time, xrsb_flux, goes_class)
    n_crossmatched : int — number of SoLEXS cadences with valid GOES data
    date           : str — YYYYMMDD
    satellite      : str — e.g. "goes16"
    source         : str — "ngdc_netcdf" | "cached" | "synthetic" | "unavailable"
    """
    goes_flux_1s:    np.ndarray
    goes_class_1s:   np.ndarray
    thresholds:      dict
    goes_df:         Optional[object] = None   # pd.DataFrame or None
    n_crossmatched:  int = 0
    date:            str = ""
    satellite:       str = "goes16"
    source:          str = "unavailable"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_and_crossmatch(
    preprocess_ds,
    date: str,                           # YYYYMMDD or YYYY-MM-DD
    prefix: str = "solexs_sdd2",
    satellite: str = "goes16",
    cache_dir: Optional[str | Path] = "/tmp/goes_cache",
    timeout_s: int = 25,
) -> CrossMatchResult:
    """
    Download GOES XRS data for *date*, align to SoLEXS grid, derive thresholds.

    Parameters
    ----------
    preprocess_ds : xr.Dataset from module2.preprocess_day()
    date          : "YYYYMMDD" or "YYYY-MM-DD"
    prefix        : detector variable prefix (e.g. "solexs_sdd2")
    satellite     : GOES satellite identifier (default "goes16")
    cache_dir     : directory for caching downloaded NetCDF files (None=no cache)
    timeout_s     : HTTP request timeout in seconds

    Returns
    -------
    CrossMatchResult
    """
    date_clean = date.replace("-", "")
    if len(date_clean) != 8:
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got: {date!r}")

    # ── Extract SoLEXS time grid and excess_A ────────────────────────────
    import xarray as xr
    times_utc = preprocess_ds.coords["time"].values          # datetime64[ns]
    n_times   = len(times_utc)

    excess_key = f"{prefix}_excess_A"
    excess_A   = (
        preprocess_ds[excess_key].values.astype(float)
        if excess_key in preprocess_ds.data_vars
        else np.full(n_times, np.nan)
    )
    quality_key = f"{prefix}_quality"
    quality = (
        preprocess_ds[quality_key].values.astype(np.uint8)
        if quality_key in preprocess_ds.data_vars
        else np.zeros(n_times, dtype=np.uint8)
    )

    # ── Try to download / load GOES data ─────────────────────────────────
    goes_df = _load_goes(date_clean, satellite, cache_dir, timeout_s)

    if goes_df is None:
        logger.warning(
            "GOES data unavailable for %s — cross-match skipped; "
            "using SoLEXS-only labels.",
            date_clean,
        )
        return CrossMatchResult(
            goes_flux_1s   = np.full(n_times, np.nan),
            goes_class_1s  = np.full(n_times, -1, dtype=np.int8),
            thresholds     = _fallback_thresholds(),
            goes_df        = None,
            n_crossmatched = 0,
            date           = date_clean,
            satellite      = satellite,
            source         = "unavailable",
        )

    # ── Align GOES 1-min → SoLEXS 1-s grid ──────────────────────────────
    goes_flux_1s, goes_class_1s = _align_goes_to_solexs(
        goes_df, times_utc
    )
    n_xm = int(np.sum(~np.isnan(goes_flux_1s)))

    logger.info(
        "GOES cross-match: %d / %d SoLEXS cadences have valid GOES data",
        n_xm, n_times,
    )

    # ── Empirical excess_A thresholds per GOES class ──────────────────────
    thresholds = _compute_empirical_thresholds(
        excess_A, quality, goes_class_1s
    )

    logger.info("Empirical thresholds: %s", thresholds)

    return CrossMatchResult(
        goes_flux_1s   = goes_flux_1s,
        goes_class_1s  = goes_class_1s,
        thresholds     = thresholds,
        goes_df        = goes_df,
        n_crossmatched = n_xm,
        date           = date_clean,
        satellite      = satellite,
        source         = goes_df.attrs.get("source", "ngdc_netcdf"),
    )


# ---------------------------------------------------------------------------
# GOES flux → class
# ---------------------------------------------------------------------------

def goes_class_from_flux(flux_wm2: float) -> str:
    """Convert GOES B-channel flux (W/m²) to class letter."""
    if np.isnan(flux_wm2) or flux_wm2 < 0:
        return "?"
    for cls in reversed(GOES_CLASS_ORDER):
        if flux_wm2 >= GOES_BOUNDARIES[cls][0]:
            return cls
    return "A"


def goes_class_number(flux_wm2: float) -> float:
    """
    Convert GOES flux to the numeric value in the class scale.
    e.g. 5.2e-5 W/m² → M5.2 → returns 5.2 (with letter 'M').
    """
    cls = goes_class_from_flux(flux_wm2)
    if cls == "?":
        return 0.0
    lo = GOES_BOUNDARIES[cls][0]
    if lo == 0.0:
        return flux_wm2 / 1e-8   # A-class: scale to 0-10 range
    return flux_wm2 / lo


# ---------------------------------------------------------------------------
# Empirical thresholds
# ---------------------------------------------------------------------------

def _compute_empirical_thresholds(
    excess_A: np.ndarray,
    quality: np.ndarray,
    goes_class_1s: np.ndarray,
) -> dict:
    """
    Compute median and 95th percentile of excess_A per GOES class.

    Only uses cadences that are:
      • inside GTI (quality bit 0 set)
      • not saturated (quality bit 2 clear)
      • have valid GOES data (goes_class_1s != -1)
    """
    from pipeline.module1.quality import is_usable
    usable = is_usable(quality)

    thresholds: dict = {}

    for cls, idx in GOES_CLASS_INT.items():
        if cls == "?" or cls == "A":
            continue
        mask = usable & (goes_class_1s == idx) & ~np.isnan(excess_A)
        vals = excess_A[mask]
        if len(vals) < 5:
            thresholds[f"{cls}_n"]   = 0
            thresholds[f"{cls}_p50"] = np.nan
            thresholds[f"{cls}_p95"] = np.nan
        else:
            thresholds[f"{cls}_n"]   = int(len(vals))
            thresholds[f"{cls}_p50"] = float(np.nanpercentile(vals, 50))
            thresholds[f"{cls}_p95"] = float(np.nanpercentile(vals, 95))

    # Detection threshold: p50 of C-class (or fallback to 3.0 if no C data)
    c_p50 = thresholds.get("C_p50", np.nan)
    thresholds["detection_p50"] = float(c_p50) if not np.isnan(c_p50) else 3.0
    c_p95 = thresholds.get("C_p95", np.nan)
    thresholds["detection_p95"] = float(c_p95) if not np.isnan(c_p95) else 5.0

    return thresholds


def _fallback_thresholds() -> dict:
    """First-principles thresholds used when GOES data is unavailable."""
    return {
        "B_p50": 0.5, "B_p95": 1.0,
        "C_p50": 3.0, "C_p95": 5.0,
        "M_p50": 8.0, "M_p95": 15.0,
        "X_p50": 25.0, "X_p95": 50.0,
        "detection_p50": 3.0,
        "detection_p95": 5.0,
        "source": "first_principles",
    }


# ---------------------------------------------------------------------------
# GOES 1-min → SoLEXS 1-s alignment
# ---------------------------------------------------------------------------

def _align_goes_to_solexs(
    goes_df,
    solexs_times_utc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward-fill GOES 1-min flux onto the SoLEXS 1-s time grid.

    Each 60-s GOES bin is held constant for 60 cadences of SoLEXS data.
    No extrapolation: SoLEXS cadences before the first GOES timestamp or
    more than 90 s after the last are left as NaN.

    Returns
    -------
    goes_flux_1s  : (n_solexs,) float64
    goes_class_1s : (n_solexs,) int8
    """
    import pandas as pd

    n = len(solexs_times_utc)
    goes_flux_1s  = np.full(n, np.nan, dtype=np.float64)
    goes_class_1s = np.full(n, -1, dtype=np.int8)

    # Convert SoLEXS times to Unix seconds for comparison
    solexs_unix = (
        solexs_times_utc.astype("datetime64[ns]").astype(np.int64) / 1e9
    )

    # GOES DataFrame must have "time_unix" and "xrsb_flux" columns
    if "time_unix" not in goes_df.columns or "xrsb_flux" not in goes_df.columns:
        logger.error("GOES DataFrame missing required columns")
        return goes_flux_1s, goes_class_1s

    goes_t = goes_df["time_unix"].values.astype(np.float64)
    goes_f = goes_df["xrsb_flux"].values.astype(np.float64)
    goes_c = goes_df["goes_class_int"].values.astype(np.int8)

    # For each GOES 1-min record, fill all SoLEXS cadences in [t_goes, t_goes+90)
    for i in range(len(goes_t)):
        t0 = goes_t[i]
        t1 = t0 + 90.0      # generous half-window: 1.5× the 60-s cadence
        mask = (solexs_unix >= t0) & (solexs_unix < t1)
        if not mask.any():
            continue
        flux = goes_f[i]
        cls  = goes_c[i]
        # Only fill where not yet set (forward-fill wins over backward edge)
        unfilled = mask & np.isnan(goes_flux_1s)
        goes_flux_1s[unfilled]  = flux
        goes_class_1s[unfilled] = cls

    return goes_flux_1s, goes_class_1s


# ---------------------------------------------------------------------------
# NOAA NGDC download
# ---------------------------------------------------------------------------

def _load_goes(
    date_clean: str,
    satellite: str,
    cache_dir: Optional[str | Path],
    timeout_s: int,
) -> Optional[object]:
    """
    Load GOES XRS 1-min data for *date_clean* (YYYYMMDD).

    Tries in order:
      1. Disk cache (if cache_dir is set)
      2. NOAA NGDC NetCDF download
      3. Returns None on failure
    """
    import pandas as pd

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"goes_{satellite}_{date_clean}.csv"
        if cache_path.exists():
            logger.info("Loading GOES data from cache: %s", cache_path)
            df = pd.read_csv(cache_path)
            df["goes_class_int"] = df["goes_class_int"].astype(np.int8)
            df.attrs["source"] = "cached"
            return df

    df = _download_ngdc(date_clean, satellite, timeout_s)
    if df is not None and cache_path is not None:
        try:
            df.to_csv(cache_path, index=False)
            logger.info("Cached GOES data to %s", cache_path)
        except Exception as e:
            logger.warning("Could not cache GOES data: %s", e)

    return df


def _download_ngdc(
    date_clean: str,
    satellite: str,
    timeout_s: int,
) -> Optional[object]:
    """
    Download GOES XRS 1-min average NetCDF from NOAA NGDC.

    Returns a DataFrame with columns: time_unix, xrsa_flux, xrsb_flux,
    goes_class, goes_class_int. Returns None on failure.
    """
    import requests
    import pandas as pd

    yyyy = date_clean[:4]
    mm   = date_clean[4:6]
    dd   = date_clean[6:8]

    # Directory listing to find the exact filename (version may vary)
    base_url = (
        f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
        f"/goes/{satellite}/l2/data/xrsf-l2-avg1m_science/{yyyy}/{mm}/"
    )

    try:
        resp = requests.get(base_url, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("NOAA NGDC directory listing failed: %s", e)
        return None

    # Find matching filename: sci_xrsf-l2-avg1m_g16_dYYYYMMDD_v*.nc
    pattern = re.compile(
        rf"sci_xrsf-l2-avg1m_{satellite.replace('goes','g')}[0-9]*"
        rf"_d{date_clean}_v[\d\-]+\.nc",
        re.IGNORECASE,
    )
    filenames = pattern.findall(resp.text)
    if not filenames:
        logger.warning(
            "No GOES NetCDF found for %s in %s", date_clean, base_url
        )
        return None

    # Prefer the highest version number
    filename = sorted(set(filenames))[-1]
    nc_url   = base_url + filename
    logger.info("Downloading GOES: %s", nc_url)

    try:
        nc_resp = requests.get(nc_url, timeout=timeout_s)
        nc_resp.raise_for_status()
    except Exception as e:
        logger.warning("GOES NetCDF download failed: %s", e)
        return None

    return _parse_goes_netcdf(nc_resp.content, date_clean)


def _parse_goes_netcdf(raw_bytes: bytes, date_clean: str) -> Optional[object]:
    """Parse NOAA NGDC GOES XRS NetCDF bytes into a DataFrame."""
    import pandas as pd

    try:
        import xarray as xr
        ds = xr.open_dataset(
            io.BytesIO(raw_bytes),
            engine="h5netcdf",
            mask_and_scale=True,
        )
    except Exception as e:
        logger.warning("Failed to open GOES NetCDF: %s", e)
        return None

    try:
        # Time: decode from the dataset
        times = ds["time"].values  # datetime64[ns]
        t_unix = times.astype("datetime64[ns]").astype(np.int64) / 1e9

        # Flux channels
        xrsa = ds.get("xrsa_flux", ds.get("a_flux", None))
        xrsb = ds.get("xrsb_flux", ds.get("b_flux", None))

        if xrsb is None:
            logger.warning("GOES NetCDF has no xrsb_flux channel")
            ds.close()
            return None

        xrsa_arr = np.asarray(xrsa, dtype=float) if xrsa is not None else np.full(len(t_unix), np.nan)
        xrsb_arr = np.asarray(xrsb, dtype=float)

        # Replace fill values (<0) with NaN
        xrsa_arr[xrsa_arr < 0] = np.nan
        xrsb_arr[xrsb_arr < 0] = np.nan

        ds.close()

    except Exception as e:
        logger.warning("Failed to parse GOES NetCDF variables: %s", e)
        try:
            ds.close()
        except Exception:
            pass
        return None

    import pandas as pd

    df = pd.DataFrame({
        "time_unix":      t_unix,
        "xrsa_flux":      xrsa_arr,
        "xrsb_flux":      xrsb_arr,
        "goes_class":     [goes_class_from_flux(f) for f in xrsb_arr],
        "goes_class_int": np.array(
            [GOES_CLASS_INT.get(goes_class_from_flux(f), -1) for f in xrsb_arr],
            dtype=np.int8,
        ),
    })

    df.attrs["source"]     = "ngdc_netcdf"
    df.attrs["satellite"]  = "goes16"
    df.attrs["date"]       = date_clean

    logger.info(
        "GOES data loaded: %d 1-min records for %s  "
        "xrsb range: %.2e – %.2e W/m²",
        len(df), date_clean,
        np.nanmin(xrsb_arr), np.nanmax(xrsb_arr),
    )

    return df
