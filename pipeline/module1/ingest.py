"""
ingest.py — Main data ingestion pipeline (Module 1 entry point)
================================================================
Reads Level-1 SoLEXS and/or HEL1OS FITS products from ISSDC/PRADAN,
applies quality filtering, aligns instruments to a common time grid,
and returns a research-grade xarray Dataset.

Usage
-----
    from pipeline.module1 import ingest_day

    # SoLEXS only (HEL1OS not available for this day)
    ds = ingest_day(
        solexs_dir="data/AL1_SLX_L1_20260612_v1.0/",
        hel1os_dir=None,
    )

    # Access per-second total light curve for SDD2
    ts = ds["solexs_sdd2_total"]          # xr.DataArray (time,)

    # Access per-second full spectrum
    spec = ds["solexs_sdd2_spectrum"]      # xr.DataArray (time, channel)

    # Energy band light curves
    band_a = ds["solexs_sdd2_band_A"]      # xr.DataArray (time,)

    # Quality flags
    qf = ds["solexs_sdd2_quality"]         # uint8 DataArray (time,)

Returned Dataset coordinates
-----------------------------
  time     : datetime64[ns] UTC, 1-second cadence
  channel  : int channel number (0-indexed), for spectrum variables
  energy_keV: float64, centre energy in keV for each channel (SoLEXS)

Dataset attributes include pipeline provenance (git hash not yet wired,
but all header metadata is preserved).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import astropy.io.fits as fits
import astropy.time

from .formats import (
    Instrument,
    ProductKind,
    FITSProduct,
    discover_products,
    _open_fits,
)
from .channels import (
    SOLEXS_CALIBRATION,
    HEL1OS_CALIBRATION,
    DetectorCalibration,
    energy_axis,
    extract_all_bands,
    total_counts,
    flag_saturated_rows,
)
from .quality import (
    parse_gti,
    build_quality_flags,
    quality_summary,
    is_usable,
)
from .gaps import find_gaps, gap_summary, interpolate_all_bands
from .align import build_master_grid, align_detectors, reindex_to_grid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_day(
    solexs_dir: Optional[str | Path] = None,
    hel1os_dir: Optional[str | Path] = None,
    cadence_s: float = 1.0,
    max_gap_fill_s: float = 10.0,
    saturation_cts: float = 1e6,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Ingest one day of Aditya-L1 Level-1 data and return an xarray Dataset.

    Parameters
    ----------
    solexs_dir     : path to the SoLEXS Level-1 day directory
                     (e.g. "AL1_SLX_L1_20260612_v1.0/")
    hel1os_dir     : path to the HEL1OS Level-1 day directory (or None)
    cadence_s      : expected time resolution in seconds (default 1.0)
    max_gap_fill_s : maximum gap length to fill by linear interpolation
                     in band light curves.  Longer gaps stay NaN.
    saturation_cts : per-channel count threshold for saturation flagging.
                     Tune with empirical detector characterisation data.
    verbose        : if True, log quality summaries to the root logger

    Returns
    -------
    xr.Dataset with variables:
      solexs_{det}_spectrum  (time, channel)  — per-second spectra
      solexs_{det}_total     (time,)           — total flux (sum all ch)
      solexs_{det}_band_{X}  (time,)           — science band X (A/B/C/D)
      solexs_{det}_quality   (time,)           — uint8 quality flags
      hel1os_{det}_*         (same structure, if hel1os_dir given)

    Dataset attributes preserve instrument metadata and pipeline parameters.
    """
    datasets = []
    global_t_start = None
    global_t_stop  = None

    # ------------------------------------------------------------------ #
    # 1. Ingest each instrument
    # ------------------------------------------------------------------ #
    if solexs_dir is not None:
        solexs_parts, s_t0, s_t1 = _ingest_instrument(
            Path(solexs_dir),
            expected_instrument=Instrument.SOLEXS,
            calibration=SOLEXS_CALIBRATION,
            cadence_s=cadence_s,
            max_gap_fill_s=max_gap_fill_s,
            saturation_cts=saturation_cts,
            verbose=verbose,
            prefix="solexs",
        )
        datasets.extend(solexs_parts)
        global_t_start = _min_notnone(global_t_start, s_t0)
        global_t_stop  = _max_notnone(global_t_stop, s_t1)

    if hel1os_dir is not None:
        hel1os_parts, h_t0, h_t1 = _ingest_instrument(
            Path(hel1os_dir),
            expected_instrument=Instrument.HEL1OS,
            calibration=HEL1OS_CALIBRATION,
            cadence_s=cadence_s,
            max_gap_fill_s=max_gap_fill_s,
            saturation_cts=saturation_cts,
            verbose=verbose,
            prefix="hel1os",
        )
        datasets.extend(hel1os_parts)
        global_t_start = _min_notnone(global_t_start, h_t0)
        global_t_stop  = _max_notnone(global_t_stop, h_t1)

    if not datasets:
        raise ValueError(
            "No usable data found. Check that solexs_dir or hel1os_dir "
            "point to a directory containing .pi.gz, .lc.gz, and .gti.gz files."
        )

    # ------------------------------------------------------------------ #
    # 2. Merge all detector Datasets onto a common time axis
    # ------------------------------------------------------------------ #
    # Build master UTC time grid
    master_grid = build_master_grid(global_t_start, global_t_stop, cadence_s)
    master_time_utc = _unix_to_datetime64(master_grid)

    merged_vars = {}
    for ds_partial in datasets:
        # Re-index each detector's variables to the master grid
        det_times_unix = _datetime64_to_unix(ds_partial.coords["time"].values)
        align_result = align_detectors(
            {"det": det_times_unix}, master_grid, tolerance_s=cadence_s * 0.5
        )["det"]

        g_idx   = align_result["grid_indices"]
        v_mask  = align_result["valid_mask"]
        n_grid  = len(master_grid)

        if verbose:
            logger.info(
                "Alignment: coverage=%.1f%%, max_residual=%.2f ms",
                align_result["coverage_frac"] * 100,
                align_result["max_residual_ms"],
            )

        new_vars = {}
        for var_name, da in ds_partial.data_vars.items():
            arr = da.values
            # Quality flag arrays are uint8 — must not be cast to float
            # (NaN cannot exist in integer arrays; fill with 0 instead)
            is_flags = np.issubdtype(arr.dtype, np.integer)
            fill = 0 if is_flags else np.nan

            if arr.ndim == 1:
                reindexed = reindex_to_grid(arr, g_idx, v_mask, n_grid, fill_value=fill)
                if is_flags:
                    reindexed = reindexed.astype(arr.dtype)
                new_vars[var_name] = xr.DataArray(reindexed, dims=["time"])
            elif arr.ndim == 2:
                reindexed = reindex_to_grid(arr, g_idx, v_mask, n_grid, fill_value=fill)
                if is_flags:
                    reindexed = reindexed.astype(arr.dtype)
                new_vars[var_name] = xr.DataArray(
                    reindexed, dims=["time", da.dims[1]]
                )

        # Copy per-channel coordinate if present
        partial_ds = xr.Dataset(new_vars, coords={"time": master_time_utc})
        for cname, cval in ds_partial.coords.items():
            if cname != "time":
                partial_ds = partial_ds.assign_coords({cname: cval})
        partial_ds.attrs.update(ds_partial.attrs)
        merged_vars[id(partial_ds)] = partial_ds

    if len(merged_vars) == 1:
        result = list(merged_vars.values())[0]
    else:
        result = xr.merge(list(merged_vars.values()), join="outer")

    result.attrs["cadence_s"]       = cadence_s
    result.attrs["max_gap_fill_s"]  = max_gap_fill_s
    result.attrs["pipeline_module"] = "module1 v1.0"

    return result


# ---------------------------------------------------------------------------
# Per-instrument ingestion
# ---------------------------------------------------------------------------

def _ingest_instrument(
    directory: Path,
    expected_instrument: Instrument,
    calibration: DetectorCalibration,
    cadence_s: float,
    max_gap_fill_s: float,
    saturation_cts: float,
    verbose: bool,
    prefix: str,
):
    """
    Ingest all detectors for one instrument from *directory*.

    Returns (list_of_partial_datasets, global_t_start, global_t_stop).
    Each partial dataset covers one detector (SDD1, SDD2, …).
    """
    products = discover_products(directory)
    if not products:
        raise FileNotFoundError(f"No FITS files found in {directory}")

    # Warn about unexpected instruments
    for p in products:
        if p.instrument != expected_instrument and p.instrument != Instrument.UNKNOWN:
            warnings.warn(
                f"Found {p.instrument.value} product in {directory} "
                f"(expected {expected_instrument.value}): {p.path.name}"
            )

    # Group by detector label
    detectors: dict[str, dict[str, FITSProduct]] = {}
    for p in products:
        det = p.detector
        if det not in detectors:
            detectors[det] = {}
        detectors[det][p.kind] = p

    partial_datasets = []
    all_t_starts = []
    all_t_stops  = []

    for det_label, kind_map in detectors.items():
        logger.info("Ingesting %s / %s", expected_instrument.value, det_label)

        pi_product  = kind_map.get(ProductKind.SPECTRUM)
        lc_product  = kind_map.get(ProductKind.LIGHTCURVE)
        gti_product = kind_map.get(ProductKind.GTI)

        # Prefer .pi (full spectrum) over .lc (total flux only)
        if pi_product is not None:
            ds_det, t0, t1 = _ingest_spectrum_product(
                pi_product, gti_product,
                calibration, det_label, prefix,
                cadence_s, max_gap_fill_s, saturation_cts, verbose,
            )
        elif lc_product is not None:
            logger.warning(
                "%s/%s: no .pi file, falling back to .lc (no per-channel data)",
                expected_instrument.value, det_label
            )
            ds_det, t0, t1 = _ingest_lc_product(
                lc_product, gti_product,
                det_label, prefix,
                cadence_s, max_gap_fill_s, verbose,
            )
        else:
            logger.warning(
                "%s/%s: no spectrum or light-curve file found, skipping",
                expected_instrument.value, det_label
            )
            continue

        partial_datasets.append(ds_det)
        all_t_starts.append(t0)
        all_t_stops.append(t1)

    if not all_t_starts:
        raise ValueError(f"No usable detector data in {directory}")

    return partial_datasets, min(all_t_starts), max(all_t_stops)


# ---------------------------------------------------------------------------
# Spectrum (.pi) product reader
# ---------------------------------------------------------------------------

def _ingest_spectrum_product(
    pi_product: FITSProduct,
    gti_product: Optional[FITSProduct],
    calibration: DetectorCalibration,
    det_label: str,
    prefix: str,
    cadence_s: float,
    max_gap_fill_s: float,
    saturation_cts: float,
    verbose: bool,
) -> tuple[xr.Dataset, float, float]:
    """
    Read one .pi FITS file and build a partial xarray Dataset.

    Actual SDD2 .pi structure (verified from ISSDC file):
      HDU 1 SPECTRUM:
        TSTART   (D)       — row start time, seconds since MJD 40587 (Unix epoch)
        TELAPSE  (D)       — exposure duration, seconds
        SPEC_NUM (J)       — sequential spectrum number
        CHANNEL  (340K)    — channel array [0..339]  (same every row)
        COUNTS   (340D)    — count in each channel; NaN = bad/missing cadence
        EXPOSURE (D)       — actual live time for this row

    Column COUNTS NaN rows = detector gap / bad readout.
    Saturation is not directly flagged in the header; we detect it via
    count-rate thresholds (see channels.flag_saturated_rows).
    """
    hdul = pi_product.hdul()
    try:
        spec_hdu = hdul["SPECTRUM"]
    except KeyError:
        spec_hdu = hdul[1]   # fallback: first extension

    data = spec_hdu.data
    header = spec_hdu.header

    # ------------------------------------------------------------------ #
    # Time array
    # ------------------------------------------------------------------ #
    tstart   = data["TSTART"].astype(np.float64)      # Unix seconds
    exposure = data["EXPOSURE"].astype(np.float64)
    n_times  = len(tstart)

    t0 = float(tstart[0])
    t1 = float(tstart[-1])

    # ------------------------------------------------------------------ #
    # Spectrum array
    # ------------------------------------------------------------------ #
    counts = data["COUNTS"].astype(np.float64)        # (n_times, n_channels)
    n_channels = counts.shape[1]

    if n_channels != calibration.n_channels:
        warnings.warn(
            f"{det_label}: spectrum has {n_channels} channels but calibration "
            f"expects {calibration.n_channels}.  Using data channel count."
        )
        # Do not crash; the calibration energy axis will be recomputed below

    # ------------------------------------------------------------------ #
    # GTI parsing
    # ------------------------------------------------------------------ #
    gti_intervals = []
    if gti_product is not None:
        gti_hdul = gti_product.hdul()
        gti_intervals = parse_gti(gti_hdul)
        if verbose:
            logger.info("%s GTI intervals: %d (total %.0f s)",
                        det_label, len(gti_intervals),
                        sum(b - a for a, b in gti_intervals))

    # ------------------------------------------------------------------ #
    # Saturation detection
    # ------------------------------------------------------------------ #
    sat_mask = flag_saturated_rows(counts, saturation_cts_per_channel=saturation_cts)

    # ------------------------------------------------------------------ #
    # Quality flags
    # ------------------------------------------------------------------ #
    quality = build_quality_flags(
        time_array       = tstart,
        counts_spectrum  = counts,
        gti_intervals    = gti_intervals,
        saturation_mask  = sat_mask,
        exposure         = exposure,
        cadence_s        = cadence_s,
    )

    if verbose:
        qs = quality_summary(quality)
        logger.info("%s quality: %s", det_label, qs)

    usable = is_usable(quality)

    # ------------------------------------------------------------------ #
    # Science band extraction
    # ------------------------------------------------------------------ #
    band_counts = extract_all_bands(counts, calibration)
    total_flux  = total_counts(counts)

    # Gap interpolation on scalar band light curves
    band_counts_filled = interpolate_all_bands(
        band_counts, usable, quality, max_gap_s=max_gap_fill_s, time_array=tstart
    )

    # ------------------------------------------------------------------ #
    # Gap reporting
    # ------------------------------------------------------------------ #
    if verbose:
        gaps = find_gaps(usable, tstart)
        gs = gap_summary(gaps)
        logger.info("%s gaps: %s", det_label, gs)

    # ------------------------------------------------------------------ #
    # Build xarray Dataset
    # ------------------------------------------------------------------ #
    time_utc = _unix_to_datetime64(tstart)

    # Energy axis for this detector
    ch_array = np.arange(n_channels)
    e_keV = calibration.e_min_keV + ch_array * (
        (calibration.e_max_keV - calibration.e_min_keV) / calibration.n_channels
    )

    var_prefix = f"{prefix}_{det_label.lower()}"

    data_vars = {
        f"{var_prefix}_spectrum": xr.DataArray(
            counts,
            dims=["time", "channel"],
            attrs={
                "long_name": f"{det_label} per-second spectrum",
                "units": "counts / s",
                "note": "NaN = bad cadence (NAN_ROW or outside GTI)",
            },
        ),
        f"{var_prefix}_total": xr.DataArray(
            total_flux,
            dims=["time"],
            attrs={
                "long_name": f"{det_label} total flux (sum all channels)",
                "units": "counts / s",
            },
        ),
        f"{var_prefix}_quality": xr.DataArray(
            quality.astype(np.uint8),
            dims=["time"],
            attrs={
                "long_name": "Quality flags",
                "flag_masks":  [0x01, 0x02, 0x04, 0x08, 0x10],
                "flag_meanings": "IN_GTI NAN_ROW SATURATED GAP_FILL LOW_EXPOSURE",
            },
        ),
    }

    for band_name, band_arr in band_counts_filled.items():
        lo, hi = calibration.science_bands[band_name]
        data_vars[f"{var_prefix}_band_{band_name}"] = xr.DataArray(
            band_arr,
            dims=["time"],
            attrs={
                "long_name": f"{det_label} Band {band_name} ({lo:.1f}–{hi:.1f} keV)",
                "units": "counts / s",
                "energy_lo_keV": lo,
                "energy_hi_keV": hi,
                "note": (
                    "Short gaps (≤ max_gap_fill_s) linearly interpolated; "
                    "interpolated cadences have GAP_FILL quality bit set."
                ),
            },
        )

    coords = {
        "time"   : time_utc,
        "channel": ch_array,
        f"{var_prefix}_energy_keV": xr.DataArray(
            e_keV, dims=["channel"],
            attrs={"long_name": "Channel centre energy (approximate linear calibration)",
                   "units": "keV",
                   "note": "Approximate: replace with CALDB RMF when available"}
        ),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Preserve instrument metadata
    ds.attrs.update({
        "instrument"   : pi_product.instrument.value,
        "detector"     : det_label,
        "obs_date"     : str(pi_product.primary_header.get("OBS_DATE", "")),
        "obs_id"       : str(pi_product.primary_header.get("OBS_ID", "")),
        "creator"      : str(pi_product.primary_header.get("CREATOR", "")),
        "source_file"  : pi_product.path.name,
        "n_channels"   : n_channels,
        "e_min_keV"    : calibration.e_min_keV,
        "e_max_keV"    : calibration.e_max_keV,
        "calibration"  : "linear_approximate (no RMF)",
    })

    pi_product.close()
    if gti_product is not None:
        gti_product.close()

    return ds, t0, t1


# ---------------------------------------------------------------------------
# Light-curve (.lc) product reader  (fallback when .pi not available)
# ---------------------------------------------------------------------------

def _ingest_lc_product(
    lc_product: FITSProduct,
    gti_product: Optional[FITSProduct],
    det_label: str,
    prefix: str,
    cadence_s: float,
    max_gap_fill_s: float,
    verbose: bool,
) -> tuple[xr.Dataset, float, float]:
    """
    Fallback reader when only a .lc file is available (e.g. SDD1 on this day).

    Actual SDD2 .lc structure (verified):
      HDU 1 RATE:
        TIME    (D) — seconds since MJD 40587
        COUNTS  (D) — total counts (sum over all channels)

    No per-channel data → no band extraction, no saturation detection.
    """
    hdul = lc_product.hdul()
    try:
        rate_hdu = hdul["RATE"]
    except KeyError:
        rate_hdu = hdul[1]

    data    = rate_hdu.data
    times   = data["TIME"].astype(np.float64)
    counts  = data["COUNTS"].astype(np.float64)

    # NaN detection: use the count value itself
    nan_mask = np.isnan(counts)

    gti_intervals = []
    if gti_product is not None:
        gti_hdul = gti_product.hdul()
        gti_intervals = parse_gti(gti_hdul)

    quality = build_quality_flags(
        time_array      = times,
        counts_spectrum = None,         # no spectrum available
        gti_intervals   = gti_intervals,
        saturation_mask = None,
        cadence_s       = cadence_s,
    )
    # Mark NaN rows manually
    from .quality import QFlag
    quality[nan_mask] |= QFlag.NAN_ROW

    if verbose:
        qs = quality_summary(quality)
        logger.info("%s (lc fallback) quality: %s", det_label, qs)

    usable      = is_usable(quality)
    total_flux  = counts.copy()
    total_flux[nan_mask] = np.nan

    var_prefix = f"{prefix}_{det_label.lower()}"
    time_utc   = _unix_to_datetime64(times)

    data_vars = {
        f"{var_prefix}_total": xr.DataArray(
            total_flux, dims=["time"],
            attrs={"long_name": f"{det_label} total flux", "units": "counts / s"},
        ),
        f"{var_prefix}_quality": xr.DataArray(
            quality.astype(np.uint8), dims=["time"],
            attrs={
                "long_name": "Quality flags",
                "flag_masks":  [0x01, 0x02, 0x04, 0x08, 0x10],
                "flag_meanings": "IN_GTI NAN_ROW SATURATED GAP_FILL LOW_EXPOSURE",
            },
        ),
    }

    ds = xr.Dataset(data_vars, coords={"time": time_utc})
    ds.attrs.update({
        "instrument" : lc_product.instrument.value,
        "detector"   : det_label,
        "source_file": lc_product.path.name,
        "note"       : "LightCurve fallback — no per-channel spectrum available",
    })

    lc_product.close()
    if gti_product is not None:
        gti_product.close()

    return ds, float(times[0]), float(times[-1])


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------

def _unix_to_datetime64(unix_seconds: np.ndarray) -> np.ndarray:
    """
    Convert an array of Unix timestamps (seconds, float64) to numpy datetime64[ns] UTC.

    SoLEXS/HEL1OS FITS files use MJD 40587 as the epoch, which equals
    1970-01-01T00:00:00 UTC (the standard Unix epoch).  No epoch offset needed.

    Note: astropy >= 7 removed Time.to_datetime64(); use integer nanoseconds
    directly for a fast, dependency-light conversion.
    """
    # Round to nearest nanosecond to avoid floating-point drift
    ns = np.round(unix_seconds * 1_000_000_000).astype(np.int64)
    return ns.astype("datetime64[ns]")


def _datetime64_to_unix(dt64: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64[ns] back to float64 Unix seconds."""
    ns_per_s = 1_000_000_000.0
    epoch = np.datetime64(0, "ns")
    return (dt64.astype("datetime64[ns]") - epoch).astype(np.float64) / ns_per_s


def _min_notnone(a, b):
    if a is None: return b
    if b is None: return a
    return min(a, b)


def _max_notnone(a, b):
    if a is None: return b
    if b is None: return a
    return max(a, b)
