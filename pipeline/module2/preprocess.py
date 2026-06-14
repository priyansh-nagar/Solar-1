"""
preprocess.py — Module 2 main entry point
==========================================
Extends the Module 1 xarray Dataset with cleaned, background-subtracted,
and feature-engineered variables ready for the nowcaster and forecaster.

Full pipeline:
  1. Extract band light curves from Module 1 Dataset
  2. Savitzky-Golay smoothing (detrend.py)
  3. SNIP background estimation (background.py)
  4. Feature engineering (features.py)
  5. Temporal split labelling (split.py)
  6. (Optional) Sliding window dataset builder (windows.py)

ingest_day() signature (Module 1)
-----------------------------------
    ds = ingest_day(
        solexs_dir: Optional[str | Path] = None,
        hel1os_dir: Optional[str | Path] = None,
        cadence_s: float = 1.0,
        max_gap_fill_s: float = 10.0,
        saturation_cts: float = 1e6,
        verbose: bool = True,
    ) -> xr.Dataset

Key Dataset variables produced by ingest_day()
-----------------------------------------------
    solexs_sdd2_spectrum     (time, channel)  — per-second spectra, float64
    solexs_sdd2_total        (time,)           — total flux, float64
    solexs_sdd2_band_A       (time,)           — 1.5–3.0 keV, float64
    solexs_sdd2_band_B       (time,)           — 3.0–4.5 keV, float64
    solexs_sdd2_band_C       (time,)           — 4.5–6.0 keV, float64
    solexs_sdd2_band_D       (time,)           — 6.0–6.95 keV, float64
    solexs_sdd2_quality      (time,)           — uint8 quality flags

Dataset coordinates:
    time                  datetime64[ns] UTC, 1-s cadence
    channel               int, 0-indexed channel number (0–339)
    solexs_sdd2_energy_keV (channel,) float64 keV per channel

Dataset attrs (day-level, added in Module 1):
    usable_fraction    float   — fraction of cadences in GTI and not NaN
    gap_seconds        list    — gap durations in seconds (sorted descending)
    quiet_day          bool    — True if peak_flux_ratio < 5×
    sdd1_offline       bool    — True if SDD1 had no operational data
    peak_flux_ratio    float   — max(Band A) / median(Band A)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr

from .detrend    import sg_smooth, DEFAULT_WINDOW_LENGTH, DEFAULT_POLYORDER
from .background import snip_background, subtract_background, DEFAULT_M
from .features   import compute_features
from .split      import (
    compute_split_boundaries, build_split_array, split_summary,
    TRAIN, VAL, TEST,
)
from .windows    import build_windows, WINDOW_S, STRIDE_S, LABEL_HORIZON_S

logger = logging.getLogger(__name__)


def preprocess_day(
    ds: xr.Dataset,
    *,
    sg_window: int     = DEFAULT_WINDOW_LENGTH,
    sg_order: int      = DEFAULT_POLYORDER,
    snip_M: int        = DEFAULT_M,
    train_frac: float  = 0.70,
    val_frac: float    = 0.15,
    test_frac: float   = 0.15,
    split_gap_s: float = 1800.0,
    build_windows: bool = False,
    window_s: int      = WINDOW_S,
    stride_s: int      = STRIDE_S,
    label_horizon_s: int = LABEL_HORIZON_S,
    verbose: bool      = True,
):
    """
    Extend the Module 1 Dataset with preprocessed and feature-engineered variables.

    Parameters
    ----------
    ds              : xr.Dataset from module1.ingest_day()
    sg_window       : Savitzky-Golay window length in samples (must be odd)
    sg_order        : Savitzky-Golay polynomial order
    snip_M          : SNIP maximum window half-width (samples)
    train_frac      : training set fraction (default 0.70)
    val_frac        : validation set fraction (default 0.15)
    test_frac       : test set fraction (default 0.15)
    split_gap_s     : gap between split boundaries to prevent leakage (seconds)
    build_windows   : if True, also build and return a WindowDataset
    window_s        : sliding window length in seconds
    stride_s        : stride between consecutive windows in seconds
    label_horizon_s : label look-ahead horizon in seconds
    verbose         : log progress information

    Returns
    -------
    If build_windows=False (default):
        Extended xr.Dataset with added variables (listed below).
    If build_windows=True:
        (xr.Dataset, WindowDataset) tuple.

    Added Dataset variables
    -----------------------
    For each detector prefix *p* and band *B* found in ds:
      {p}_band_{B}_smooth         — Savitzky-Golay smoothed Band B flux
      {p}_band_{B}_background     — SNIP background estimate
      {p}_band_{B}_residual       — smoothed flux minus background
      {p}_band_{B}_excess         — residual / (background + ε)
      {p}_derivative_1s           — d(Band A smooth)/dt [cts/s/s]
      {p}_derivative_60s          — 60-s smoothed derivative
      {p}_rate_of_rise            — 5-min mean positive derivative
      {p}_hardness_ratio          — Band C / Band A
      {p}_softness_ratio          — Band A / Band B
      {p}_rolling_std_5min        — 5-min rolling σ of Band A
      {p}_rolling_std_15min       — 15-min rolling σ of Band A
      {p}_cumulative_excess        — running integral of positive Band A excess
    split                         — uint8 (time,) TRAIN/VAL/TEST/GAP labels
    """
    cadence_s = float(ds.attrs.get("cadence_s", 1.0))
    n_times   = len(ds.time)

    # ── Discover detector prefixes (e.g. "solexs_sdd2") ──
    band_vars = [v for v in ds.data_vars if v.endswith("_band_A")]
    if not band_vars:
        raise ValueError(
            "No band_A variable found in Dataset.  "
            "Ensure the Dataset was produced by module1.ingest_day()."
        )

    new_vars: dict = {}

    for band_a_var in band_vars:
        # prefix = everything before "_band_A"
        prefix = band_a_var[: -len("_band_A")]
        bands  = ["A", "B", "C", "D"]

        if verbose:
            logger.info("Preprocessing detector prefix: %s", prefix)

        # ── Quality mask: NaN out cadences that are outside GTI or otherwise bad ──
        #
        # WHY HERE (before smoothing, not after features):
        #   The SG filter interpolates over NaN spans when smoothing.
        #   If we mask bad cadences here, the filter sees a gap where the
        #   spike was and interpolates cleanly from its neighbours.
        #   If we masked only after features, the spike would already have
        #   been smoothed into neighbouring cadences and contaminated the
        #   background / derivative estimates.
        #
        # IMPORTANT — flag semantics:
        #   quality == 0 does NOT mean "clean".  In our bitmask, quality = 0
        #   means IN_GTI bit is unset, which makes the cadence NOT usable.
        #   Always call is_usable() rather than comparing to a literal.
        from pipeline.module1.quality import is_usable as _is_usable
        q_var   = f"{prefix}_quality"
        if q_var in ds.data_vars:
            q_arr   = ds[q_var].values.astype(np.uint8)
            good    = _is_usable(q_arr)           # (n_times,) bool
            n_bad   = int((~good).sum())
            if verbose:
                logger.info(
                    "%s: masking %d / %d bad-quality cadences before smoothing "
                    "(%.1f%% excluded)",
                    prefix, n_bad, n_times, 100.0 * n_bad / max(n_times, 1),
                )
        else:
            good = np.ones(n_times, dtype=bool)   # no quality var → trust all
            if verbose:
                logger.warning("%s: no quality variable found; treating all cadences as good", prefix)

        # ── Collect band arrays (bad cadences → NaN before smoothing) ──
        raw_bands: dict = {}
        for b in bands:
            var = f"{prefix}_band_{b}"
            if var in ds.data_vars:
                arr = ds[var].values.astype(float)
                arr[~good] = np.nan    # mask in-place on the copy
                raw_bands[b] = arr

        if not raw_bands:
            continue

        # ── Step 1: Savitzky-Golay smoothing ──
        smooth_bands: dict = {}
        for b, arr in raw_bands.items():
            smooth_bands[b] = sg_smooth(
                arr, window_length=sg_window, polyorder=sg_order,
                cadence_s=cadence_s,
            )
        if verbose:
            n_nan_after = np.isnan(smooth_bands["A"]).sum()
            logger.info("SG smooth done.  Band A NaN after smooth: %d", n_nan_after)

        # ── Step 2: SNIP background estimation ──
        bg_bands: dict = {}
        for b, arr in smooth_bands.items():
            bg_bands[b] = snip_background(arr, M=snip_M)
        if verbose:
            bg_a = bg_bands["A"]
            valid = bg_a[~np.isnan(bg_a)]
            logger.info(
                "SNIP done.  Band A background: mean=%.1f  max=%.1f",
                valid.mean() if len(valid) else 0,
                valid.max()  if len(valid) else 0,
            )

        # ── Step 3: Feature engineering ──
        feats = compute_features(smooth_bands, bg_bands, cadence_s=cadence_s)

        # ── Add smoothed bands and background to Dataset vars ──
        for b in smooth_bands:
            new_vars[f"{prefix}_band_{b}_smooth"]      = (["time"], smooth_bands[b])
            new_vars[f"{prefix}_band_{b}_background"]  = (["time"], bg_bands[b])

        # ── Add feature arrays as Dataset vars ──
        for feat_name, feat_arr in feats.items():
            # Skip redundant copies of the smoothed bands already added
            if feat_name.startswith("flux_smooth_") or feat_name.startswith("background_"):
                continue
            new_vars[f"{prefix}_{feat_name}"] = (["time"], feat_arr)

        if verbose:
            excess_a = feats["excess_A"]
            valid_ex = excess_a[~np.isnan(excess_a)]
            logger.info(
                "Features done.  excess_A: mean=%.3f  max=%.3f  std=%.3f",
                valid_ex.mean() if len(valid_ex) else 0,
                valid_ex.max()  if len(valid_ex) else 0,
                valid_ex.std()  if len(valid_ex) else 0,
            )

    # ── Step 4: Temporal split labels ──
    bounds     = compute_split_boundaries(
        n_times, train_frac, val_frac, test_frac, split_gap_s, cadence_s
    )
    split_arr  = build_split_array(bounds)
    new_vars["split"] = (["time"], split_arr)

    if verbose:
        logger.info("Split: %s", split_summary(split_arr))

    # ── Assemble extended Dataset ──
    ds_out = ds.assign({
        k: xr.DataArray(v[1], dims=v[0]) if isinstance(v, tuple) else v
        for k, v in new_vars.items()
    })

    ds_out.attrs["pipeline_module"] = "module1+2 v1.0"
    ds_out.attrs["sg_window"]       = sg_window
    ds_out.attrs["sg_order"]        = sg_order
    ds_out.attrs["snip_M"]          = snip_M
    ds_out.attrs["split_train_frac"]= train_frac
    ds_out.attrs["split_val_frac"]  = val_frac
    ds_out.attrs["split_test_frac"] = test_frac

    if not build_windows:
        return ds_out

    # ── Step 5 (optional): Sliding window dataset ──
    # Gather feature dict for the first detector prefix
    first_prefix = band_vars[0][: -len("_band_A")]
    feat_dict = {
        k.removeprefix(f"{first_prefix}_"): ds_out[k].values.astype(float)
        for k in ds_out.data_vars
        if k.startswith(f"{first_prefix}_") and "spectrum" not in k and "quality" not in k
    }

    from .windows import build_windows as _build_windows
    wd = _build_windows(
        features        = feat_dict,
        split_array     = split_arr,
        excess_A        = feat_dict.get("excess_A"),
        window_s        = window_s,
        stride_s        = stride_s,
        label_horizon_s = label_horizon_s,
        cadence_s       = cadence_s,
    )

    if verbose:
        logger.info(
            "WindowDataset: %d windows  shape=%s  imbalance=%s",
            len(wd.X), wd.X.shape, wd.imbalance_ratio,
        )

    return ds_out, wd
