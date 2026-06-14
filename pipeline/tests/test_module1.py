"""
Tests for Module 1 — Data Ingestion
=====================================
Run with:  python -m pytest pipeline/tests/test_module1.py -v

Each test checks one specific contract of the pipeline.  When a test fails
it tells you exactly which invariant broke, not just "something went wrong".
"""

import gzip
import io
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solexs_dir():
    """Path to the SoLEXS day directory extracted from the attached zip."""
    import os
    # Try extracted location first (CI / bash test run)
    candidates = [
        "/tmp/solexs_data/AL1_SLX_L1_20260612_v1.0",
        "data/AL1_SLX_L1_20260612_v1.0",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    pytest.skip("SoLEXS day directory not found; extract the zip first")


# ─────────────────────────────────────────────────────────────────────────────
# formats.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFormats:
    def test_discover_products_finds_files(self):
        from pipeline.module1.formats import discover_products, ProductKind, Instrument
        products = discover_products(_solexs_dir())
        assert len(products) > 0, "discover_products returned nothing"
        kinds = {p.kind for p in products}
        assert ProductKind.SPECTRUM in kinds or ProductKind.LIGHTCURVE in kinds

    def test_discover_products_instrument_detection(self):
        from pipeline.module1.formats import discover_products, Instrument
        products = discover_products(_solexs_dir())
        for p in products:
            assert p.instrument == Instrument.SOLEXS, (
                f"Expected SoLEXS, got {p.instrument} for {p.path.name}"
            )

    def test_discover_products_detector_labels(self):
        from pipeline.module1.formats import discover_products
        products = discover_products(_solexs_dir())
        detectors = {p.detector for p in products}
        # SDD2 must be present (SDD1 has only a GTI file this day)
        assert "SDD2" in detectors, f"SDD2 not found in detectors: {detectors}"

    def test_product_hdul_opens(self):
        from pipeline.module1.formats import discover_products, ProductKind
        products = discover_products(_solexs_dir())
        spectrum_products = [p for p in products if p.kind == ProductKind.SPECTRUM]
        assert spectrum_products, "No .pi products found"
        hdul = spectrum_products[0].hdul()
        assert hdul is not None
        assert len(hdul) > 1


# ─────────────────────────────────────────────────────────────────────────────
# channels.py
# ─────────────────────────────────────────────────────────────────────────────

class TestChannels:
    def test_calibration_energy_range(self):
        from pipeline.module1.channels import SOLEXS_CALIBRATION, energy_axis
        e = energy_axis(SOLEXS_CALIBRATION)
        assert len(e) == 340
        assert e[0] == pytest.approx(SOLEXS_CALIBRATION.e_min_keV, abs=0.001)

    def test_science_bands_cover_spectrum(self):
        from pipeline.module1.channels import SOLEXS_CALIBRATION
        cal = SOLEXS_CALIBRATION
        for band, (lo, hi) in cal.science_bands.items():
            ch_lo, ch_hi = cal.band_channel_range(band)
            assert 0 <= ch_lo < ch_hi < cal.n_channels, (
                f"Band {band}: channel range [{ch_lo},{ch_hi}] invalid"
            )

    def test_extract_band_counts_shape(self):
        from pipeline.module1.channels import SOLEXS_CALIBRATION, extract_band_counts
        rng = np.random.default_rng(42)
        counts = rng.poisson(10, size=(100, 340)).astype(float)
        band_a = extract_band_counts(counts, "A", SOLEXS_CALIBRATION)
        assert band_a.shape == (100,)
        assert not np.any(np.isnan(band_a))

    def test_extract_band_counts_all_nan_rows(self):
        from pipeline.module1.channels import SOLEXS_CALIBRATION, extract_band_counts
        counts = np.full((5, 340), np.nan)
        result = extract_band_counts(counts, "A", SOLEXS_CALIBRATION)
        assert np.all(np.isnan(result)), "All-NaN row should produce NaN output"

    def test_total_counts_nan_propagation(self):
        from pipeline.module1.channels import total_counts
        counts = np.full((3, 340), np.nan)
        result = total_counts(counts)
        assert np.all(np.isnan(result))

    def test_saturation_flag_overflow(self):
        from pipeline.module1.channels import flag_saturated_rows
        counts = np.zeros((10, 340))
        counts[5, 100] = 2e6   # overflow row
        sat = flag_saturated_rows(counts, saturation_cts_per_channel=1e6)
        assert sat[5], "Row 5 should be flagged as saturated"
        assert not sat[0], "Row 0 should not be flagged"


# ─────────────────────────────────────────────────────────────────────────────
# quality.py
# ─────────────────────────────────────────────────────────────────────────────

class TestQuality:
    def test_gti_mask_inside(self):
        from pipeline.module1.quality import gti_mask
        times = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intervals = [(150.0, 350.0)]
        mask = gti_mask(times, intervals)
        assert not mask[0]    # 100 < 150
        assert mask[1]        # 200 in [150, 350]
        assert mask[2]        # 300 in [150, 350]
        assert not mask[3]    # 400 > 350
        assert not mask[4]

    def test_gti_mask_multiple_intervals(self):
        from pipeline.module1.quality import gti_mask
        times = np.arange(0, 1000, 1.0)
        intervals = [(100.0, 200.0), (500.0, 600.0)]
        mask = gti_mask(times, intervals)
        assert mask[150]
        assert mask[550]
        assert not mask[50]
        assert not mask[350]

    def test_quality_flags_nan_row(self):
        from pipeline.module1.quality import build_quality_flags, QFlag
        times  = np.arange(10, dtype=float)
        counts = np.ones((10, 5))
        counts[3, :] = np.nan
        q = build_quality_flags(times, counts, [], cadence_s=1.0)
        assert q[3] & QFlag.NAN_ROW, "Row 3 should have NAN_ROW flag"
        assert not (q[0] & QFlag.NAN_ROW), "Row 0 should not have NAN_ROW flag"

    def test_is_usable_requires_gti(self):
        from pipeline.module1.quality import build_quality_flags, is_usable, QFlag
        times  = np.arange(10, dtype=float)
        counts = np.ones((10, 5))
        # No GTI provided → IN_GTI bit never set → nothing usable
        q = build_quality_flags(times, counts, [], cadence_s=1.0)
        assert not np.any(is_usable(q)), (
            "Without GTI, no cadence should be usable"
        )

    def test_is_usable_with_gti(self):
        from pipeline.module1.quality import build_quality_flags, is_usable
        times  = np.arange(10, dtype=float)
        counts = np.ones((10, 5))
        q = build_quality_flags(times, counts, [(2.0, 7.0)], cadence_s=1.0)
        usable = is_usable(q)
        assert usable[3]
        assert usable[5]
        assert not usable[0]
        assert not usable[9]


# ─────────────────────────────────────────────────────────────────────────────
# gaps.py
# ─────────────────────────────────────────────────────────────────────────────

class TestGaps:
    def test_find_gaps_simple(self):
        from pipeline.module1.gaps import find_gaps
        usable = np.array([True, True, False, False, False, True, True])
        times  = np.arange(7, dtype=float)
        gaps = find_gaps(usable, times)
        assert len(gaps) == 1
        assert gaps[0][0] == 2   # start index
        assert gaps[0][1] == 4   # end index

    def test_find_gaps_no_gaps(self):
        from pipeline.module1.gaps import find_gaps
        usable = np.ones(10, dtype=bool)
        times  = np.arange(10, dtype=float)
        gaps = find_gaps(usable, times)
        assert len(gaps) == 0

    def test_interpolate_short_gap_fills(self):
        from pipeline.module1.gaps import interpolate_gaps
        usable = np.ones(10, dtype=bool)
        usable[3] = False
        usable[4] = False
        values = np.array([0.0, 1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0])
        quality = np.zeros(10, dtype=np.uint8)
        time_a  = np.arange(10, dtype=float)
        filled = interpolate_gaps(values, usable, quality, max_gap_s=10.0, time_array=time_a)
        assert not np.isnan(filled[3]), "Short gap should be filled"
        assert not np.isnan(filled[4])

    def test_interpolate_long_gap_stays_nan(self):
        from pipeline.module1.gaps import interpolate_gaps
        usable = np.ones(10, dtype=bool)
        for i in range(2, 8):
            usable[i] = False
        values = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0, 9.0])
        quality = np.zeros(10, dtype=np.uint8)
        time_a  = np.arange(10, dtype=float)
        filled = interpolate_gaps(values, usable, quality, max_gap_s=3.0, time_array=time_a)
        # Gap is 6 seconds > 3 → should stay NaN
        assert np.isnan(filled[4]), "Long gap should remain NaN"


# ─────────────────────────────────────────────────────────────────────────────
# align.py
# ─────────────────────────────────────────────────────────────────────────────

class TestAlign:
    def test_build_master_grid_length(self):
        from pipeline.module1.align import build_master_grid
        grid = build_master_grid(0.0, 99.0, cadence_s=1.0)
        assert len(grid) == 100

    def test_align_exact_match(self):
        from pipeline.module1.align import build_master_grid, align_to_grid
        grid   = build_master_grid(0.0, 9.0, 1.0)
        source = np.array([0.0, 3.0, 7.0])
        idx, res, valid = align_to_grid(source, grid)
        assert np.all(valid)
        assert list(idx) == [0, 3, 7]
        assert np.all(np.abs(res) < 1.0)   # residuals in ms

    def test_align_sub_millisecond_offset(self):
        from pipeline.module1.align import build_master_grid, align_to_grid
        grid   = build_master_grid(0.0, 9.0, 1.0)
        source = np.array([2.0004, 5.0002])  # 0.4 ms, 0.2 ms offsets
        idx, res, valid = align_to_grid(source, grid, tolerance_s=0.5)
        assert np.all(valid)
        assert list(idx) == [2, 5]

    def test_align_outside_tolerance(self):
        from pipeline.module1.align import build_master_grid, align_to_grid
        grid   = build_master_grid(0.0, 9.0, 1.0)
        # 2.51 is 0.51 s from grid point 2 and 0.49 s from grid point 3 — nearest is 3
        # BUT with tolerance=0.3 s, 0.49 s > 0.3 s → should be invalid
        source = np.array([2.51])
        idx, res, valid = align_to_grid(source, grid, tolerance_s=0.3)
        assert not valid[0], "Point outside tolerance should be marked invalid"
        assert idx[0] == -1


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: ingest_day on real FITS data
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestDay:
    def test_returns_xarray_dataset(self):
        import xarray as xr
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        assert isinstance(ds, xr.Dataset)

    def test_has_expected_variables(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        # SDD2 is the only detector with spectrum data this day
        assert "solexs_sdd2_spectrum" in ds
        assert "solexs_sdd2_total"    in ds
        assert "solexs_sdd2_quality"  in ds
        assert "solexs_sdd2_band_A"   in ds
        assert "solexs_sdd2_band_B"   in ds
        assert "solexs_sdd2_band_C"   in ds

    def test_time_dimension_is_datetime64(self):
        from pipeline.module1 import ingest_day
        import numpy as np
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        assert np.issubdtype(ds.time.dtype, np.datetime64), (
            f"time dimension should be datetime64, got {ds.time.dtype}"
        )

    def test_spectrum_shape(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        spec = ds["solexs_sdd2_spectrum"]
        n_times, n_ch = spec.shape
        assert n_times == 86400, f"Expected 86400 time steps, got {n_times}"
        assert n_ch == 340, f"Expected 340 channels, got {n_ch}"

    def test_nan_fraction_reasonable(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        total = ds["solexs_sdd2_total"].values
        nan_frac = np.isnan(total).mean()
        # We know 16209/86400 ≈ 18.8% are NaN from direct inspection
        assert 0.10 < nan_frac < 0.30, f"NaN fraction {nan_frac:.3f} outside expected range"

    def test_quality_flags_in_gti(self):
        from pipeline.module1 import ingest_day
        from pipeline.module1.quality import QFlag, is_usable
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        q = ds["solexs_sdd2_quality"].values
        n_usable = is_usable(q).sum()
        n_total  = len(q)
        # SDD2 has 49828 s of GTI out of 86400 s
        assert 40000 < n_usable < 60000, (
            f"Usable cadences {n_usable} outside expected range for this day"
        )

    def test_energy_axis_in_expected_range(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        e = ds["solexs_sdd2_energy_keV"].values
        assert e.min() >= 1.0, "Energy below 1 keV unexpected for SoLEXS"
        assert e.max() <= 10.0, "Energy above 10 keV unexpected for SoLEXS nominal range"

    def test_band_counts_non_negative_where_valid(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        for band in ["A", "B", "C", "D"]:
            arr = ds[f"solexs_sdd2_band_{band}"].values
            valid = arr[~np.isnan(arr)]
            assert np.all(valid >= 0), f"Band {band} has negative counts"

    def test_metadata_preserved(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        assert ds.attrs.get("obs_date") == "20260612"
        assert "SoLEXS" in ds.attrs.get("instrument", "")

    def test_no_duplicate_timestamps(self):
        from pipeline.module1 import ingest_day
        ds = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
        times = ds.time.values
        assert len(times) == len(np.unique(times)), "Duplicate timestamps in output"
