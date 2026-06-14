"""
Tests for Module 2 — Preprocessing
=====================================
Run with:  python -m pytest pipeline/tests/test_module2.py -v

Covers: SG smoothing, SNIP background, features, split, windows, and the
end-to-end preprocess_day() on real FITS data.
"""

import numpy as np
import pytest


def _solexs_dir():
    import os
    for c in ["/tmp/solexs_data/AL1_SLX_L1_20260612_v1.0", "data/AL1_SLX_L1_20260612_v1.0"]:
        if os.path.isdir(c):
            return c
    pytest.skip("SoLEXS day directory not found")


# Module-level caches: ingest_day and preprocess_day each take ~25 s on real data.
# Without caching, the 5 TestPreprocessDay methods would call ingest_day 5×.
_DS_CACHE: dict = {}


def _ingested_ds():
    """Module-level cached Module 1 result (read real FITS only once per session)."""
    if "ds" not in _DS_CACHE:
        from pipeline.module1 import ingest_day
        _DS_CACHE["ds"] = ingest_day(solexs_dir=_solexs_dir(), verbose=False)
    return _DS_CACHE["ds"]


def _preprocessed_ds():
    """Module-level cached Module 2 result."""
    if "ds2" not in _DS_CACHE:
        from pipeline.module2 import preprocess_day
        _DS_CACHE["ds2"] = preprocess_day(_ingested_ds(), verbose=False)
    return _DS_CACHE["ds2"]


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 attribute additions
# ─────────────────────────────────────────────────────────────────────────────

class TestDaySummaryAttrs:
    def test_usable_fraction_in_attrs(self):
        ds = _ingested_ds()
        assert "usable_fraction" in ds.attrs
        frac = ds.attrs["usable_fraction"]
        assert 0.0 < frac <= 1.0

    def test_gap_seconds_is_list(self):
        ds = _ingested_ds()
        assert "gap_seconds" in ds.attrs
        assert isinstance(ds.attrs["gap_seconds"], list)

    def test_gap_seconds_sorted_descending(self):
        ds = _ingested_ds()
        gaps = ds.attrs["gap_seconds"]
        assert gaps == sorted(gaps, reverse=True)

    def test_quiet_day_is_bool(self):
        ds = _ingested_ds()
        assert "quiet_day" in ds.attrs
        # xarray may load booleans as numpy bool; accept both
        assert ds.attrs["quiet_day"] in (True, False, np.True_, np.False_)

    def test_sdd1_offline_is_bool(self):
        ds = _ingested_ds()
        assert "sdd1_offline" in ds.attrs

    def test_peak_flux_ratio_positive(self):
        ds = _ingested_ds()
        assert ds.attrs.get("peak_flux_ratio", 0) >= 0

    def test_this_day_peak_ratio_matches_data(self):
        ds = _ingested_ds()
        # 2026-06-12: Band A max=432, median=32 → ratio ~13.5 → NOT quiet
        # Verify the ratio is physically consistent (>1, <200)
        ratio = ds.attrs["peak_flux_ratio"]
        assert 1.0 < ratio < 200.0, f"peak_flux_ratio={ratio} out of expected range"
        # quiet_day must be consistent with ratio vs threshold (5.0)
        assert ds.attrs["quiet_day"] == (ratio < 5.0)

    def test_sdd1_offline_this_day(self):
        ds = _ingested_ds()
        assert ds.attrs["sdd1_offline"] is True or ds.attrs["sdd1_offline"] == np.True_


# ─────────────────────────────────────────────────────────────────────────────
# detrend.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSGSmooth:
    def test_no_nan_output_matches_scipy(self):
        from scipy.signal import savgol_filter
        from pipeline.module2.detrend import sg_smooth
        rng = np.random.default_rng(0)
        x = rng.normal(10, 1, 200).astype(float)
        result   = sg_smooth(x, window_length=11, polyorder=2)
        expected = savgol_filter(x, 11, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nan_positions_preserved(self):
        from pipeline.module2.detrend import sg_smooth
        x = np.ones(100, dtype=float)
        x[20:25] = np.nan
        result = sg_smooth(x, window_length=11, polyorder=2)
        assert np.all(np.isnan(result[20:25]))
        assert not np.any(np.isnan(result[:15]))
        assert not np.any(np.isnan(result[30:]))

    def test_peak_preservation(self):
        """SG should preserve sharp peak height better than moving average."""
        from pipeline.module2.detrend import sg_smooth
        # Synthetic flare: step rise over 30 s, sharp peak at t=200
        t = np.arange(400)
        x = np.ones(400, dtype=float) * 10
        x[180:200] = 10 + (t[180:200] - 180) * 2   # rise
        x[200] = 50.0                                 # peak
        x[200:220] = 50 - (t[200:220] - 200) * 2    # decay
        sg = sg_smooth(x, window_length=31, polyorder=2)
        # SG should preserve at least 85% of the peak
        assert sg[200] >= 0.85 * 50.0, f"Peak too attenuated: {sg[200]:.1f} < 42.5"

    def test_all_nan_returns_nan(self):
        from pipeline.module2.detrend import sg_smooth
        x = np.full(100, np.nan)
        result = sg_smooth(x, window_length=11, polyorder=2)
        assert np.all(np.isnan(result))

    def test_guard_margin_applied_near_large_gap(self):
        from pipeline.module2.detrend import sg_smooth
        x = np.ones(300, dtype=float)
        # 80-sample gap at the middle (> 60-s default guard)
        x[100:180] = np.nan
        result = sg_smooth(x, window_length=11, polyorder=2, max_gap_guard_s=60.0)
        # A few samples just outside the gap should also be NaN (guard margin)
        assert np.isnan(result[95])   # within guard_samples=5 of gap start


# ─────────────────────────────────────────────────────────────────────────────
# background.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSNIP:
    def test_background_leq_signal(self):
        """SNIP background must always be ≤ the input signal (it only clips down)."""
        from pipeline.module2.background import snip_background
        rng = np.random.default_rng(1)
        x = rng.poisson(50, 1000).astype(float)
        bg = snip_background(x, M=50)
        valid = ~np.isnan(bg)
        # Allow small numerical tolerance
        assert np.all(bg[valid] <= x[valid] + 1e-6), \
            "Background must be ≤ input signal everywhere"

    def test_flat_signal_unchanged(self):
        """SNIP on a constant signal should return approximately that constant."""
        from pipeline.module2.background import snip_background
        x = np.full(500, 100.0)
        bg = snip_background(x, M=50)
        np.testing.assert_allclose(bg, 100.0, rtol=0.01)

    def test_spike_removed_from_background(self):
        """A sharp spike should NOT appear in the background estimate."""
        from pipeline.module2.background import snip_background
        x = np.full(600, 20.0)
        x[295:305] = 500.0   # sharp 10-sample spike
        bg = snip_background(x, M=100)
        # Background at the spike position should be close to 20, not 500
        assert bg[300] < 50.0, \
            f"SNIP failed to remove spike from background: bg[300]={bg[300]:.1f}"

    def test_nan_preserved(self):
        from pipeline.module2.background import snip_background
        x = np.full(200, 10.0)
        x[50:60] = np.nan
        bg = snip_background(x, M=30)
        assert np.all(np.isnan(bg[50:60]))
        assert not np.any(np.isnan(bg[:45]))

    def test_subtract_background_shapes(self):
        from pipeline.module2.background import subtract_background
        x  = np.random.default_rng(2).normal(50, 5, 200).astype(float)
        bg = np.full(200, 40.0)
        resid, excess = subtract_background(x, bg)
        assert resid.shape == (200,)
        assert excess.shape == (200,)

    def test_subtract_background_values(self):
        from pipeline.module2.background import subtract_background
        x  = np.array([50.0, 100.0, 30.0])
        bg = np.array([40.0,  40.0, 40.0])
        resid, excess = subtract_background(x, bg)
        np.testing.assert_allclose(resid, [10.0, 60.0, -10.0], atol=1e-9)
        assert excess[1] > excess[0]   # 100 has larger excess than 50


# ─────────────────────────────────────────────────────────────────────────────
# features.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatures:
    def _make_bands(self, n=3600):
        rng = np.random.default_rng(3)
        return {
            "A": rng.poisson(50, n).astype(float),
            "B": rng.poisson(2,  n).astype(float),
            "C": rng.poisson(1,  n).astype(float),
            "D": rng.poisson(0,  n).astype(float),
        }

    def _make_bg(self, bands):
        return {b: arr * 0.9 for b, arr in bands.items()}

    def test_output_keys(self):
        from pipeline.module2.features import compute_features
        bands = self._make_bands()
        bg    = self._make_bg(bands)
        feats = compute_features(bands, bg)
        required = [
            "flux_smooth_A", "background_A", "residual_A", "excess_A",
            "derivative_1s", "derivative_60s", "rate_of_rise",
            "hardness_ratio", "softness_ratio",
            "rolling_std_5min", "rolling_std_15min", "cumulative_excess",
        ]
        for k in required:
            assert k in feats, f"Missing feature: {k}"

    def test_all_feature_shapes(self):
        from pipeline.module2.features import compute_features
        n = 3600
        bands = self._make_bands(n)
        bg    = self._make_bg(bands)
        feats = compute_features(bands, bg)
        for name, arr in feats.items():
            assert arr.shape == (n,), f"{name}: expected ({n},), got {arr.shape}"

    def test_hardness_ratio_rises_with_band_c(self):
        from pipeline.module2.features import compute_features
        n = 100
        bands = {"A": np.full(n, 50.0), "B": np.full(n, 5.0),
                 "C": np.full(n, 1.0),  "D": np.full(n, 0.0)}
        bg    = {"A": np.full(n, 45.0), "B": np.full(n, 4.0),
                 "C": np.full(n, 0.8),  "D": np.full(n, 0.0)}
        feats1 = compute_features(bands, bg)
        bands["C"] = np.full(n, 10.0)   # Band C 10× higher
        feats2 = compute_features(bands, bg)
        assert feats2["hardness_ratio"][50] > feats1["hardness_ratio"][50]

    def test_derivative_positive_on_rising_signal(self):
        from pipeline.module2.features import compute_features
        t = np.arange(1000)
        bands = {
            "A": t.astype(float),   # monotonically rising
            "B": np.ones(1000),
            "C": np.ones(1000),
            "D": np.zeros(1000),
        }
        bg = {b: np.zeros(1000) for b in bands}
        feats = compute_features(bands, bg)
        deriv = feats["derivative_1s"]
        valid = deriv[~np.isnan(deriv)]
        assert np.all(valid > 0), "Derivative should be positive on rising signal"

    def test_cumulative_excess_non_decreasing(self):
        from pipeline.module2.features import compute_features
        n = 200
        bands = {"A": np.full(n, 50.0), "B": np.ones(n),
                 "C": np.ones(n), "D": np.zeros(n)}
        bg    = {"A": np.full(n, 40.0), "B": np.ones(n),
                 "C": np.ones(n), "D": np.zeros(n)}
        feats = compute_features(bands, bg)
        ce = feats["cumulative_excess"]
        valid = ce[~np.isnan(ce)]
        diffs = np.diff(valid)
        assert np.all(diffs >= -1e-9), "Cumulative excess must be non-decreasing"


# ─────────────────────────────────────────────────────────────────────────────
# split.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_fractions_sum_to_one(self):
        from pipeline.module2.split import compute_split_boundaries
        b = compute_split_boundaries(86400, 0.70, 0.15, 0.15)
        assert b.n_total == 86400

    def test_split_array_covers_all_samples(self):
        from pipeline.module2.split import compute_split_boundaries, build_split_array
        b  = compute_split_boundaries(86400)
        sa = build_split_array(b)
        assert len(sa) == 86400

    def test_train_before_val_before_test(self):
        from pipeline.module2.split import (
            compute_split_boundaries, build_split_array, TRAIN, VAL, TEST
        )
        b  = compute_split_boundaries(86400)
        sa = build_split_array(b)
        # Find last train index and first val index
        train_indices = np.where(sa == TRAIN)[0]
        val_indices   = np.where(sa == VAL)[0]
        test_indices  = np.where(sa == TEST)[0]
        assert train_indices[-1] < val_indices[0], "Train must end before val starts"
        assert val_indices[-1] < test_indices[0],  "Val must end before test starts"

    def test_gap_between_splits(self):
        from pipeline.module2.split import (
            compute_split_boundaries, build_split_array, TRAIN, VAL, GAP
        )
        b  = compute_split_boundaries(86400, gap_s=1800.0)
        sa = build_split_array(b)
        # There should be a GAP region between TRAIN and VAL
        train_end = np.where(sa == TRAIN)[0][-1]
        val_start = np.where(sa == VAL)[0][0]
        gap_region = sa[train_end + 1 : val_start]
        assert np.all(gap_region == GAP), "Gap region must be all GAP labels"

    def test_fractions_approximately_correct(self):
        from pipeline.module2.split import (
            compute_split_boundaries, build_split_array, TRAIN, VAL, TEST, split_summary
        )
        n = 100000
        b  = compute_split_boundaries(n, gap_s=0.0)   # no gap for clean math
        sa = build_split_array(b)
        s  = split_summary(sa)
        assert abs(s["train_samples"] / n - 0.70) < 0.02
        assert abs(s["val_samples"]   / n - 0.15) < 0.02
        assert abs(s["test_samples"]  / n - 0.15) < 0.02

    def test_invalid_fractions_raises(self):
        from pipeline.module2.split import compute_split_boundaries
        with pytest.raises(ValueError):
            compute_split_boundaries(86400, train_frac=0.5, val_frac=0.3, test_frac=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# windows.py
# ─────────────────────────────────────────────────────────────────────────────

class TestWindows:
    def _make_features_and_split(self, n=43200, n_feats=5):
        """12-hour synthetic dataset (large enough for 30-min windows in all splits)."""
        from pipeline.module2.split import (
            compute_split_boundaries, build_split_array
        )
        rng = np.random.default_rng(42)
        feat_names = [f"f{i}" for i in range(n_feats)]
        features = {k: rng.normal(0, 1, n).astype(float) for k in feat_names}
        features["excess_A"] = rng.exponential(0.5, n).astype(float)
        bounds = compute_split_boundaries(n, gap_s=0.0)
        split_arr = build_split_array(bounds)
        return features, split_arr

    def test_window_shape(self):
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        features, split_arr = self._make_features_and_split()
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S)
        n_feats = len(features)
        assert wd.X.ndim == 3
        assert wd.X.shape[1] == WINDOW_S
        assert wd.X.shape[2] == n_feats

    def test_labels_binary(self):
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        features, split_arr = self._make_features_and_split()
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S)
        assert set(wd.y_binary.tolist()).issubset({0, 1})

    def test_no_boundary_windows(self):
        """No window should span two different split partitions."""
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        from pipeline.module2.split import TRAIN, VAL, TEST, GAP
        features, split_arr = self._make_features_and_split()
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S)
        # Every window's split label must be one of {TRAIN, VAL, TEST}
        assert set(wd.splits.tolist()).issubset({TRAIN, VAL, TEST})

    def test_splits_in_order(self):
        """window_starts within each partition must be monotonically increasing."""
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        from pipeline.module2.split import TRAIN
        features, split_arr = self._make_features_and_split()
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S)
        train_starts = wd.window_starts[wd.splits == TRAIN]
        assert np.all(np.diff(train_starts) > 0)

    def test_get_partition_returns_correct_subset(self):
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        from pipeline.module2.split import TRAIN, VAL
        features, split_arr = self._make_features_and_split()
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S)
        X_train, yb_train, _ = wd.get_partition(TRAIN)
        X_val,   yb_val,   _ = wd.get_partition(VAL)
        assert len(X_train) + len(X_val) <= len(wd.X)
        assert len(X_train) > 0
        assert len(X_val) > 0

    def test_oversample_increases_flare_fraction(self):
        from pipeline.module2.windows import build_windows, WINDOW_S, STRIDE_S
        from pipeline.module2.split import TRAIN
        features, split_arr = self._make_features_and_split()
        # Inject a few "flares" in excess_A
        features["excess_A"][1000:1010] = 20.0
        wd = build_windows(features, split_arr, window_s=WINDOW_S, stride_s=STRIDE_S,
                           flare_threshold=5.0)
        X_orig, yb_orig, _ = wd.get_partition(TRAIN, balance=False)
        X_bal,  yb_bal,  _ = wd.get_partition(TRAIN, balance=True)
        orig_frac = yb_orig.mean()
        bal_frac  = yb_bal.mean()
        assert bal_frac >= orig_frac, "Balanced fraction should be >= original"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: preprocess_day on real FITS data
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessDay:
    """
    End-to-end tests against real SoLEXS FITS data.

    All tests share a single cached call to preprocess_day() via
    _preprocessed_ds() — the real dataset takes ~25 s to ingest and
    ~5 s to preprocess; without caching this class would time out.
    """

    def test_returns_xarray_dataset(self):
        import xarray as xr
        assert isinstance(_preprocessed_ds(), xr.Dataset)

    def test_has_smooth_and_background_vars(self):
        ds = _preprocessed_ds()
        assert "solexs_sdd2_band_A_smooth"     in ds
        assert "solexs_sdd2_band_A_background" in ds

    def test_has_feature_vars(self):
        ds = _preprocessed_ds()
        for feat in ["derivative_1s", "rate_of_rise", "hardness_ratio",
                     "rolling_std_5min", "cumulative_excess", "excess_A"]:
            assert f"solexs_sdd2_{feat}" in ds, f"Missing variable: solexs_sdd2_{feat}"

    def test_has_split_variable(self):
        assert "split" in _preprocessed_ds()

    def test_split_labels_cover_expected_fractions(self):
        from pipeline.module2.split import split_summary
        sa   = _preprocessed_ds()["split"].values
        summ = split_summary(sa)
        n    = summ["total"]
        assert abs(summ["train_samples"] / n - 0.70) < 0.03
        assert abs(summ["val_samples"]   / n - 0.15) < 0.03
        assert abs(summ["test_samples"]  / n - 0.15) < 0.03

    def test_background_leq_smooth(self):
        ds     = _preprocessed_ds()
        smooth = ds["solexs_sdd2_band_A_smooth"].values
        bg     = ds["solexs_sdd2_band_A_background"].values
        valid  = ~np.isnan(smooth) & ~np.isnan(bg)
        assert np.all(bg[valid] <= smooth[valid] + 1.0), \
            "Background must be ≤ smooth signal"

    def test_excess_mostly_non_negative(self):
        excess = _preprocessed_ds()["solexs_sdd2_excess_A"].values
        valid  = excess[~np.isnan(excess)]
        assert np.mean(valid >= -0.5) > 0.90

    def test_pipeline_preserves_time_length(self):
        ds_in  = _ingested_ds()
        ds_out = _preprocessed_ds()
        assert len(ds_out.time) == len(ds_in.time)

    def test_build_windows_returns_tuple(self):
        from pipeline.module2 import preprocess_day, WindowDataset
        result = preprocess_day(_ingested_ds(), verbose=False, build_windows=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[1], WindowDataset)
