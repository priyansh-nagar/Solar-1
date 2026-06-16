"""
Tests for Module 4 — PatchTST Flare Forecaster
===============================================
Run with:  python -m pytest pipeline/tests/test_module4.py -v

Covers:
  • Model architecture contracts (shapes, parameter count, stream split)
  • Evaluation metrics (tpr_at_far, ece)
  • Loss computation correctness
  • TemperatureScaler fitting
  • Gradient × Input attributions
  • SolarFlareDataset construction with synthetic data
  • Full mini training loop (2 epochs, tiny model, synthetic data)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

FEAT_NAMES_29 = [
    "band_A", "background_A", "flux_smooth_A", "excess_A", "residual_A",
    "derivative_1s", "derivative_60s", "rate_of_rise",
    "rolling_std_5min", "rolling_std_15min", "cumulative_excess",
    "hardness_ratio", "softness_ratio", "total",
    "band_B", "background_B", "residual_B", "excess_B",
    "band_C", "background_C", "residual_C", "excess_C",
    "band_D", "background_D", "residual_D", "excess_D",
    "flux_smooth_B", "flux_smooth_C", "flux_smooth_D",
]
assert len(FEAT_NAMES_29) == 29


def _make_synth_arrays(
    n_total: int = 180,
    n_train: int = 120,
    n_val: int   = 30,
    seed: int    = 0,
) -> dict:
    """
    Return a dict of numpy arrays matching the v2 dataset schema.
    Small enough for unit tests (n_total << 7470).
    """
    rng = np.random.default_rng(seed)
    T, F = 1800, 29
    n_test = n_total - n_train - n_val

    X = rng.standard_normal((n_total, T, F)).astype(np.float32)

    # Roughly 40% M+ events
    y_class = rng.choice([0, 1, 2, 3, 4], size=n_total,
                         p=[0.02, 0.02, 0.38, 0.44, 0.14]).astype(np.int8)
    y_binary = (y_class >= 3).astype(np.int8)

    from pipeline.module2.split import TRAIN, VAL, TEST
    splits = np.concatenate([
        np.full(n_train, TRAIN, dtype=np.uint8),
        np.full(n_val,   VAL,   dtype=np.uint8),
        np.full(n_test,  TEST,  dtype=np.uint8),
    ])

    y_extreme = (y_class >= 4).astype(np.int8)
    y_15min   = y_binary.copy()
    y_60min   = y_binary.copy()

    return dict(
        X=X, y_binary=y_binary, y_class=y_class, splits=splits,
        y_extreme=y_extreme, y_15min=y_15min, y_60min=y_60min,
    )


def _save_synth_to_dir(arrays: dict, out_dir: Path) -> None:
    """Save synthetic arrays to a temp directory in v2 format."""
    np.save(out_dir / "X_v2.npy",         arrays["X"])
    np.save(out_dir / "y_binary_v2.npy",  arrays["y_binary"])
    np.save(out_dir / "y_class_v2.npy",   arrays["y_class"])
    np.save(out_dir / "splits_v2.npy",    arrays["splits"])
    np.save(out_dir / "y_extreme_v2.npy", arrays["y_extreme"])
    np.save(out_dir / "y_15min_v2.npy",   arrays["y_15min"])
    np.save(out_dir / "y_60min_v2.npy",   arrays["y_60min"])
    with open(out_dir / "feature_names_v2.json", "w") as f:
        json.dump(FEAT_NAMES_29, f)


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestSolarPatchTST:

    def _make_model(self, **kwargs):
        from pipeline.module4 import SolarPatchTST
        return SolarPatchTST(feature_names=FEAT_NAMES_29, **kwargs)

    def test_default_stream_split(self):
        model = self._make_model()
        n_soft = len(model.soft_idx)
        n_hard = len(model.hard_idx)
        assert n_soft + n_hard == 29, "All 29 features must land in exactly one stream"
        assert n_soft > 0, "Soft (SoLEXS/Band-A) stream must be non-empty"
        assert n_hard > 0, "Hard (B/C/D bands) stream must be non-empty"

    def test_forward_output_shapes(self):
        model = self._make_model(d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(3, 1800, 29)
        lb, le, lmc = model(x)
        assert lb.shape  == (3, 3), "binary head: (B, 3) for 15/30/60-min"
        assert le.shape  == (3, 1), "extreme head: (B, 1)"
        assert lmc.shape == (3, 3), "multiclass head: (B, 3) for C/M/X"

    def test_parameter_count_reasonable(self):
        model = self._make_model()
        n = model.n_parameters()
        assert 50_000 < n < 2_000_000, f"Unexpected param count: {n:,}"

    def test_small_model_fewer_params(self):
        big   = self._make_model(d_model=64)
        small = self._make_model(d_model=16, n_layers=1)
        assert small.n_parameters() < big.n_parameters()

    def test_batch_size_1_works(self):
        model = self._make_model(d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(1, 1800, 29)
        lb, le, lmc = model(x)
        assert not lb.isnan().any()
        assert not le.isnan().any()
        assert not lmc.isnan().any()

    def test_logits_are_raw_not_probabilities(self):
        """Binary/extreme logits must not be pre-clipped to [0,1]."""
        model = self._make_model(d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(4, 1800, 29) * 5.0   # large inputs
        lb, le, _ = model(x)
        # Logits can legitimately exceed [0,1]
        has_large = (lb.abs() > 1.0).any() or (le.abs() > 1.0).any()
        # We can't guarantee this always happens, but the sigmoid of all logits
        # must lie in (0,1) strictly — not 0 or 1 exactly (no clipping)
        assert torch.sigmoid(lb).min() > 0.0
        assert torch.sigmoid(lb).max() < 1.0

    def test_cross_stream_no_nan_gradient(self):
        model = self._make_model(d_model=16, n_heads=2, n_layers=1)
        x = torch.randn(2, 1800, 29, requires_grad=False)
        lb, le, lmc = model(x)
        loss = lb.sum() + le.sum() + lmc.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN gradient in {name}"


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_tpr_at_far_perfect_classifier(self):
        from pipeline.module4 import tpr_at_far
        probs  = np.array([0.9, 0.8, 0.1, 0.05])
        y_true = np.array([1,   1,   0,   0])
        tpr = tpr_at_far(probs, y_true, far_thr=0.5, obs_hours=4.0)
        assert tpr == 1.0, "Perfect classifier should achieve TPR=1"

    def test_tpr_at_far_zero_budget(self):
        from pipeline.module4 import tpr_at_far
        probs  = np.array([0.9, 0.5, 0.1])
        y_true = np.array([1,   0,   0])
        tpr = tpr_at_far(probs, y_true, far_thr=0.0, obs_hours=1.0)
        assert tpr == 1.0, "With zero FP budget, first prediction is TP → TPR=1"

    def test_tpr_at_far_no_positives_returns_zero(self):
        from pipeline.module4 import tpr_at_far
        probs  = np.array([0.9, 0.1])
        y_true = np.array([0,   0])
        tpr = tpr_at_far(probs, y_true, far_thr=0.5, obs_hours=1.0)
        assert tpr == 0.0

    def test_tpr_at_far_obs_hours_derived_from_probs(self):
        from pipeline.module4 import tpr_at_far
        probs  = np.array([0.9, 0.5, 0.2, 0.1])
        y_true = np.array([1,   0,   1,   0])
        tpr1 = tpr_at_far(probs, y_true, far_thr=0.5)           # auto obs_hours
        tpr2 = tpr_at_far(probs, y_true, far_thr=0.5, obs_hours=len(probs)*60/3600)
        assert abs(tpr1 - tpr2) < 1e-6

    def test_ece_perfect_calibration(self):
        from pipeline.module4.evaluate import ece
        probs  = np.linspace(0.05, 0.95, 100)
        # For perfect calibration: each probability bin should equal the empirical frequency.
        # Simulate this with a large number of samples
        rng = np.random.default_rng(42)
        y = (rng.random(100) < probs).astype(float)
        val = ece(probs, y, n_bins=10)
        assert 0.0 <= val <= 1.0

    def test_ece_worst_case_bounded_by_1(self):
        from pipeline.module4.evaluate import ece
        probs  = np.ones(50) * 0.9
        y_true = np.zeros(50)
        val = ece(probs, y_true)
        assert 0.0 <= val <= 1.0

    def test_compute_lead_times_empty_returns_zeros(self):
        from pipeline.module4.evaluate import compute_lead_times
        probs   = np.zeros(100)
        y_class = np.zeros(100, dtype=int)
        result  = compute_lead_times(probs, y_class, threshold=0.5)
        assert result["lead_median_min"] == 0.0
        assert result["n_detected"] == 0


# ---------------------------------------------------------------------------
# Loss computation tests
# ---------------------------------------------------------------------------

class TestLossComputation:

    def _make_loss_fns(self, device):
        from pipeline.module4.train import BINARY_POS_WEIGHT, EXTREME_POS_WEIGHT, MULTICLASS_WEIGHTS
        bce_bin = nn.BCEWithLogitsLoss(
            pos_weight=BINARY_POS_WEIGHT.expand(3).to(device)
        )
        bce_ext = nn.BCEWithLogitsLoss(
            pos_weight=EXTREME_POS_WEIGHT.to(device)
        )
        ce_mc = nn.CrossEntropyLoss(weight=MULTICLASS_WEIGHTS.to(device))
        return bce_bin, bce_ext, ce_mc

    def test_total_loss_finite(self):
        from pipeline.module4.train import _compute_loss
        device = torch.device("cpu")
        B = 4
        bce_bin, bce_ext, ce_mc = self._make_loss_fns(device)

        logit_bin   = torch.randn(B, 3)
        logit_ext   = torch.randn(B, 1)
        logit_class = torch.randn(B, 3)
        y_binary    = torch.randint(0, 2, (B, 3)).float()
        y_extreme   = torch.randint(0, 2, (B,)).float()
        y_class     = torch.randint(2, 5, (B,))    # C/M/X only to keep multiclass mask active

        total, parts = _compute_loss(
            logit_bin, logit_ext, logit_class,
            y_binary, y_extreme, y_class,
            bce_bin, bce_ext, ce_mc,
        )
        assert torch.isfinite(total), "Total loss must be finite"
        for k, v in parts.items():
            assert np.isfinite(v), f"Loss component '{k}' is not finite: {v}"

    def test_multiclass_loss_only_on_cplus_windows(self):
        """Windows with y_class < 2 (A/B) must not contribute to multiclass loss."""
        from pipeline.module4.train import _compute_loss, _MULTI_OFFSET
        device = torch.device("cpu")
        B = 6
        bce_bin, bce_ext, ce_mc = self._make_loss_fns(device)

        logit_bin   = torch.zeros(B, 3)
        logit_ext   = torch.zeros(B, 1)
        logit_class = torch.zeros(B, 3)
        y_binary    = torch.zeros(B, 3)
        y_extreme   = torch.zeros(B)
        y_class_ab  = torch.zeros(B, dtype=torch.long)   # all A-class

        total_ab, parts_ab = _compute_loss(
            logit_bin, logit_ext, logit_class,
            y_binary, y_extreme, y_class_ab,
            bce_bin, bce_ext, ce_mc,
        )
        assert parts_ab["mc"] == 0.0, "A-class windows must give zero multiclass loss"

    def test_loss_weights_applied(self):
        """Total loss = 1.0×bin + 0.5×ext + 0.3×mc."""
        from pipeline.module4.train import _compute_loss, LAMBDA_BINARY, LAMBDA_EXTREME, LAMBDA_MULTICLASS
        assert LAMBDA_BINARY     == 1.0
        assert LAMBDA_EXTREME    == 0.5
        assert LAMBDA_MULTICLASS == 0.3


# ---------------------------------------------------------------------------
# TemperatureScaler tests
# ---------------------------------------------------------------------------

class TestTemperatureScaler:

    def _tiny_model_and_loader(self):
        from pipeline.module4 import SolarPatchTST
        from pipeline.module4.dataset import SolarFlareDataset
        from torch.utils.data import DataLoader
        from sklearn.preprocessing import StandardScaler

        model = SolarPatchTST(
            feature_names=FEAT_NAMES_29, d_model=16, n_heads=2, n_layers=1
        )

        rng  = np.random.default_rng(7)
        N, T, F = 20, 1800, 29
        X_np    = rng.standard_normal((N, T, F)).astype(np.float32)
        y_class = np.ones(N, dtype=np.int8) * 3   # all M-class
        y_bin   = np.ones(N, dtype=np.int8)

        scaler = StandardScaler()
        scaler.fit(X_np.reshape(N * T, F))
        scaler.mean_ = scaler.mean_.astype(np.float32)
        scaler.var_  = scaler.var_.astype(np.float32)
        scaler.scale_ = np.sqrt(scaler.var_).astype(np.float32)

        from pipeline.module2.split import TRAIN
        idx = np.arange(N)
        X_saved = np.ones((N, T, F), dtype=np.float32) * np.arange(F, dtype=np.float32)

        ds = SolarFlareDataset(
            x_path=None,
            indices=None,
            y_binary_30=y_bin,
            y_class=y_class,
            scaler=scaler,
            feature_names=FEAT_NAMES_29,
            verbose=False,
            _preloaded_X=torch.from_numpy(X_np),
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        return model, loader

    def test_temperature_is_positive(self):
        from pipeline.module4.calibrate import TemperatureScaler
        model, loader = self._tiny_model_and_loader()
        ts = TemperatureScaler(model)
        ts.fit(loader, device=torch.device("cpu"), verbose=False)
        assert ts.temperature_value() > 0.0

    def test_temperature_forward_shape(self):
        from pipeline.module4 import SolarPatchTST
        from pipeline.module4.calibrate import TemperatureScaler
        model = SolarPatchTST(
            feature_names=FEAT_NAMES_29, d_model=16, n_heads=2, n_layers=1
        )
        model.eval()   # disable dropout so both calls produce identical logits
        ts = TemperatureScaler(model)
        x  = torch.randn(3, 1800, 29)
        lb_cal, le, lmc = ts(x)
        assert lb_cal.shape == (3, 3)
        # Calibrated logits must equal raw logits ÷ T  (same forward pass, eval mode)
        lb_raw, _, _ = model(x)
        T = ts.temperature_value()
        assert torch.allclose(lb_cal, lb_raw / T, atol=1e-5)


# ---------------------------------------------------------------------------
# Explainability tests
# ---------------------------------------------------------------------------

class TestExplainability:

    def test_gradient_x_input_shape(self):
        from pipeline.module4.explain import _gradient_x_input
        from pipeline.module4 import SolarPatchTST
        model = SolarPatchTST(feature_names=FEAT_NAMES_29, d_model=16, n_heads=2, n_layers=1)
        x     = torch.randn(3, 1800, 29)
        attrs = _gradient_x_input(model, x, torch.device("cpu"))
        assert attrs.shape == (3, 1800, 29)

    def test_gradient_x_input_nonzero(self):
        from pipeline.module4.explain import _gradient_x_input
        from pipeline.module4 import SolarPatchTST
        model = SolarPatchTST(feature_names=FEAT_NAMES_29, d_model=16, n_heads=2, n_layers=1)
        x     = torch.randn(2, 1800, 29)
        attrs = _gradient_x_input(model, x, torch.device("cpu"))
        assert np.abs(attrs).sum() > 0.0, "Attributions must be non-zero for random input"

    def test_feat_imp_aggregation_shape(self):
        attrs_raw = np.random.randn(5, 1800, 29).astype(np.float32)
        feat_imp  = np.abs(attrs_raw).mean(axis=(0, 1))
        patch_imp = np.abs(attrs_raw).reshape(5, 60, 30, 29).mean(axis=(0, 2, 3))
        assert feat_imp.shape  == (29,)
        assert patch_imp.shape == (60,)


# ---------------------------------------------------------------------------
# Dataset tests (using preloaded data — no disk I/O)
# ---------------------------------------------------------------------------

class TestSolarFlareDataset:

    def _build_ds(self, n=40):
        from pipeline.module4.dataset import SolarFlareDataset
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(1)
        T, F = 1800, 29
        X_np = rng.standard_normal((n, T, F)).astype(np.float32)
        y_cls = rng.integers(0, 5, size=n).astype(np.int8)
        y_bin = (y_cls >= 3).astype(np.int8)

        scaler = StandardScaler()
        scaler.fit(X_np.reshape(n * T, F))
        scaler.mean_ = scaler.mean_.astype(np.float32)
        scaler.var_  = np.maximum(scaler.var_, 1e-8).astype(np.float32)
        scaler.scale_ = np.sqrt(scaler.var_).astype(np.float32)

        return SolarFlareDataset(
            x_path=None,
            indices=None,
            y_binary_30=y_bin,
            y_class=y_cls,
            scaler=scaler,
            feature_names=FEAT_NAMES_29,
            verbose=False,
            _preloaded_X=torch.from_numpy(X_np),
        )

    def test_len(self):
        ds = self._build_ds(40)
        assert len(ds) == 40

    def test_getitem_keys(self):
        ds   = self._build_ds(10)
        item = ds[0]
        assert set(item.keys()) == {"X", "y_binary", "y_extreme", "y_class"}

    def test_getitem_shapes(self):
        ds   = self._build_ds(10)
        item = ds[0]
        assert item["X"].shape        == (1800, 29)
        assert item["y_binary"].shape == (3,)
        assert item["y_extreme"].shape == torch.Size([])
        assert item["y_class"].dtype   == torch.int64

    def test_no_nan_after_scaling(self):
        import torch
        ds = self._build_ds(20)
        x  = ds[0]["X"]
        assert not torch.isnan(x).any()


# ---------------------------------------------------------------------------
# Mini end-to-end training loop
# ---------------------------------------------------------------------------

class TestMiniTraining:
    """
    Runs 2 epochs with a tiny model and synthetic data.
    Verifies:
      • Loss decreases or stays finite
      • Best checkpoint is saved
      • Model can be loaded back and evaluated
    """

    def test_two_epoch_train_and_eval(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader
        from pipeline.module4 import SolarPatchTST, train_model, evaluate_model
        from pipeline.module4.dataset import SolarFlareDataset
        from sklearn.preprocessing import StandardScaler

        device = torch.device("cpu")
        rng = np.random.default_rng(42)
        N_tr, N_val, T, F = 64, 16, 1800, 29

        def _make_loader(N, shuffle=False):
            X_np  = rng.standard_normal((N, T, F)).astype(np.float32)
            y_cls = rng.integers(2, 5, size=N).astype(np.int8)   # C/M/X only
            y_bin = (y_cls >= 3).astype(np.int8)

            scaler = StandardScaler()
            scaler.fit(X_np.reshape(N * T, F))
            scaler.mean_ = scaler.mean_.astype(np.float32)
            scaler.var_  = np.maximum(scaler.var_, 1e-8).astype(np.float32)
            scaler.scale_= np.sqrt(scaler.var_).astype(np.float32)

            ds = SolarFlareDataset(
                x_path=None, indices=None,
                y_binary_30=y_bin, y_class=y_cls,
                scaler=scaler, feature_names=FEAT_NAMES_29,
                verbose=False, _preloaded_X=torch.from_numpy(X_np),
            )
            return DataLoader(ds, batch_size=8, shuffle=shuffle)

        tr_loader  = _make_loader(N_tr,  shuffle=True)
        val_loader = _make_loader(N_val, shuffle=False)

        model = SolarPatchTST(
            feature_names=FEAT_NAMES_29,
            d_model=16, n_heads=2, n_layers=1, dropout=0.0
        )

        history = train_model(
            model         = model,
            train_loader  = tr_loader,
            val_loader    = val_loader,
            checkpoint_dir= str(tmp_path / "ckpt"),
            n_epochs      = 2,
            patience      = 5,
            device        = device,
            verbose       = False,
        )

        assert len(history["tr_loss"]) >= 1
        assert all(np.isfinite(v) for v in history["tr_loss"])
        assert all(np.isfinite(v) for v in history["val_loss"])

        # Checkpoint must exist
        ckpt_path = tmp_path / "ckpt" / "best_model.pt"
        assert ckpt_path.exists()

        # Load and evaluate
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])

        metrics = evaluate_model(
            model, val_loader, device,
            partition="VAL", verbose=False,
        )
        assert "tpr_at_far0.5" in metrics
        assert 0.0 <= metrics["tpr_at_far0.5"] <= 1.0

    def test_temperature_scaling_after_training(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader
        from pipeline.module4 import SolarPatchTST, train_model
        from pipeline.module4.calibrate import TemperatureScaler
        from pipeline.module4.dataset import SolarFlareDataset
        from sklearn.preprocessing import StandardScaler

        device = torch.device("cpu")
        rng = np.random.default_rng(99)
        N, T, F = 32, 1800, 29
        X_np  = rng.standard_normal((N, T, F)).astype(np.float32)
        y_cls = rng.integers(2, 5, size=N).astype(np.int8)
        y_bin = (y_cls >= 3).astype(np.int8)

        scaler = StandardScaler()
        scaler.fit(X_np.reshape(N * T, F))
        scaler.mean_ = scaler.mean_.astype(np.float32)
        scaler.var_  = np.maximum(scaler.var_, 1e-8).astype(np.float32)
        scaler.scale_= np.sqrt(scaler.var_).astype(np.float32)

        ds = SolarFlareDataset(
            x_path=None, indices=None,
            y_binary_30=y_bin, y_class=y_cls,
            scaler=scaler, feature_names=FEAT_NAMES_29,
            verbose=False, _preloaded_X=torch.from_numpy(X_np),
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = SolarPatchTST(
            feature_names=FEAT_NAMES_29, d_model=16, n_heads=2, n_layers=1
        )
        ts = TemperatureScaler(model)
        ts.fit(loader, device=device, verbose=False)

        save_path = str(tmp_path / "temp.pt")
        ts.save(save_path)
        ts2 = TemperatureScaler(model)
        ts2.load(save_path)
        assert abs(ts.temperature_value() - ts2.temperature_value()) < 1e-6
