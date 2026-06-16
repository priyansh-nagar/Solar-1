"""
train_module4.py — End-to-end training pipeline for Module 4 (PatchTST)
========================================================================
Steps:
  1. Load v2 dataset from /tmp/ (run rebuild_windows_v2.py first)
  2. Fit StandardScaler on TRAIN partition (no leakage)
  3. Instantiate SolarPatchTST + train with early stopping on TPR@FAR0.5
  4. Evaluate on VAL (⚠  optimistic) and TEST partitions
  5. Apply temperature scaling for calibration
  6. Compute SHAP attributions on 128 TEST windows
  7. Save checkpoint + artefacts to /tmp/module4_ckpt/

Usage
-----
    python pipeline/scripts/train_module4.py [options]

    --data-dir   /tmp            where X_v2.npy etc. live
    --ckpt-dir   /tmp/module4_ckpt
    --epochs     30
    --batch      32
    --d-model    64
    --n-heads    4
    --n-layers   2
    --lr         1e-4
    --dropout    0.2
    --patience   7
    --no-shap    skip SHAP computation (faster)
    --device     cpu / cuda
    --synthetic  generate and use synthetic data (no FITS/npy files required)
                 useful for smoke-testing the full pipeline in CI / demo environments
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure the workspace root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SolarPatchTST (Module 4)"
    )
    p.add_argument("--data-dir",   default="/tmp",                    type=str)
    p.add_argument("--ckpt-dir",   default="/tmp/module4_ckpt",       type=str)
    p.add_argument("--epochs",     default=30,   type=int)
    p.add_argument("--batch",      default=32,   type=int)
    p.add_argument("--d-model",    default=64,   type=int)
    p.add_argument("--n-heads",    default=4,    type=int)
    p.add_argument("--n-layers",   default=2,    type=int)
    p.add_argument("--lr",         default=1e-4, type=float)
    p.add_argument("--dropout",    default=0.2,  type=float)
    p.add_argument("--patience",   default=7,    type=int)
    p.add_argument("--no-shap",         action="store_true")
    p.add_argument("--device",          default="auto",        type=str)
    p.add_argument("--horizon",         default=30,  type=int,
                   choices=[15, 30, 60],
                   help="Primary forecast horizon in minutes (default 30). "
                        "Use 60 to teach the model to fire earlier.")
    p.add_argument("--label-smoothing", default=0.0, type=float,
                   help="Label smoothing ε for binary/extreme BCE losses "
                        "(0=off, 0.1 recommended). Reduces overconfidence and ECE.")
    p.add_argument("--calibration",     default="temperature",
                   choices=["temperature", "isotonic"],
                   help="Post-hoc calibration method (default temperature). "
                        "Use isotonic to fix prior mismatch between TRAIN/TEST.")
    p.add_argument(
        "--synthetic", action="store_true",
        help=(
            "Generate synthetic data and run a reduced training pipeline. "
            "No .npy files or FITS data required. "
            "Uses d_model=32, n_layers=1 and epochs=5 unless overridden. "
            "Useful for CI, smoke-testing, or demo environments."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic dataset builder (no disk I/O except saving the final arrays)
# ---------------------------------------------------------------------------

_SYNTH_FEAT_NAMES = [
    "band_A", "background_A", "flux_smooth_A", "excess_A", "residual_A",
    "derivative_1s", "derivative_60s", "rate_of_rise",
    "rolling_std_5min", "rolling_std_15min", "cumulative_excess",
    "hardness_ratio", "softness_ratio", "total",
    "band_B", "background_B", "residual_B", "excess_B",
    "band_C", "background_C", "residual_C", "excess_C",
    "band_D", "background_D", "residual_D", "excess_D",
    "flux_smooth_B", "flux_smooth_C", "flux_smooth_D",
]
assert len(_SYNTH_FEAT_NAMES) == 29


def _make_synthetic_dataset(
    out_dir: Path,
    n_train: int = 240,
    n_val:   int = 60,
    n_test:  int = 60,
    seed:    int = 42,
) -> None:
    """
    Generate synthetic v2-compatible arrays and save them to *out_dir*.

    The synthetic signal has:
      • Random Gaussian background for all features
      • Flare precursor injected in the final 5 patches of excess_A / band_C
        for M+/X-class labelled windows (so the model has a learnable signal)

    This is NOT real solar data — it only validates that the training loop
    runs correctly end-to-end.
    """
    from pipeline.module2.split import TRAIN, VAL, TEST

    rng = np.random.default_rng(seed)
    T, F = 1800, 29
    n_total = n_train + n_val + n_test
    print(f"  Generating {n_total} synthetic windows ({T}×{F}) …")

    X = rng.standard_normal((n_total, T, F)).astype(np.float32) * 0.5

    # Class distribution: roughly 40% M+, 5% X
    probs = [0.02, 0.03, 0.55, 0.30, 0.10]
    y_class = rng.choice(5, size=n_total, p=probs).astype(np.int8)
    y_binary = (y_class >= 3).astype(np.int8)

    # Inject a precursor signal in excess_A (col 3) and band_C (col 18)
    # for M+ windows: rising ramp in the last 150 s (patches 55–59)
    excess_a_col = _SYNTH_FEAT_NAMES.index("excess_A")
    band_c_col   = _SYNTH_FEAT_NAMES.index("band_C")
    ramp = np.linspace(0, 3.0, 150).astype(np.float32)
    for i in np.where(y_binary == 1)[0]:
        X[i, -150:, excess_a_col] += ramp * (1.0 + rng.standard_normal() * 0.1)
        X[i, -150:, band_c_col]   += ramp * 0.5

    # Day-level splits (simple contiguous assignment)
    splits = np.concatenate([
        np.full(n_train, TRAIN, dtype=np.uint8),
        np.full(n_val,   VAL,   dtype=np.uint8),
        np.full(n_test,  TEST,  dtype=np.uint8),
    ])

    y_extreme = (y_class >= 4).astype(np.int8)
    y_15min   = y_binary.copy()
    y_60min   = y_binary.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_v2.npy",         X)
    np.save(out_dir / "y_binary_v2.npy",  y_binary)
    np.save(out_dir / "y_class_v2.npy",   y_class)
    np.save(out_dir / "splits_v2.npy",    splits)
    np.save(out_dir / "y_extreme_v2.npy", y_extreme)
    np.save(out_dir / "y_15min_v2.npy",   y_15min)
    np.save(out_dir / "y_60min_v2.npy",   y_60min)

    import json
    with open(out_dir / "feature_names_v2.json", "w") as f:
        json.dump(_SYNTH_FEAT_NAMES, f)

    m_plus = int(y_binary.sum())
    x_cls  = int((y_class == 4).sum())
    print(
        f"  Saved to {out_dir}:  "
        f"{n_total} windows  M+={m_plus} ({100*m_plus/n_total:.0f}%)  "
        f"X-class={x_cls}"
    )


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Synthetic mode: override hyper-params and generate data ──────────────
    if args.synthetic:
        import tempfile
        _synth_tmp = tempfile.mkdtemp(prefix="m4_synth_")
        args.data_dir = _synth_tmp
        # Use smaller model/training for speed unless user explicitly set them
        if args.d_model  == 64: args.d_model  = 32
        if args.n_layers == 2:  args.n_layers = 1
        if args.epochs   == 30: args.epochs   = 5
        if args.patience == 7:  args.patience = 3

    print(f"\n{'='*60}")
    print(f"  SoLEXS Module 4 — PatchTST Flare Forecaster")
    if args.synthetic:
        print(f"  ⚠  SYNTHETIC DATA MODE — for smoke-testing only")
    print(f"{'='*60}")
    # Derive horizon_idx from --horizon flag: 15→0, 30→1, 60→2
    horizon_idx = {15: 0, 30: 1, 60: 2}[args.horizon]

    print(f"  Device      : {device}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Checkpoint  : {args.ckpt_dir}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    print(f"  lr={args.lr}  batch={args.batch}  patience={args.patience}")
    print(f"  horizon={args.horizon}min  label_smoothing={args.label_smoothing}  calibration={args.calibration}")
    print()

    # ── Generate synthetic data if requested ─────────────────────────────────
    if args.synthetic:
        print("Generating synthetic dataset …")
        _make_synthetic_dataset(Path(args.data_dir))
        print()

    # ── Load data ────────────────────────────────────────────────────────────
    from pipeline.module4.dataset import load_splits

    print("Loading v2 dataset …")
    tr_loader, val_loader, te_loader, scaler, feature_names = load_splits(
        data_dir=args.data_dir,
        batch_size=args.batch,
    )

    n_features = len(feature_names)
    print(f"  Features ({n_features}): {feature_names[:5]} … {feature_names[-3:]}")
    print(f"  TRAIN batches: {len(tr_loader)}  "
          f"VAL batches: {len(val_loader)}  "
          f"TEST batches: {len(te_loader)}\n")

    # ── Build model ──────────────────────────────────────────────────────────
    from pipeline.module4.model import SolarPatchTST

    model = SolarPatchTST(
        feature_names = feature_names,
        d_model       = args.d_model,
        n_heads       = args.n_heads,
        n_layers      = args.n_layers,
        dropout       = args.dropout,
    )
    n_params = model.n_parameters()
    soft_idx = model.soft_idx.tolist()
    hard_idx = model.hard_idx.tolist()
    print(f"Model: {n_params:,} parameters")
    print(f"  Soft stream: {len(soft_idx)} features")
    print(f"  Hard stream: {len(hard_idx)} features")
    print(f"  Soft → {[feature_names[i] for i in soft_idx]}")
    print(f"  Hard → {[feature_names[i] for i in hard_idx]}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    from pipeline.module4.train import train_model

    history = train_model(
        model                = model,
        train_loader         = tr_loader,
        val_loader           = val_loader,
        checkpoint_dir       = args.ckpt_dir,
        n_epochs             = args.epochs,
        lr                   = args.lr,
        patience             = args.patience,
        device               = device,
        verbose              = True,
        label_smoothing      = args.label_smoothing,
        primary_horizon_idx  = horizon_idx,
    )

    # Save history
    ckpt_dir = Path(args.ckpt_dir)
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("History saved.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    from pipeline.module4.evaluate import evaluate_model

    model = model.to(device)
    val_metrics = evaluate_model(
        model, val_loader, device,
        partition="VAL", verbose=True, val_caveat=True,
        horizon_idx=horizon_idx,
    )
    te_metrics = evaluate_model(
        model, te_loader, device,
        partition="TEST", verbose=True,
        horizon_idx=horizon_idx,
    )

    all_metrics = {"val": val_metrics, "test": te_metrics}
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved.")

    # ── Calibration ──────────────────────────────────────────────────────────
    from pipeline.module4.evaluate import ece as compute_ece

    if args.calibration == "temperature":
        print("\nFitting temperature scaler on VAL …")
        from pipeline.module4.calibrate import TemperatureScaler

        cal = TemperatureScaler(model)
        cal.fit(val_loader, device, verbose=True, horizon_idx=horizon_idx)
        cal_path = str(ckpt_dir / "temperature.pt")
        cal.save(cal_path)
        print(f"Temperature = {cal.temperature_value():.4f}  saved to {cal_path}")
        cal_preds = cal.predict_30min(te_loader, device, horizon_idx=horizon_idx)

    else:  # isotonic
        print("\nFitting isotonic calibrator on VAL …")
        from pipeline.module4.calibrate import IsotonicCalibrator

        cal = IsotonicCalibrator(model, horizon_idx=horizon_idx)
        cal.fit(val_loader, device, verbose=True)
        cal_path = str(ckpt_dir / "isotonic_calibrator.pkl")
        cal.save(cal_path)
        print(f"Isotonic calibrator saved to {cal_path}")
        cal_preds = cal.predict_loader(te_loader, device)

    ece_after = compute_ece(
        cal_preds["prob_30_calibrated"],
        cal_preds["y_true_30"],
    )
    print(f"TEST ECE after calibration: {ece_after:.4f}")
    te_metrics["ece_30min_calibrated"] = round(float(ece_after), 4)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if not args.no_shap:
        print("\nComputing SHAP attributions (128 TEST samples) …")
        from pipeline.module4.explain import SHAPExplainer

        try:
            explainer = SHAPExplainer(model, tr_loader, device, n_bg=64)
            attrs     = explainer.explain(te_loader, n_samples=128, verbose=True)

            feat_imp_dict = {
                feature_names[i]: round(float(attrs["feat_imp"][i]), 6)
                for i in range(len(feature_names))
            }
            with open(ckpt_dir / "feature_importance.json", "w") as f:
                json.dump(feat_imp_dict, f, indent=2)

            explainer.plot_feature_importance(
                attrs, feature_names,
                save_path=str(ckpt_dir / "shap_feature_importance.png"),
            )
            explainer.plot_patch_timeline(
                attrs,
                save_path=str(ckpt_dir / "shap_patch_timeline.png"),
            )
            print("SHAP artefacts saved.")
        except Exception as exc:
            print(f"SHAP failed: {exc}")
    else:
        print("Skipping SHAP (--no-shap).")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Module 4 Training Complete")
    print(f"{'='*60}")
    print(f"  VAL  TPR@FAR0.5 : {val_metrics.get('tpr_at_far0.5', '?')}")
    print(f"  TEST TPR@FAR0.5 : {te_metrics.get('tpr_at_far0.5', '?')}")
    print(f"  TEST ROC-AUC    : {te_metrics.get('roc_auc_30min', '?')}")
    print(f"  TEST ECE (raw)  : {te_metrics.get('ece_30min', '?')}")
    print(f"  TEST ECE (cal)  : {te_metrics.get('ece_30min_calibrated', '?')}")
    print(f"  Lead time (med) : {te_metrics.get('lead_median_min', '?')} min")
    print(f"  Checkpoint dir  : {args.ckpt_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
