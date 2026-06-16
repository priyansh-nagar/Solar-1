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
    p.add_argument("--data-dir",  default="/tmp",                     type=str)
    p.add_argument("--ckpt-dir",  default="/tmp/module4_ckpt",        type=str)
    p.add_argument("--epochs",    default=30,   type=int)
    p.add_argument("--batch",     default=32,   type=int)
    p.add_argument("--d-model",   default=64,   type=int)
    p.add_argument("--n-heads",   default=4,    type=int)
    p.add_argument("--n-layers",  default=2,    type=int)
    p.add_argument("--lr",        default=1e-4, type=float)
    p.add_argument("--dropout",   default=0.2,  type=float)
    p.add_argument("--patience",  default=7,    type=int)
    p.add_argument("--no-shap",   action="store_true")
    p.add_argument("--device",    default="auto", type=str)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  SoLEXS Module 4 — PatchTST Flare Forecaster")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Checkpoint  : {args.ckpt_dir}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    print(f"  lr={args.lr}  batch={args.batch}  patience={args.patience}")
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
        model         = model,
        train_loader  = tr_loader,
        val_loader    = val_loader,
        checkpoint_dir= args.ckpt_dir,
        n_epochs      = args.epochs,
        lr            = args.lr,
        patience      = args.patience,
        device        = device,
        verbose       = True,
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
    )
    te_metrics = evaluate_model(
        model, te_loader, device,
        partition="TEST", verbose=True,
    )

    all_metrics = {"val": val_metrics, "test": te_metrics}
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved.")

    # ── Temperature scaling ──────────────────────────────────────────────────
    print("\nFitting temperature scaler on VAL …")
    from pipeline.module4.calibrate import TemperatureScaler

    ts = TemperatureScaler(model).fit(val_loader, device, verbose=True)
    ts_path = str(ckpt_dir / "temperature.pt")
    ts.save(ts_path)
    print(f"Temperature = {ts.temperature_value():.4f}  saved to {ts_path}")

    # Post-calibration ECE on TEST
    from pipeline.module4.evaluate import run_inference, ece as compute_ece
    from pipeline.module4.model    import SolarPatchTST as _M

    cal_preds = ts.predict_30min(te_loader, device)
    ece_after = compute_ece(
        cal_preds["prob_30_calibrated"],
        cal_preds["y_true_30"],
    )
    print(f"TEST ECE after calibration: {ece_after:.4f}")
    te_metrics["ece_30min_calibrated"] = round(ece_after, 4)

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
