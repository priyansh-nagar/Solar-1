"""
rebuild_windows_v2.py — Rebuild WindowDataset with correct GOES labels
=======================================================================
Mandatory before Module 4 training.  The v1 WindowDataset (X_all.npy /
y_class_all.npy) used the wrong excess_A proxy thresholds for classification
(C=5.0, M=15.0, X=50.0 — first-principles values that are 15-100x too high)
and had X=0 in y_class because the detector saturated on the X4.5 event.

This script:
  1. Runs Modules 1+2 on each day folder
  2. Adds per-cadence GOES labels via build_goes_labels() (downloads NOAA NGDC
     data, or falls back to empirical excess_A thresholds if offline)
  3. Rebuilds sliding windows with label_source="goes_class"
  4. Saves X_v2.npy / y_binary_v2.npy / y_class_v2.npy to /tmp/
  5. Prints the new class breakdown — X must no longer be 0

HOW TO RUN
----------
With real PRADAN data:

    python pipeline/scripts/rebuild_windows_v2.py --data-dir /tmp/solexs_data

With synthetic data (CI / demo — no FITS files needed):

    python pipeline/scripts/rebuild_windows_v2.py --synthetic

PATCH SIZE DECISION (recorded here per ISRO presentation requirements)
-----------------------------------------------------------------------
patch_size = 30 samples = 30 seconds

Rationale:
  • Window = 1800 s → 30 s patches → 60 patches per window
  • Solar flare onsets evolve over 1-3 minutes (60-180 s).
    30 s patches capture the rise shape: each patch sees the derivative
    within a 30 s interval, and cross-attention between adjacent patches
    reconstructs the rise trajectory.
  • 15 s patches (120 patches): too fine — self-attention on 120 tokens
    per channel is expensive and catches Poisson noise spikes.
  • 60 s patches (30 patches): too coarse — a fast M-class onset (60-90 s
    rise time) falls entirely within one patch; the model can't distinguish
    a fast onset from a slow one.
  • 30 s is the PatchTST paper's recommended patch size for 1-s-cadence
    signals with feature periods of 1-5 minutes.

This decision is documented here and defended in the ISRO presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.module1            import ingest_day
from pipeline.module2.preprocess import preprocess_day
from pipeline.module2.windows    import build_windows, WINDOW_S, STRIDE_S, LABEL_HORIZON_S
from pipeline.module3.goes_crossmatch import build_goes_labels


# ---------------------------------------------------------------------------
# Real-data day configs — edit paths when running on real PRADAN data
# ---------------------------------------------------------------------------
REAL_DAY_CONFIGS = [
    # (path, date_for_GOES_download or None)
    ("AL1_SLX_L1_20240209_v1.0", datetime(2024,  2,  9)),
    ("AL1_SLX_L1_20240222_v1.0", datetime(2024,  2, 22)),
    ("AL1_SLX_L1_20240506_v1.0", datetime(2024,  5,  6)),
    ("AL1_SLX_L1_20240510_v1.0", datetime(2024,  5, 10)),
    ("AL1_SLX_L1_20241003_v1.0", datetime(2024, 10,  3)),
    ("AL1_SLX_L1_20260612_v1.0", None),    # quiet day — no GOES needed
]


# ---------------------------------------------------------------------------
# Single-day processor
# ---------------------------------------------------------------------------

def process_day(
    day_dir: Path,
    date: datetime | None,
    label: str,
) -> dict | None:
    """
    Run Modules 1+2 → GOES labels → build_windows(label_source='goes_class').

    Returns dict with X, y_binary, y_class arrays plus stats, or None on failure.
    """
    t0 = time.perf_counter()
    print(f"  {label} …", end=" ", flush=True)

    # ── Module 1 ─────────────────────────────────────────────────────────
    try:
        ds = ingest_day(solexs_dir=str(day_dir), verbose=False)
    except Exception as exc:
        print(f"✗ ingest: {exc}")
        return None

    usable = ds.attrs.get("usable_fraction", 0.0)
    if usable < 0.05:
        print(f"✗ usable_fraction={usable:.2f} — skipping")
        return None

    # ── Module 2 ─────────────────────────────────────────────────────────
    try:
        ds_pp = preprocess_day(ds, verbose=False)
    except Exception as exc:
        print(f"✗ preprocess: {exc}")
        return None

    # ── GOES labels ───────────────────────────────────────────────────────
    try:
        ds_pp = build_goes_labels(ds_pp, date=date)
        source = ds_pp.attrs.get("goes_label_source", "?")
    except Exception as exc:
        print(f"✗ goes_labels: {exc}")
        return None

    # ── Extract feature dict from Dataset ─────────────────────────────────
    # Include goes_class as a feature so build_windows can strip & use it as label
    prefix = "solexs_sdd2"
    feat_dict = {}
    for var in ds_pp.data_vars:
        if var == "split":
            continue
        if "spectrum" in var:
            continue
        if var == "goes_class":
            # Include goes_class so windows.py can use it as label source
            feat_dict["goes_class"] = ds_pp["goes_class"].values.astype(float)
            continue
        if var.startswith(prefix) and "quality" not in var:
            key = var.removeprefix(f"{prefix}_")
            feat_dict[key] = ds_pp[var].values.astype(float)

    split_arr = ds_pp["split"].values

    # ── Build windows ─────────────────────────────────────────────────────
    try:
        wd = build_windows(
            features        = feat_dict,
            split_array     = split_arr,
            label_source    = "goes_class",   # THE KEY CHANGE
            window_s        = WINDOW_S,
            stride_s        = STRIDE_S,
            label_horizon_s = LABEL_HORIZON_S,
            cadence_s       = float(ds.attrs.get("cadence_s", 1.0)),
        )
    except Exception as exc:
        print(f"✗ build_windows: {exc}")
        return None

    elapsed = time.perf_counter() - t0
    n_flare = int(wd.y_binary.sum())
    n_total = len(wd.y_binary)
    cls_counts = {
        cls: int((wd.y_class == i).sum())
        for i, cls in enumerate(["A", "B", "C", "M", "X"])
        if int((wd.y_class == i).sum()) > 0
    }

    print(
        f"✓  {n_total} windows  flare%={100*n_flare/max(n_total,1):.1f}%  "
        f"classes={cls_counts}  source={source}  [{elapsed:.1f}s]"
    )

    return {
        "X":        wd.X,
        "y_binary": wd.y_binary,
        "y_class":  wd.y_class,
        "n_total":  n_total,
        "n_flare":  n_flare,
        "label":    label,
        "source":   source,
    }


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

def run_synthetic() -> None:
    """Run rebuild on synthetic data (no FITS files needed)."""
    import tempfile
    from pipeline.scripts.make_synthetic_day import make_day

    print("=" * 65)
    print("  SYNTHETIC DATA MODE — rebuild_windows_v2")
    print("=" * 65)
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Two flare days + one quiet day
        configs = [
            (make_day(tmp, "20240222", n_hours=4, seed=42), datetime(2024, 2, 22), "2024-02-22 (X+M+C)"),
            (make_day(tmp, "20240506", n_hours=4, seed=99), datetime(2024, 5,  6), "2024-05-06 (M+C)"),
            (_make_quiet_day(tmp, "20260612"),               None,                  "2026-06-12 (quiet)"),
        ]

        print("Processing days:")
        results = []
        for day_dir, date, label in configs:
            r = process_day(day_dir, date, label)
            if r is not None:
                results.append(r)

        _finish(results)


def _make_quiet_day(out_dir: Path, date_tag: str) -> Path:
    """Synthetic quiet day — background only, no flares."""
    from pipeline.scripts.make_synthetic_day import (
        make_spectrum, build_pi_fits, build_gti_fits, write_fits_gz,
    )
    n_hours = 4
    n_times = n_hours * 3600
    t0 = 1749686400.0
    tstart = t0 + np.arange(n_times, dtype=np.float64)
    counts = make_spectrum(tstart, bg_band_a=30.0, flares=None, seed=7)
    day_name = f"AL1_SLX_L1_{date_tag}_v1.0"
    sdd2_dir = out_dir / day_name / "SDD2"
    sdd2_dir.mkdir(parents=True, exist_ok=True)
    base = f"AL1_SLX_SDD2_L1_{date_tag}"
    write_fits_gz(build_pi_fits(tstart, counts),             sdd2_dir / f"{base}.pi.gz")
    write_fits_gz(build_gti_fits([(tstart[0], tstart[-1])]), sdd2_dir / f"{base}.gti.gz")
    return out_dir / day_name


# ---------------------------------------------------------------------------
# Real-data path
# ---------------------------------------------------------------------------

def run_real(data_dir: Path) -> None:
    """Run rebuild on real PRADAN FITS data."""
    print("=" * 65)
    print(f"  REAL DATA MODE — {data_dir}")
    print("=" * 65)
    print()

    results = []
    for folder_name, date in REAL_DAY_CONFIGS:
        day_dir = data_dir / folder_name
        if not day_dir.exists():
            print(f"  {folder_name}: directory not found — skipping")
            continue
        label = folder_name.split("_")[3]   # extract YYYYMMDD
        r = process_day(day_dir, date, label)
        if r is not None:
            results.append(r)

    _finish(results)


# ---------------------------------------------------------------------------
# Shared finish
# ---------------------------------------------------------------------------

def _finish(results: list) -> None:
    if not results:
        print("\n  ✗ No days processed successfully.")
        sys.exit(1)

    X_all       = np.concatenate([r["X"]        for r in results], axis=0).astype(np.float32)
    y_binary    = np.concatenate([r["y_binary"]  for r in results], axis=0)
    y_class     = np.concatenate([r["y_class"]   for r in results], axis=0)

    n_total = len(X_all)
    n_flare = int(y_binary.sum())

    # Per-class counts
    cls_names = ["A", "B", "C", "M", "X"]
    counts    = [int((y_class == i).sum()) for i in range(5)]

    print()
    print("=" * 65)
    print("  NEW DATASET (v2) — CLASS BREAKDOWN")
    print("=" * 65)
    print(f"  Total windows : {n_total:,}")
    print(f"  Feature shape : {X_all.shape}")
    print()
    print(f"  {'Class':<6} {'Windows':>10} {'%':>8}")
    print("  " + "─" * 28)
    for cls, cnt in zip(cls_names, counts):
        bar = "█" * int(20 * cnt / max(n_total, 1))
        print(f"  {cls:<6} {cnt:>10,} {100*cnt/max(n_total,1):>7.2f}%  {bar}")
    print("  " + "─" * 28)
    print(f"  {'Total':<6} {n_total:>10,}  100.00%")
    print()

    flare_pct = 100 * n_flare / max(n_total, 1)
    imbalance = (n_total - n_flare) / max(n_flare, 1)
    print(f"  Flare windows (C+) : {n_flare:,}  ({flare_pct:.2f}%)")
    print(f"  Imbalance ratio    : {imbalance:.1f}× (quiet/flare)")
    print()

    # Key check: X-class must not be 0
    x_count = counts[4]
    if x_count > 0:
        print(f"  ✓ X-class correctly labelled: {x_count} windows (was 0 in v1)")
    else:
        print("  ⚠ X-class = 0.  Check GOES cross-match or flare data availability.")

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path("/tmp")
    np.save(out_dir / "X_v2.npy",        X_all)
    np.save(out_dir / "y_binary_v2.npy", y_binary)
    np.save(out_dir / "y_class_v2.npy",  y_class)

    stats = {
        "n_total":       n_total,
        "shape":         list(X_all.shape),
        "n_flare":       n_flare,
        "flare_pct":     round(flare_pct, 4),
        "imbalance":     round(imbalance, 2),
        "class_counts":  dict(zip(cls_names, counts)),
        "label_source":  "goes_class",
        "patch_size_s":  30,
        "n_patches":     WINDOW_S // 30,
    }

    with open(out_dir / "dataset_v2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved:")
    print(f"    /tmp/X_v2.npy           {X_all.nbytes / 1e6:.1f} MB  shape={X_all.shape}")
    print(f"    /tmp/y_binary_v2.npy    {y_binary.nbytes / 1e3:.1f} KB")
    print(f"    /tmp/y_class_v2.npy     {y_class.nbytes / 1e3:.1f} KB")
    print(f"    /tmp/dataset_v2_stats.json")
    print()
    print("  ── MODULE 4 HANDOFF ─────────────────────────────────────")
    print(f"  X shape        : {X_all.shape}  (windows × time × features)")
    print(f"  Imbalance      : {imbalance:.1f}×  → focal_loss γ=2.0 α=0.80")
    print(f"  Patch size     : 30 s → 60 patches per window (defended)")
    print(f"  Label source   : goes_class (GOES cross-match + empirical fallback)")
    print(f"  StandardScaler : fit on TRAIN partition only")
    print("  ─────────────────────────────────────────────────────────")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild WindowDataset with GOES labels (Module 4 prep)"
    )
    parser.add_argument("--data-dir", default="/tmp/solexs_data")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data (no FITS files required)"
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
    else:
        run_real(Path(args.data_dir))


if __name__ == "__main__":
    main()
