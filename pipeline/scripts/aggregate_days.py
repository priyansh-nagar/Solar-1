"""
aggregate_days.py — run Modules 1+2 over every day folder and report stats
===========================================================================
Usage:

    python pipeline/scripts/aggregate_days.py [--data-dir /tmp/solexs_data]

Produces the 6 numbers needed to configure Modules 3 and 4, then saves:

    /tmp/X_all.npy          (n_windows, 1800, 29) float32
    /tmp/y_binary_all.npy   (n_windows,) int8
    /tmp/y_class_all.npy    (n_windows,) int8
    /tmp/dataset_stats.json  stats dict for reference
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── silence verbose module-level logging during batch runs ────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)

# Pipeline imports — run from workspace root so `pipeline` is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.module1 import ingest_day                          # noqa: E402
from pipeline.module2.preprocess import preprocess_day           # noqa: E402


def process_one_day(day_dir: Path) -> dict | None:
    """
    Run Module 1 + 2 (with windowing) on a single day folder.

    Returns a dict with keys X, y_binary, y_class, stats, or None on failure.
    """
    t0 = time.perf_counter()
    try:
        ds = ingest_day(solexs_dir=str(day_dir), verbose=False)
    except Exception as e:
        print(f"  ✗ ingest_day failed: {e}")
        return None

    usable = ds.attrs.get("usable_fraction", 0.0)
    if usable < 0.10:
        print(f"  ⚠ usable_fraction={usable:.2f} — skipping (< 10% usable data)")
        return None

    try:
        ds_out, wd = preprocess_day(ds, build_windows=True, verbose=False)
    except Exception as e:
        print(f"  ✗ preprocess_day failed: {e}")
        return None

    elapsed = time.perf_counter() - t0
    n_total  = len(wd.y_binary)
    n_flare  = int(wd.y_binary.sum())
    pct      = 100.0 * n_flare / max(n_total, 1)
    cls_counts = list(np.bincount(wd.y_class.astype(np.int64), minlength=5).tolist())

    print(
        f"  ✓ {n_total:>5} windows  "
        f"{n_flare:>4} flares ({pct:4.1f}%)  "
        f"classes={cls_counts}  "
        f"[{elapsed:.0f}s]"
    )

    return {
        "X":        wd.X,
        "y_binary": wd.y_binary,
        "y_class":  wd.y_class,
        "stats": {
            "date":            day_dir.name,
            "n_windows":       n_total,
            "n_flare":         n_flare,
            "flare_pct":       round(pct, 2),
            "class_counts":    cls_counts,
            "usable_fraction": round(float(usable), 4),
            "quiet_day":       bool(ds.attrs.get("quiet_day", True)),
            "peak_flux_ratio": round(float(ds.attrs.get("peak_flux_ratio", 0.0)), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Module 1+2 across all day folders")
    parser.add_argument(
        "--data-dir", default="/tmp/solexs_data",
        help="Directory containing AL1_SLX_L1_YYYYMMDD_v1.0/ subfolders",
    )
    parser.add_argument(
        "--out-dir", default="/tmp",
        help="Where to write X_all.npy / y_binary_all.npy / y_class_all.npy",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        print("Run  python pipeline/scripts/extract_uploads.py  first.")
        sys.exit(1)

    day_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not day_dirs:
        print(f"No day folders found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(day_dirs)} day folder(s) in {data_dir}")
    print()

    all_X:      list[np.ndarray] = []
    all_y_bin:  list[np.ndarray] = []
    all_y_cls:  list[np.ndarray] = []
    per_day_stats: list[dict]    = []
    n_ok = n_fail = 0

    for day_dir in day_dirs:
        print(f"Processing {day_dir.name} ...")
        result = process_one_day(day_dir)
        if result is None:
            n_fail += 1
            continue
        all_X.append(result["X"])
        all_y_bin.append(result["y_binary"])
        all_y_cls.append(result["y_class"])
        per_day_stats.append(result["stats"])
        n_ok += 1

    print()
    if not all_X:
        print(f"No usable days processed ({n_fail} failed).  Exiting.")
        sys.exit(1)

    X       = np.concatenate(all_X,     axis=0).astype(np.float32)
    y_bin   = np.concatenate(all_y_bin, axis=0)
    y_cls   = np.concatenate(all_y_cls, axis=0)

    n_total  = len(y_bin)
    n_flare  = int(y_bin.sum())
    n_quiet  = n_total - n_flare
    imb      = round(n_quiet / max(n_flare, 1), 1)
    flare_pct = round(100.0 * n_flare / max(n_total, 1), 1)
    quiet_pct = round(100.0 * n_quiet / max(n_total, 1), 1)

    # GOES class breakdown: 0=quiet, 1=B, 2=C, 3=M, 4=X
    cls_labels = ["quiet", "B", "C", "M", "X"]
    cls_counts = list(np.bincount(y_cls.astype(np.int64), minlength=5).tolist())
    cls_str = "  ".join(f"{cls_labels[i]}={cls_counts[i]}" for i in range(5))

    print("=" * 60)
    print("=== FINAL DATASET ===")
    print("=" * 60)
    print(f"Total windows  : {n_total}")
    print(f"Flare windows  : {n_flare} ({flare_pct}%)")
    print(f"Quiet windows  : {n_quiet} ({quiet_pct}%)")
    print(f"Imbalance ratio: {imb}x")
    print(f"Class breakdown: [{cls_str}]")
    print(f"X tensor shape : {X.shape}")
    print()

    # ── Focal loss recommendation ─────────────────────────────────────────
    if flare_pct >= 10:
        gamma, alpha = 2.0, 0.75
    elif flare_pct >= 5:
        gamma, alpha = 2.0, 0.80
    elif flare_pct >= 1:
        gamma, alpha = 3.0, 0.85
    else:
        gamma, alpha = 4.0, 0.95

    print(f"Module 4 focal loss recommendation: γ={gamma}  α={alpha}")
    if n_flare < 50:
        print("⚠ WARNING: < 50 flare windows — consider adding more flare days.")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_all.npy",        X)
    np.save(out_dir / "y_binary_all.npy", y_bin)
    np.save(out_dir / "y_class_all.npy",  y_cls)

    stats = {
        "n_days_ok":       n_ok,
        "n_days_failed":   n_fail,
        "n_total":         n_total,
        "n_flare":         n_flare,
        "n_quiet":         n_quiet,
        "flare_pct":       flare_pct,
        "imbalance_ratio": imb,
        "class_counts":    {cls_labels[i]: cls_counts[i] for i in range(5)},
        "X_shape":         list(X.shape),
        "focal_gamma":     gamma,
        "focal_alpha":     alpha,
        "per_day":         per_day_stats,
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {out_dir}/")
    print(f"  X_all.npy          {X.nbytes / 1_073_741_824:.2f} GB")
    print(f"  y_binary_all.npy")
    print(f"  y_class_all.npy")
    print(f"  dataset_stats.json")


if __name__ == "__main__":
    main()
