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
  4. Applies DAY-LEVEL split assignment (not within-day 70/15/15)
  5. Recasts y_binary as M+ (y_class >= 3) — C+ threshold was degenerate
     (98.4% positive on these peak Solar Cycle 25 days)
  6. Saves X_v2.npy / y_binary_v2.npy / y_class_v2.npy / splits_v2.npy
  7. Prints per-partition class breakdown

DAY-LEVEL SPLIT RATIONALE
--------------------------
The original within-day 70/15/15 split put X-class events into TRAIN only
(all X events happened early UTC, falling in the first 70% of each day).
VAL and TEST had 0 X-class windows — the model could not be evaluated on
extreme events.

Day-level assignment fixes this:
  TRAIN: 2024-02-22 (X6.4), 2024-10-03 (M9), 2024-05-06 (X2.7), 2026-06-12
  VAL:   2024-05-10  (active, but contained — no X-class)
  TEST:  2024-02-09  (X3.4 — model sees X-class at evaluation time)

No 30-minute leakage gap is required between days (days are weeks apart).

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
from pipeline.module2.split      import TRAIN, VAL, TEST
from pipeline.module3.goes_crossmatch import build_goes_labels


# ---------------------------------------------------------------------------
# Real-data day configs and DAY-LEVEL split assignment
# ---------------------------------------------------------------------------

REAL_DAY_CONFIGS = [
    # (folder_name, date_for_GOES_download_or_None, day_split_partition)
    ("AL1_SLX_L1_20240209_v1.0", datetime(2024,  2,  9), TEST),   # X3.4 — held-out test
    ("AL1_SLX_L1_20240222_v1.0", datetime(2024,  2, 22), TRAIN),  # X6.4 — strongest signal
    ("AL1_SLX_L1_20240506_v1.0", datetime(2024,  5,  6), TRAIN),  # X2.7
    ("AL1_SLX_L1_20240510_v1.0", datetime(2024,  5, 10), VAL),    # moderate — validation
    ("AL1_SLX_L1_20241003_v1.0", datetime(2024, 10,  3), TRAIN),  # M9 — second strongest
    ("AL1_SLX_L1_20260612_v1.0", None,                   TRAIN),  # quiet — negative examples
]

# M+ binary threshold: only M-class and above trigger a "flare alert"
# C+ was degenerate (98.4% positive) because GOES background in Solar Cycle 25
# peak stays at C-level even during inter-flare periods on these active days.
BINARY_MIN_CLASS = 3   # 0=A, 1=B, 2=C, 3=M, 4=X  → M+ = flare


# ---------------------------------------------------------------------------
# Multi-horizon label helper
# ---------------------------------------------------------------------------

def _horizon_binary(
    goes_class:    np.ndarray,
    window_starts: np.ndarray,
    window_s:      int,
    horizon_s:     int,
    min_class:     int,
    n_cad:         int,
) -> np.ndarray:
    """
    Compute a binary M+/X label for each window using a different look-ahead horizon.

    For window starting at cadence w, the label = 1 iff
        max( goes_class[ w+window_s : w+window_s+horizon_s ] ) >= min_class

    Parameters
    ----------
    goes_class    : (N_cadences,) per-cadence GOES class integer
    window_starts : (n_windows,) start cadence of each window
    window_s      : window length in cadences (1800)
    horizon_s     : look-ahead horizon in cadences (900 / 1800 / 3600)
    min_class     : minimum GOES class integer for a positive label (3=M, 4=X)
    n_cad         : total cadence count (len(goes_class))
    """
    labels = np.zeros(len(window_starts), dtype=np.int8)
    for i, w in enumerate(window_starts):
        s = int(w) + window_s
        e = min(s + horizon_s, n_cad)
        if e > s:
            labels[i] = int(np.nanmax(goes_class[s:e]) >= min_class)
    return labels


# ---------------------------------------------------------------------------
# Single-day processor
# ---------------------------------------------------------------------------

def process_day(
    day_dir: Path,
    date: datetime | None,
    label: str,
    assigned_split: int,
) -> dict | None:
    """
    Run Modules 1+2 → GOES labels → build_windows(label_source='goes_class').

    Parameters
    ----------
    assigned_split : TRAIN / VAL / TEST constant — ALL windows from this day
                     are assigned to this partition (day-level split).

    Returns dict with X, y_binary, y_class, splits arrays plus stats, or None.
    """
    t0 = time.perf_counter()
    split_name = {TRAIN: "TRAIN", VAL: "VAL", TEST: "TEST"}.get(assigned_split, "?")
    print(f"  {label} [{split_name}] …", end=" ", flush=True)

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

    # ── Feature dict ──────────────────────────────────────────────────────
    prefix = "solexs_sdd2"
    feat_dict = {}
    for var in ds_pp.data_vars:
        if var == "split":
            continue
        if "spectrum" in var:
            continue
        if var == "goes_class":
            feat_dict["goes_class"] = ds_pp["goes_class"].values.astype(float)
            continue
        if var.startswith(prefix) and "quality" not in var:
            key = var.removeprefix(f"{prefix}_")
            feat_dict[key] = ds_pp[var].values.astype(float)

    # Use the within-day split_array only to exclude GAP cadences from windows.
    # We will override window split assignments below.
    split_arr = ds_pp["split"].values

    # ── Build windows ─────────────────────────────────────────────────────
    try:
        wd = build_windows(
            features        = feat_dict,
            split_array     = split_arr,
            label_source    = "goes_class",
            window_s        = WINDOW_S,
            stride_s        = STRIDE_S,
            label_horizon_s = LABEL_HORIZON_S,
            cadence_s       = float(ds.attrs.get("cadence_s", 1.0)),
        )
    except Exception as exc:
        print(f"✗ build_windows: {exc}")
        return None

    # ── DAY-LEVEL SPLIT OVERRIDE ──────────────────────────────────────────
    # Replace the within-day split assignments with the day's designated partition.
    # All windows from this day → assigned_split.  No gap needed between days
    # (days are weeks to months apart — no temporal continuity).
    day_splits = np.full(len(wd.splits), assigned_split, dtype=np.uint8)

    # ── M+ binary labels ──────────────────────────────────────────────────
    # Recast y_binary as M+ (y_class >= BINARY_MIN_CLASS).
    # The original C+ threshold (y_class >= 2) was degenerate on these days.
    y_binary_mp = (wd.y_class >= BINARY_MIN_CLASS).astype(np.int8)

    # ── Multi-horizon labels (15 / 60 min M+, 30 min X-class) ─────────────
    gc    = feat_dict.get("goes_class")
    n_cad = len(gc) if gc is not None else 0
    if gc is not None and hasattr(wd, "window_starts") and n_cad > 0:
        y_15min  = _horizon_binary(gc, wd.window_starts, WINDOW_S, 900,  BINARY_MIN_CLASS, n_cad)
        y_60min  = _horizon_binary(gc, wd.window_starts, WINDOW_S, 3600, BINARY_MIN_CLASS, n_cad)
        y_extreme = _horizon_binary(gc, wd.window_starts, WINDOW_S, LABEL_HORIZON_S, 4, n_cad)
    else:
        y_15min   = y_binary_mp.copy()
        y_60min   = y_binary_mp.copy()
        y_extreme = (wd.y_class >= 4).astype(np.int8)

    elapsed = time.perf_counter() - t0
    cls_names = ["A", "B", "C", "M", "X"]
    n_total = len(wd.y_class)
    n_mp    = int(y_binary_mp.sum())
    cls_counts = {
        c: int((wd.y_class == i).sum())
        for i, c in enumerate(cls_names)
        if int((wd.y_class == i).sum()) > 0
    }
    imb = (n_total - n_mp) / max(n_mp, 1)

    print(
        f"✓  {n_total} windows  M+%={100*n_mp/max(n_total,1):.1f}%  "
        f"imbal={imb:.1f}×  classes={cls_counts}  [{elapsed:.1f}s]"
    )

    return {
        "X":            wd.X,
        "y_binary":     y_binary_mp,
        "y_class":      wd.y_class,
        "splits":       day_splits,
        "y_15min":      y_15min,
        "y_60min":      y_60min,
        "y_extreme":    y_extreme,
        "feature_names": wd.feature_names,
        "n_total":      n_total,
        "n_mp":         n_mp,
        "label":        label,
        "source":       source,
        "assigned_split": assigned_split,
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

        configs = [
            (make_day(tmp, "20240209", n_hours=4, seed=11), datetime(2024, 2,  9), "2024-02-09", TEST),
            (make_day(tmp, "20240222", n_hours=4, seed=42), datetime(2024, 2, 22), "2024-02-22", TRAIN),
            (make_day(tmp, "20240506", n_hours=4, seed=99), datetime(2024, 5,  6), "2024-05-06", TRAIN),
            (_make_quiet_day(tmp, "20260612"),               None,                  "2026-06-12", TRAIN),
        ]

        print("Processing days:")
        results = []
        for day_dir, date, label, sp in configs:
            r = process_day(day_dir, date, label, sp)
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
    print("  Day-level split assignment:")
    split_labels = {TRAIN: "TRAIN", VAL: "VAL", TEST: "TEST"}
    for folder_name, date, sp in REAL_DAY_CONFIGS:
        print(f"    {folder_name.split('_')[3]}  →  {split_labels[sp]}")
    print()

    results = []
    for folder_name, date, assigned_split in REAL_DAY_CONFIGS:
        day_dir = data_dir / folder_name
        if not day_dir.exists():
            print(f"  {folder_name}: directory not found — skipping")
            continue
        label = folder_name.split("_")[3]
        r = process_day(day_dir, date, label, assigned_split)
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

    X_all    = np.concatenate([r["X"]        for r in results], axis=0).astype(np.float32)
    y_binary = np.concatenate([r["y_binary"] for r in results], axis=0)
    y_class  = np.concatenate([r["y_class"]  for r in results], axis=0)
    splits   = np.concatenate([r["splits"]   for r in results], axis=0)

    n_total = len(X_all)
    cls_names = ["A", "B", "C", "M", "X"]
    counts    = [int((y_class == i).sum()) for i in range(5)]

    print()
    print("=" * 65)
    print("  NEW DATASET (v2) — CLASS BREAKDOWN")
    print("=" * 65)
    print(f"  Total windows : {n_total:,}  shape={X_all.shape}")
    print()
    print(f"  {'Class':<6} {'Windows':>10} {'%':>8}")
    print("  " + "─" * 30)
    for cls, cnt in zip(cls_names, counts):
        bar = "█" * int(20 * cnt / max(n_total, 1))
        print(f"  {cls:<6} {cnt:>10,} {100*cnt/max(n_total,1):>7.2f}%  {bar}")
    print("  " + "─" * 30)
    print(f"  {'Total':<6} {n_total:>10,}  100.00%")

    x_count = counts[4]
    if x_count > 0:
        print(f"\n  ✓ X-class correctly labelled: {x_count} windows (was 0 in v1)")
    else:
        print("\n  ⚠ X-class = 0. Check GOES cross-match or data availability.")

    # ── Per-partition breakdown ───────────────────────────────────────────
    print()
    print("=" * 65)
    print("  PARTITION BREAKDOWN  (M+ binary = y_class >= M)")
    print("=" * 65)
    print(f"  {'Part':<8} {'Total':>6}  {'A':>5} {'B':>5} {'C':>5} {'M':>5} {'X':>5}  {'M+%':>6}  {'Imbal':>6}")
    print("  " + "─" * 63)

    part_stats = {}
    for part, pname in [(TRAIN, "TRAIN"), (VAL, "VAL"), (TEST, "TEST")]:
        mask = splits == part
        yc = y_class[mask]
        tot = len(yc)
        if tot == 0:
            continue
        pc = [int((yc == i).sum()) for i in range(5)]
        mp = pc[3] + pc[4]
        nm = tot - mp
        imb = nm / max(mp, 1)
        mp_pct = 100 * mp / max(tot, 1)
        xcls = pc[4]
        print(
            f"  {pname:<8} {tot:>6}  "
            + "  ".join(f"{c:>5}" for c in pc)
            + f"  {mp_pct:>5.1f}%  {imb:>5.2f}×"
        )
        part_stats[pname] = {
            "n": tot, "M+": mp, "X": xcls,
            "M+_pct": round(mp_pct, 2), "imbalance": round(imb, 2),
        }
    print("  " + "─" * 63)

    # Key checks
    print()
    tr = part_stats.get("TRAIN", {})
    te = part_stats.get("TEST", {})
    if te.get("X", 0) > 0:
        print(f"  ✓ TEST has {te['X']} X-class windows (was 0 in within-day split)")
    else:
        print("  ⚠ TEST has 0 X-class windows")
    if tr.get("X", 0) > 0:
        print(f"  ✓ TRAIN has {tr['X']} X-class windows (strongest events in training)")

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path("/tmp")
    np.save(out_dir / "X_v2.npy",        X_all)
    np.save(out_dir / "y_binary_v2.npy", y_binary)
    np.save(out_dir / "y_class_v2.npy",  y_class)
    np.save(out_dir / "splits_v2.npy",   splits)

    # Multi-horizon labels
    y_15min  = np.concatenate([r["y_15min"]  for r in results], axis=0)
    y_60min  = np.concatenate([r["y_60min"]  for r in results], axis=0)
    y_extreme = np.concatenate([r["y_extreme"] for r in results], axis=0)
    np.save(out_dir / "y_15min_v2.npy",  y_15min)
    np.save(out_dir / "y_60min_v2.npy",  y_60min)
    np.save(out_dir / "y_extreme_v2.npy", y_extreme)

    # Feature names
    feature_names = results[0].get("feature_names", [])
    with open(out_dir / "feature_names_v2.json", "w") as f:
        json.dump(feature_names, f)

    # Focal loss parameters for M+ binary and multiclass
    tr_mp  = tr.get("M+", 1)
    tr_tot = tr.get("n", 1)
    tr_nm  = tr_tot - tr_mp
    tr_imb = tr_nm / max(tr_mp, 1)

    # Effective Number of Samples per class (Lin et al. β=0.9999)
    beta = 0.9999
    ens = [((1 - beta**max(c, 1)) / (1 - beta)) for c in counts]
    alpha_ens = [1.0 / max(e, 1e-9) for e in ens]
    alpha_sum = sum(alpha_ens)
    alpha_norm = [round(a / alpha_sum * len(cls_names), 4) for a in alpha_ens]

    stats = {
        "n_total":        n_total,
        "shape":          list(X_all.shape),
        "class_counts":   dict(zip(cls_names, counts)),
        "label_source":   "goes_class",
        "binary_def":     "M+ (y_class >= 3)",
        "split_strategy": "day-level",
        "partitions":     part_stats,
        "focal_loss": {
            "binary_M+": {
                "train_imbalance": round(tr_imb, 2),
                "gamma":  1.0,
                "alpha":  round(tr_nm / max(tr_tot, 1), 3),
                "note": "barely imbalanced; class_weight preferred over focal",
            },
            "multiclass_alpha_ENS": dict(zip(cls_names, alpha_norm)),
        },
        "patch_size_s":   30,
        "n_patches":      WINDOW_S // 30,
    }

    with open(out_dir / "dataset_v2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"  Saved:")
    print(f"    /tmp/X_v2.npy           {X_all.nbytes / 1e6:.0f} MB  shape={X_all.shape}")
    print(f"    /tmp/y_binary_v2.npy    {y_binary.nbytes / 1e3:.1f} KB   (M+ binary)")
    print(f"    /tmp/y_class_v2.npy     {y_class.nbytes / 1e3:.1f} KB")
    print(f"    /tmp/splits_v2.npy      {splits.nbytes / 1e3:.1f} KB   (0=TRAIN 1=VAL 2=TEST)")
    print(f"    /tmp/dataset_v2_stats.json")

    print()
    print("  ── MODULE 4 HANDOFF ─────────────────────────────────────")
    print(f"  X shape        : {X_all.shape}  (windows × time × features)")
    print(f"  y_binary def   : M+  (y_class >= 3, M or X)")
    print(f"  TRAIN imbalance: {tr_imb:.2f}×  → class_weight={{0:{tr_imb:.2f}, 1:1.0}}")
    print(f"  focal_loss     : γ=1.0, α={round(tr_nm/max(tr_tot,1), 3)} (M+ head)")
    print(f"  Multiclass α   : {dict(zip(cls_names, alpha_norm))}  (ENS)")
    print(f"  Patch size     : 30 s → 60 patches per window (defended)")
    print(f"  Label source   : goes_class (GOES NGDC NetCDF + empirical fallback)")
    print(f"  StandardScaler : fit on TRAIN partition only")
    print(f"  Split strategy : day-level  (no within-day leakage gap needed)")
    print("  ─────────────────────────────────────────────────────────")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild WindowDataset with GOES labels and day-level splits"
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
