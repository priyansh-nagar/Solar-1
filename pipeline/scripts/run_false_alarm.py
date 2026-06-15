"""
run_false_alarm.py — Module 3 false alarm analysis
===================================================
Computes the aggregate False Alarm Rate (FAR) across all available days.
This is the required pre-Module-4 gate: ISRO evaluation criteria explicitly
list "low False Alarm Rate" as a judging metric.

HOW TO RUN
----------
With your real PRADAN data already processed by Modules 1-3:

    python pipeline/scripts/run_false_alarm.py --data-dir /tmp/solexs_data

With synthetic data (no FITS files needed — useful for CI / demo):

    python pipeline/scripts/run_false_alarm.py --synthetic

WHAT IT DOES
------------
For each day folder found under --data-dir (or one synthetic day):
  1. Runs Module 1 (ingest) + Module 2 (preprocess)
  2. Runs Module 3 detector with the two-channel voter
  3. Attempts GOES cross-match (falls back to SoLEXS-only labels if offline)
  4. Calls false_alarm_report() to aggregate FAR across all days
  5. Prints the full report + saves it to /tmp/false_alarm_report.json

TARGET NUMBERS (ISRO benchmarks)
---------------------------------
  < 0.5 FA/hr → excellent
  < 1.0 FA/hr → good (acceptable for submission)
  ≥ 1.0 FA/hr → needs detector tuning before Module 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_false_alarm")

# ── Make sure the workspace root is on sys.path ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.module1            import ingest_day
from pipeline.module2.preprocess import preprocess_day
from pipeline.module3            import (
    fetch_and_crossmatch,
    detect_flares,
    build_catalogue,
    false_alarm_report,
    print_false_alarm_report,
    NowcasterConfig,
)


# ---------------------------------------------------------------------------
# Process one day
# ---------------------------------------------------------------------------

def process_day(
    day_dir: Path,
    day_label: str,
    try_goes: bool = True,
) -> tuple | None:
    """
    Run Modules 1-3 on a single day directory.

    Returns (catalogue_df, preprocess_ds, goes_class_1s, day_label) or None.
    """
    print(f"\n  Processing {day_label} …", end="", flush=True)

    # ── Module 1: ingest ──────────────────────────────────────────────────
    try:
        ds = ingest_day(solexs_dir=str(day_dir), verbose=False)
    except Exception as exc:
        print(f"  ✗ ingest failed: {exc}")
        return None

    usable = ds.attrs.get("usable_fraction", 0.0)
    if usable < 0.05:
        print(f"  ⚠ usable_fraction={usable:.2f} — skipping")
        return None

    # ── Module 2: preprocess ──────────────────────────────────────────────
    try:
        ds_pp = preprocess_day(ds, verbose=False)
    except Exception as exc:
        print(f"  ✗ preprocess failed: {exc}")
        return None

    # ── GOES cross-match ──────────────────────────────────────────────────
    goes_class_1s = None
    if try_goes:
        # Infer date from day directory name (expects AL1_SLX_L1_YYYYMMDD_*)
        date_str = _extract_date(day_dir.name)
        if date_str:
            try:
                xm = fetch_and_crossmatch(ds_pp, date=date_str)
                if xm.n_crossmatched > 0:
                    goes_class_1s = xm.goes_class_1s
            except Exception as exc:
                logger.debug("GOES cross-match skipped for %s: %s", day_label, exc)

    # ── Module 3: detect ─────────────────────────────────────────────────
    config = NowcasterConfig(
        base_sigma=3.0,
        floor_excess_A=1.0,
        band_C_sigma=3.0,
        band_C_floor=1.0,
        min_sustain_s=30,
        min_gap_s=300,
    )

    try:
        events = detect_flares(
            ds_pp,
            goes_class_1s=goes_class_1s,
            config=config,
        )
    except Exception as exc:
        print(f"  ✗ detector failed: {exc}")
        return None

    cat = build_catalogue(events)
    n_ev = len(cat)
    gc_counts = dict(cat["goes_class"].value_counts()) if not cat.empty else {}
    print(f"  {n_ev} events {gc_counts}")

    return (cat, ds_pp, goes_class_1s, day_label)


# ---------------------------------------------------------------------------
# Synthetic-data path (no FITS files needed)
# ---------------------------------------------------------------------------

def run_synthetic() -> None:
    """
    Run the full false alarm pipeline on synthetic data.

    Creates three scenarios:
      • Two flare days (X+M+C flares)
      • One quiet day (background only)
    """
    from pathlib import Path
    import tempfile

    from pipeline.scripts.make_synthetic_day import make_day

    print("=" * 60)
    print("  SYNTHETIC DATA MODE")
    print("  (no PRADAN FITS files required)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Create synthetic days ─────────────────────────────────────────
        day_dirs = []

        # Flare day 1: X+M+C
        d1 = make_day(tmp, date_tag="20240222", n_hours=4, seed=42)
        day_dirs.append((d1, "2024-02-22 (X+M+C)"))

        # Flare day 2: M+C only (lower amplitude flares)
        d2 = make_day(tmp, date_tag="20240506", n_hours=4, seed=99)
        day_dirs.append((d2, "2024-05-06 (M+C)"))

        # Quiet day: background only — achieved by creating a day with
        # a very small peak so Module 3 should produce zero detections
        d3 = _make_quiet_day(tmp, date_tag="20260612")
        day_dirs.append((d3, "2026-06-12 (quiet)"))

        # ── Process each day ─────────────────────────────────────────────
        print("\nRunning Modules 1-3 on synthetic days:")
        records = []
        for day_dir, label in day_dirs:
            rec = process_day(day_dir, label, try_goes=False)
            if rec is not None:
                records.append(rec)

        _finish(records)


def _make_quiet_day(out_dir: Path, date_tag: str) -> Path:
    """
    Make a synthetic day with only background — no flares.
    Uses make_synthetic_day machinery but passes no flare list.
    """
    import gzip, io
    import astropy.io.fits as fits
    from pipeline.scripts.make_synthetic_day import (
        make_spectrum, build_pi_fits, build_gti_fits, write_fits_gz,
        N_CHANNELS,
    )

    n_hours  = 4
    n_times  = n_hours * 3600
    t0_unix  = 1749686400.0   # 2026-06-12 00:00:00 UTC (approx)
    tstart   = t0_unix + np.arange(n_times, dtype=np.float64)

    counts = make_spectrum(tstart, bg_band_a=30.0, flares=None, seed=7)

    gti_intervals = [(tstart[0], tstart[-1])]

    day_name = f"AL1_SLX_L1_{date_tag}_v1.0"
    sdd2_dir = out_dir / day_name / "SDD2"
    sdd2_dir.mkdir(parents=True, exist_ok=True)

    base     = f"AL1_SLX_SDD2_L1_{date_tag}"
    pi_path  = sdd2_dir / f"{base}.pi.gz"
    gti_path = sdd2_dir / f"{base}.gti.gz"

    write_fits_gz(build_pi_fits(tstart, counts),   pi_path)
    write_fits_gz(build_gti_fits(gti_intervals),   gti_path)

    return out_dir / day_name


# ---------------------------------------------------------------------------
# Real-data path
# ---------------------------------------------------------------------------

def run_real(data_dir: Path) -> None:
    """Run the pipeline on real PRADAN FITS data under data_dir."""
    print("=" * 60)
    print(f"  REAL DATA MODE — {data_dir}")
    print("=" * 60)

    # Find all day directories: AL1_SLX_L1_YYYYMMDD_*
    day_dirs = sorted(data_dir.glob("AL1_SLX_L1_*"))
    if not day_dirs:
        print(f"\n  ✗ No day directories found under {data_dir}")
        print("    Expected pattern: AL1_SLX_L1_YYYYMMDD_v*/")
        print("    Use --synthetic to run without FITS files.")
        sys.exit(1)

    print(f"\n  Found {len(day_dirs)} day directories")

    records = []
    for day_dir in day_dirs:
        date_str = _extract_date(day_dir.name) or day_dir.name
        rec = process_day(day_dir, date_str, try_goes=True)
        if rec is not None:
            records.append(rec)

    _finish(records)


# ---------------------------------------------------------------------------
# Shared finish step
# ---------------------------------------------------------------------------

def _finish(records: list) -> None:
    if not records:
        print("\n  ✗ No days processed successfully.")
        sys.exit(1)

    print(f"\n  Processed {len(records)} days — computing aggregate FAR …")

    report = false_alarm_report(records)
    print_false_alarm_report(report)

    # ── Save JSON report ──────────────────────────────────────────────────
    out_path = Path("/tmp/false_alarm_report.json")
    _safe_save_json(report, out_path)
    print(f"  Report saved → {out_path}")

    # ── Gate check for Module 4 ───────────────────────────────────────────
    far = report["far_aggregate"]
    print("\n  ── MODULE 4 GATE ─────────────────────────────────────────")
    if far < 1.0:
        print(f"  ✓ FAR = {far:.4f} FA/hr  → PASS  (< 1.0 threshold)")
        print("    Safe to proceed to Module 4 (PatchTST forecaster).")
        print()
        print("  Module 4 inherits from this run:")
        print(f"    • {report['total_tp_events']} true positives as training ground truth")
        print(f"    • Lead time p50 = {report['lead_time_all_p50_s']} s (all), "
              f"{report['lead_time_Mplus_p50_s']} s (M+)")
        print("    • Empirical thresholds: C≈0.07-0.15, M≈0.12-0.99, X≈3-25 (excess_A)")
        print("    • Use goes_class labels (NOT excess_A proxy) for WindowDataset rebuild")
    else:
        print(f"  ✗ FAR = {far:.4f} FA/hr  → FAIL  (≥ 1.0 threshold)")
        print("    Tighten NowcasterConfig.base_sigma or min_sustain_s before Module 4.")
    print("  ─────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_date(name: str) -> str | None:
    """Extract YYYYMMDD from a directory name like AL1_SLX_L1_20240222_v1.0."""
    import re
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None


def _safe_save_json(report: dict, path: Path) -> None:
    """Save report dict to JSON, converting numpy scalars to Python types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        return _convert(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_walk(report), f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 3 false alarm analysis — required before Module 4"
    )
    parser.add_argument(
        "--data-dir", default="/tmp/solexs_data",
        help="Root directory containing AL1_SLX_L1_YYYYMMDD_* day folders",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run on synthetic data (no FITS files required)",
    )
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
    else:
        run_real(Path(args.data_dir))


if __name__ == "__main__":
    main()
