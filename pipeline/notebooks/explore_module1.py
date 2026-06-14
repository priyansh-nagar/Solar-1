"""
explore_module1.py — Verification notebook for Module 1
=========================================================
Run with:  python pipeline/notebooks/explore_module1.py

Prints a structured sanity-check report and saves a text summary.
No matplotlib required — all output is plain text so it runs in any env.
"""

import logging
import sys
import os

# Add repo root to path so 'pipeline' is importable from any cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)

import numpy as np

SOLEXS_DIR = "/tmp/solexs_data/AL1_SLX_L1_20260612_v1.0"

print("=" * 70)
print("Module 1 — Data Ingestion   Verification Report")
print("=" * 70)

# ─── 1. Discovery ─────────────────────────────────────────────────────────
from pipeline.module1.formats import discover_products
products = discover_products(SOLEXS_DIR)
print(f"\n[1] Product discovery")
for p in products:
    print(f"    {p}")

# ─── 2. Ingest ────────────────────────────────────────────────────────────
from pipeline.module1 import ingest_day
print(f"\n[2] Running ingest_day() ...")
ds = ingest_day(solexs_dir=SOLEXS_DIR, verbose=True)
print(f"\n    Dataset variables:")
for var in sorted(ds.data_vars):
    da = ds[var]
    print(f"      {var:<40s} {str(da.dims):<25s} dtype={da.dtype}")

print(f"\n    Dataset coordinates:")
for coord in ds.coords:
    print(f"      {coord:<40s} shape={ds.coords[coord].shape}")

# ─── 3. Time axis ─────────────────────────────────────────────────────────
times = ds.time.values
print(f"\n[3] Time axis")
print(f"    Start : {times[0]}")
print(f"    End   : {times[-1]}")
print(f"    Length: {len(times)} cadences")
dt_ns = np.diff(times.astype("datetime64[ns]").astype(np.int64))
print(f"    Cadence (s): min={dt_ns.min()/1e9:.3f}  max={dt_ns.max()/1e9:.3f}  "
      f"mean={dt_ns.mean()/1e9:.3f}")

# ─── 4. Quality summary ───────────────────────────────────────────────────
from pipeline.module1.quality import quality_summary, is_usable
q = ds["solexs_sdd2_quality"].values
qs = quality_summary(q)
print(f"\n[4] Quality flags (SDD2)")
for k, v in qs.items():
    print(f"    {k:<25s}: {v}")

# ─── 5. Spectrum statistics ───────────────────────────────────────────────
spec = ds["solexs_sdd2_spectrum"].values
total = ds["solexs_sdd2_total"].values
usable_mask = is_usable(q)

valid_total = total[usable_mask & ~np.isnan(total)]
print(f"\n[5] Spectrum statistics (usable cadences only)")
print(f"    Total flux — mean={valid_total.mean():.1f}  "
      f"median={np.median(valid_total):.1f}  "
      f"max={valid_total.max():.1f}  min={valid_total.min():.1f}  cts/s")

# Peak channel across all usable spectra
spec_usable = spec[usable_mask]
spec_usable_nonnan = spec_usable[~np.all(np.isnan(spec_usable), axis=1)]
mean_spectrum = np.nanmean(spec_usable_nonnan, axis=0)
peak_ch = int(np.nanargmax(mean_spectrum))
from pipeline.module1.channels import SOLEXS_CALIBRATION
e_peak = SOLEXS_CALIBRATION.channel_to_energy(np.array([peak_ch]))[0]
print(f"    Mean spectrum peak at channel {peak_ch} ({e_peak:.2f} keV)  "
      f"mean_counts={mean_spectrum[peak_ch]:.3f}")

# ─── 6. Energy band breakdown ─────────────────────────────────────────────
print(f"\n[6] Energy band light curves (SDD2)")
for band in ["A", "B", "C", "D"]:
    arr = ds[f"solexs_sdd2_band_{band}"].values
    valid = arr[~np.isnan(arr)]
    lo, hi = SOLEXS_CALIBRATION.science_bands[band]
    print(f"    Band {band} ({lo:.1f}–{hi:.1f} keV): "
          f"mean={valid.mean():.2f}  max={valid.max():.2f}  "
          f"valid_pts={len(valid)}")

# ─── 7. Gap summary ───────────────────────────────────────────────────────
from pipeline.module1.gaps import find_gaps, gap_summary
gaps = find_gaps(usable_mask, np.arange(len(times), dtype=float))
gs = gap_summary(gaps)
print(f"\n[7] Gap analysis (SDD2)")
for k, v in gs.items():
    print(f"    {k:<25s}: {v}")
if gaps:
    print(f"    Largest gaps (top 5):")
    for g in sorted(gaps, key=lambda x: -x[2])[:5]:
        t_start_str = str(times[g[0]])[:23]
        print(f"      idx {g[0]:6d}–{g[1]:6d}  duration={g[2]:.0f} s  "
              f"starts at {t_start_str}")

# ─── 8. Dataset metadata ──────────────────────────────────────────────────
print(f"\n[8] Dataset attributes")
for k, v in ds.attrs.items():
    print(f"    {k:<25s}: {v}")

print("\n" + "=" * 70)
print("Module 1 verification complete.  No crashes — pipeline is functional.")
print("=" * 70)
