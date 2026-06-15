---
name: SoLEXS flare detection pipeline
description: Modules 1-3 complete; critical calibration facts for Module 4
---

## Data facts
- SDD1 is offline on all 6 PRADAN days (only .gti files, no .lc/.pi)
- SDD2 is the working detector throughout
- `quality==0x00` means NOT usable; `is_usable()` checks bit0 (IN_GTI) must be SET, bit1/2 must be CLEAR
- X4.5 day (2024-05-06): SoLEXS saturated → 0 flare windows from Band A alone; GOES cross-match rescues the X label

## Calibration bugs fixed (important for Module 4)
- `rolling_std_5min` is in raw cts/s (~6 median, 4355 max); `excess_A` is dimensionless (max ~29). NEVER use rolling_std_5min as threshold for excess_A — the ratio is ~12,000× wrong.
- `hardness_ratio` = Band C / Band A. During flares Band A rises ~60×, Band C ~10×, so ratio DECREASES — wrong sign for detection. Do not use as trigger.

## Module 3 empirical thresholds (real SoLEXS calibration)
From GOES cross-match across 5 active days:
- B-class: excess_A p50 ≈ 0.07–0.15
- C-class: excess_A p50 ≈ 0.07–0.15
- M-class: excess_A p50 ≈ 0.12–0.99
- X-class: excess_A p50 ≈ 3–25 (highly variable)
Replace first-principles thresholds in windows.py before Module 4 training.

## Module 3 results
- 10 events detected across 5 days; 0 on quiet day (2026-06-12)
- Lead time: p25=42s, p50=68s, p75=92s; M+ only: p50=77s
- GOES class: M×7, X×2, C×1; Precision=0.67–1.0; FAR=0.0–0.043/hr
- Catalogue saved to /tmp/module3_catalogue.csv

## GOES download
- NOAA NGDC URL pattern (GOES-16 1-min averages): https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/{YYYY}/{MM}/
- Requires h5py + h5netcdf (NOT netCDF4); cache as CSV (no pyarrow/fastparquet)
- 2026-06-12: 404 from NOAA (future date, not yet in archive)

## Module 3 false alarm analysis (pre-Module-4 gate) — COMPLETE
- Script: pipeline/scripts/run_false_alarm.py --synthetic | --data-dir /path
- false_alarm_report() + print_false_alarm_report() now in evaluate.py + __init__.py
- Synthetic result: FAR = 0.1006 FA/hr → EXCELLENT (< 0.5 threshold)
- Quiet day (2026-06-12): 0 false alarms (correct)
- FAR gate: < 1.0 FA/hr to proceed to Module 4
- ±120-s buffer around GOES flare windows prevents penalising early-rise triggers as FA

## Module 4 inputs — v2 dataset (REAL DATA, GOES labels)
- /tmp/X_v2.npy shape=(7470,1800,29), y_binary_v2.npy, y_class_v2.npy
- GOES source: ngdc_netcdf for 5 active days; excess_A_fallback for 20260612 (future date)
- Class breakdown: A=119(1.6%), B=2(0.03%), C=4723(63.2%), M=2311(30.9%), X=315(4.2%)
- DATASET BIAS: all 6 days are peak Solar Cycle 25 — GOES background stays at C-level
  → C+ binary is useless (98.4% positive). Use M+ as the binary "flare" definition.
- Binary thresholds:
    C+: imbalance 0.02× (degenerate — do NOT use)
    M+: imbalance 1.84× → focal loss not needed; use class_weight={0:1, 1:1.84}
    X:  imbalance 22.7× → γ=2.0, α=0.958
- Multiclass focal α: A=0.984, B=1.00, C=0.368, M=0.691, X=0.958
- Patch size: 30 s → 60 patches per window (patch_size=30, num_patches=60) — LOCKED
- h5py installed; StandardScaler must be fit on TRAIN partition only
