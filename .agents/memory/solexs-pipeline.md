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

## Module 4 inputs
- X_all.npy: (7459, 1800, 29) @ /tmp/
- y_binary_all.npy, y_class_all.npy @ /tmp/
- focal_loss: γ=2.0, α=0.80; imbalance 17.9×
- Catalogue ground truth: /tmp/module3_catalogue.csv
- StandardScaler must be fit on TRAIN partition only
