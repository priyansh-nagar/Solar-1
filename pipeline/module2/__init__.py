"""
Module 2 — Preprocessing
=========================
Takes the xarray Dataset from module1.ingest_day() and produces a
cleaned, background-subtracted, feature-engineered Dataset ready for
the nowcaster (Module 3) and the forecasting model (Module 4).

Pipeline steps
--------------
  1. Savitzky-Golay smoothing   (detrend.py)   — remove high-freq noise
  2. SNIP background estimation (background.py) — model slow solar variation
  3. Feature engineering        (features.py)   — derivative, rate-of-rise,
                                                   band ratio, rolling stats
  4. Temporal train/val/test split (split.py)   — 70 / 15 / 15 by time
  5. Sliding window builder     (windows.py)    — 30-min windows + labels

Public API
----------
    from pipeline.module2 import preprocess_day

    ds_clean = preprocess_day(ds)             # extends the Module 1 Dataset
    windows  = preprocess_day(ds, build_windows=True)   # returns WindowDataset
"""

from .preprocess import preprocess_day
from .windows import WindowDataset

__all__ = ["preprocess_day", "WindowDataset"]
