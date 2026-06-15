"""
Module 3 — Flare Nowcaster
==========================
Answers one question per cadence: is a flare happening right now?

Components (in call order):
  goes.py        — Download GOES XRS 1-min data, align to SoLEXS grid,
                   compute empirical excess_A↔GOES-class mapping.
  threshold.py   — Per-cadence adaptive detection threshold scaled by
                   rolling_std_5min so background noise sets the bar.
  detector.py    — Multi-channel voter: requires excess_A spike AND
                   hardness_ratio shift to confirm; 30-s sustain enforced.
  catalogue.py   — Assemble FlareEvent list → clean pandas DataFrame.
  evaluate.py    — TPR, FAR, lead-time metrics on GTI-valid cadences only.

Typical usage
-------------
    from pipeline.module3 import (
        fetch_and_crossmatch,
        detect_flares,
        build_catalogue,
        evaluate_catalogue,
    )

    # 3a: GOES cross-match + empirical thresholds
    xm = fetch_and_crossmatch(preprocess_ds, date="2024-02-22")

    # 3b+3c: Nowcaster
    events = detect_flares(preprocess_ds, xm.thresholds)

    # Catalogue
    cat = build_catalogue(events)

    # Metrics
    metrics = evaluate_catalogue(cat, xm.goes_df, preprocess_ds)
"""

from .goes       import fetch_and_crossmatch, CrossMatchResult
from .detector   import detect_flares, NowcasterConfig, FlareEvent
from .catalogue  import build_catalogue, save_catalogue, load_catalogue
from .evaluate   import evaluate_catalogue

__all__ = [
    "fetch_and_crossmatch",
    "CrossMatchResult",
    "detect_flares",
    "NowcasterConfig",
    "FlareEvent",
    "build_catalogue",
    "save_catalogue",
    "load_catalogue",
    "evaluate_catalogue",
]
