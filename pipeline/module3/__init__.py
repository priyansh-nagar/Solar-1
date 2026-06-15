"""
Module 3 — Flare Nowcaster
==========================
Answers one question per cadence: is a flare happening right now?

Components (in call order):
  goes.py        — Download GOES XRS 1-min data, align to SoLEXS grid,
                   compute empirical excess_A↔GOES-class mapping.
  threshold.py   — Per-cadence adaptive detection threshold scaled by
                   rolling stats of excess_A itself (NOT rolling_std_5min
                   which is in raw cts/s — that bug is documented in memory).
  detector.py    — Two-channel voter: excess_A spike AND Band C count-rate
                   must both agree for ≥30 s. Band C is the hard channel
                   (NOT hardness_ratio, which decreases during flares).
  catalogue.py   — Assemble FlareEvent list → clean pandas DataFrame.
  evaluate.py    — TPR, FAR, lead-time metrics on GTI-valid cadences only.
                   Also: false_alarm_report() for multi-day aggregate FAR
                   (ISRO evaluation criterion — required before Module 4).

Typical usage
-------------
    from pipeline.module3 import (
        fetch_and_crossmatch,
        detect_flares,
        build_catalogue,
        evaluate_catalogue,
        false_alarm_report,
        print_false_alarm_report,
    )

    # 3a: GOES cross-match + empirical thresholds
    xm = fetch_and_crossmatch(preprocess_ds, date="2024-02-22")

    # 3b+3c: Nowcaster
    events = detect_flares(
        preprocess_ds,
        goes_class_1s=xm.goes_class_1s,
        empirical_thresholds=xm.thresholds,
    )

    # Catalogue
    cat = build_catalogue(events)

    # Per-day metrics
    metrics = evaluate_catalogue(cat, preprocess_ds, xm.goes_class_1s)

    # Multi-day false alarm report (required before Module 4)
    report = false_alarm_report([
        (cat_day1, ds_day1, xm_day1.goes_class_1s, "2024-02-22"),
        (cat_day2, ds_day2, xm_day2.goes_class_1s, "2024-05-06"),
        (cat_quiet, ds_quiet, None,                 "2026-06-12"),  # quiet day
    ])
    print_false_alarm_report(report)
"""

from .goes       import fetch_and_crossmatch, CrossMatchResult
from .detector   import detect_flares, NowcasterConfig, FlareEvent
from .catalogue  import build_catalogue, save_catalogue, load_catalogue
from .evaluate   import (
    evaluate_catalogue,
    false_alarm_report,
    print_false_alarm_report,
)

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
    "false_alarm_report",
    "print_false_alarm_report",
]
