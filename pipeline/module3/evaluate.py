"""
evaluate.py — Nowcaster performance metrics
============================================

Computes detection performance against the GOES cross-match ground truth.

CRITICAL: evaluation runs ONLY on GTI-valid, non-saturated cadences.
  Saturated cadences (GOES X-class events that overloaded SoLEXS) must be
  excluded from both numerator and denominator — counting them as misses
  would artificially deflate recall.

Metrics produced
----------------
  TPR (True Positive Rate / Recall)
    = detected_flare_cadences / all_flare_cadences
    where "all_flare_cadences" = GOES says ≥ C-class AND cadence is GTI-valid.

  FAR (False Alarm Rate per hour)
    = false_alarm_events / observation_hours
    A "false alarm event" is a detection that does not overlap any GOES flare.

  Precision
    = true_positive_events / detected_events

  F1  = 2 × Precision × TPR / (Precision + TPR)

  Lead time (per event)
    = onset_time − GOES_peak_time   [seconds]
    Positive = detected BEFORE peak (good).
    Stored per-event in the catalogue; summary reports p25/p50/p75.

  Lead time (overall)
    Reported only for M+ events where lead time matters operationally.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# GOES class integer threshold for "flare" in evaluation
EVAL_FLARE_CLASS_MIN = 2   # C-class (int=2) and above


def evaluate_catalogue(
    catalogue_df: "pd.DataFrame",
    preprocess_ds,
    goes_class_1s: Optional[np.ndarray] = None,
    prefix: str = "solexs_sdd2",
    cadence_s: float = 1.0,
) -> dict:
    """
    Compute detection performance metrics.

    Parameters
    ----------
    catalogue_df  : output of build_catalogue() — detected events
    preprocess_ds : xr.Dataset from module2.preprocess_day()
    goes_class_1s : (n_times,) int8 from CrossMatchResult (None = use SoLEXS labels)
    prefix        : detector variable prefix
    cadence_s     : seconds per sample

    Returns
    -------
    dict with keys: tpr, far_per_hour, precision, f1, lead_time_*, n_*
    """
    import pandas as pd
    from pipeline.module1.quality import is_usable, QFlag

    # ── Quality mask ─────────────────────────────────────────────────────
    q_key = f"{prefix}_quality"
    if q_key in preprocess_ds.data_vars:
        quality = preprocess_ds[q_key].values.astype(np.uint8)
        usable  = is_usable(quality)
        not_sat = (quality & QFlag.SATURATED) == 0
        eval_mask = usable & not_sat        # GTI-valid and not saturated
    else:
        n = len(preprocess_ds.coords["time"])
        eval_mask = np.ones(n, dtype=bool)

    n_eval = int(eval_mask.sum())
    obs_hours = (n_eval * cadence_s) / 3600.0

    # ── Ground truth: GOES C+ cadences within eval_mask ──────────────────
    if goes_class_1s is not None:
        gt_flare = eval_mask & (goes_class_1s >= EVAL_FLARE_CLASS_MIN)
    else:
        # Fallback: use SoLEXS excess_A ≥ C-class proxy threshold
        exc_key = f"{prefix}_excess_A"
        if exc_key in preprocess_ds.data_vars:
            excess_A = preprocess_ds[exc_key].values
            gt_flare = eval_mask & (excess_A >= 5.0) & ~np.isnan(excess_A)
        else:
            gt_flare = np.zeros(len(eval_mask), dtype=bool)

    n_gt_flare_cadences = int(gt_flare.sum())

    # ── Build detected-flare binary array from catalogue events ──────────
    n_times  = len(eval_mask)
    detected = np.zeros(n_times, dtype=bool)

    if not catalogue_df.empty:
        for _, row in catalogue_df.iterrows():
            s = int(row["onset_idx"])
            e = int(row["end_idx"])
            s = max(0, s);  e = min(n_times - 1, e)
            detected[s : e + 1] = True

    # Restrict to eval cadences
    det_in_eval = detected & eval_mask

    # ── Cadence-level TP / FP / FN ───────────────────────────────────────
    tp_cadences = int((det_in_eval & gt_flare).sum())
    fp_cadences = int((det_in_eval & ~gt_flare).sum())
    fn_cadences = int((~det_in_eval & gt_flare).sum())

    tpr = tp_cadences / max(n_gt_flare_cadences, 1)

    # ── Event-level FAR (false alarm events per hour) ─────────────────────
    # A detection event is a false alarm if it has no overlap with GOES flare cadences
    n_detected_events   = len(catalogue_df)
    n_tp_events = 0
    n_fa_events = 0

    for _, row in catalogue_df.iterrows():
        s = max(0, int(row["onset_idx"]))
        e = min(n_times - 1, int(row["end_idx"]))
        if gt_flare[s : e + 1].any():
            n_tp_events += 1
        else:
            n_fa_events += 1

    far_per_hour = n_fa_events / max(obs_hours, 1.0 / 3600.0)
    precision    = n_tp_events / max(n_detected_events, 1)
    f1           = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0 else 0.0
    )

    # ── Lead times from catalogue ─────────────────────────────────────────
    if not catalogue_df.empty and "lead_time_s" in catalogue_df.columns:
        lt = catalogue_df["lead_time_s"].dropna()
        lt_mplus = catalogue_df.loc[
            catalogue_df["goes_class_int"] >= 3, "lead_time_s"
        ].dropna()
    else:
        lt = lt_mplus = __import__("pandas").Series([], dtype=float)

    def _stats(s):
        if len(s) == 0:
            return {"p25": np.nan, "p50": np.nan, "p75": np.nan}
        return {
            "p25": round(float(s.quantile(0.25)), 1),
            "p50": round(float(s.median()), 1),
            "p75": round(float(s.quantile(0.75)), 1),
        }

    return {
        # Cadence-level
        "tpr":                    round(tpr, 4),
        "precision":              round(precision, 4),
        "f1":                     round(f1, 4),

        # Event-level
        "n_detected_events":      n_detected_events,
        "n_tp_events":            n_tp_events,
        "n_fa_events":            n_fa_events,
        "far_per_hour":           round(far_per_hour, 4),

        # Cadence counts
        "n_eval_cadences":        n_eval,
        "n_gt_flare_cadences":    n_gt_flare_cadences,
        "n_tp_cadences":          tp_cadences,
        "n_fp_cadences":          fp_cadences,
        "n_fn_cadences":          fn_cadences,
        "obs_hours":              round(obs_hours, 2),

        # Lead times (all events)
        "lead_time_all":          _stats(lt),
        # Lead times (M+ only — operationally relevant)
        "lead_time_M_plus":       _stats(lt_mplus),
    }
