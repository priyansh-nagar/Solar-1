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

false_alarm_report()
--------------------
  Aggregates FAR across a list of (catalogue, preprocess_ds, goes_class_1s,
  day_label) tuples so you can compute the headline number across all days.
  Required by ISRO evaluation criteria ("low False Alarm Rate" is an explicit
  judging metric).

  ISRO benchmarks:
    < 1.0 false alert per hour : acceptable
    < 0.5 false alert per hour : excellent
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Multi-day false alarm analysis
# ---------------------------------------------------------------------------

DayRecord = Tuple[
    "pd.DataFrame",           # catalogue_df from build_catalogue()
    object,                   # preprocess_ds (xr.Dataset)
    Optional[np.ndarray],     # goes_class_1s, or None
    str,                      # human-readable day label e.g. "2024-02-22"
]


def false_alarm_report(
    days: Sequence[DayRecord],
    prefix: str = "solexs_sdd2",
    cadence_s: float = 1.0,
    quiet_hours_override: Optional[float] = None,
) -> dict:
    """
    Compute aggregate False Alarm Rate across multiple observation days.

    This is the pre-Module-4 gate: ISRO evaluation criteria explicitly require
    a low FAR. Run this after Module 3 finishes all days.

    Parameters
    ----------
    days : list of (catalogue_df, preprocess_ds, goes_class_1s, day_label)
        Each tuple is one day's worth of data as produced by the Module 3
        pipeline.  Pass goes_class_1s=None to use SoLEXS-only ground truth.
    prefix : detector variable prefix (default "solexs_sdd2")
    cadence_s : seconds per sample
    quiet_hours_override : if set, use this as total quiet hours instead of
        computing from quality flags (useful when running on a subset of data).

    Returns
    -------
    dict with keys:
      per_day               : list of per-day result dicts
      total_fa_events       : int   — sum of false alarm events
      total_quiet_hours     : float — sum of non-flare GTI hours
      far_aggregate         : float — total_fa_events / total_quiet_hours
      far_rating            : str   — "excellent" / "good" / "marginal" / "poor"
      total_detections      : int   — TP + FA events across all days
      total_tp_events       : int
      aggregate_precision   : float
      lead_time_all_p50_s   : float
      lead_time_Mplus_p50_s : float

    FAR rating benchmarks (ISRO-aligned)
    -------------------------------------
      < 0.5 / hr  → "excellent"
      < 1.0 / hr  → "good"
      < 2.0 / hr  → "marginal"
      ≥ 2.0 / hr  → "poor"
    """
    import pandas as pd
    from pipeline.module1.quality import is_usable, QFlag

    per_day: list = []
    total_fa   = 0
    total_tp   = 0
    total_det  = 0
    total_quiet_h = 0.0
    all_lead_times    : list = []
    all_lt_mplus      : list = []

    for catalogue_df, preprocess_ds, goes_class_1s, day_label in days:
        n_times = len(preprocess_ds.coords["time"])

        # ── Quality / eval mask ───────────────────────────────────────────
        q_key = f"{prefix}_quality"
        if q_key in preprocess_ds.data_vars:
            quality   = preprocess_ds[q_key].values.astype(np.uint8)
            usable    = is_usable(quality)
            not_sat   = (quality & QFlag.SATURATED) == 0
            eval_mask = usable & not_sat
        else:
            eval_mask = np.ones(n_times, dtype=bool)

        # ── Ground-truth flare cadences ───────────────────────────────────
        if goes_class_1s is not None:
            gt_flare = eval_mask & (goes_class_1s >= EVAL_FLARE_CLASS_MIN)
        else:
            exc_key = f"{prefix}_excess_A"
            if exc_key in preprocess_ds.data_vars:
                excess_A = preprocess_ds[exc_key].values
                gt_flare = eval_mask & (excess_A >= 5.0) & ~np.isnan(excess_A)
            else:
                gt_flare = np.zeros(n_times, dtype=bool)

        # ── Quiet cadences: eval-valid AND not in any GOES flare window ───
        # Expand GOES flare windows by ±120 s to avoid edge false-positives
        # being counted (flare rise/decay that GOES hasn't classified yet).
        flare_expanded = _expand_bool_mask(gt_flare, margin_samples=int(120 / cadence_s))
        quiet_mask = eval_mask & ~flare_expanded
        quiet_hours = float(quiet_mask.sum() * cadence_s / 3600.0)

        # ── Count false alarm events ──────────────────────────────────────
        fa_count = 0
        tp_count = 0
        fa_details: list = []

        if not catalogue_df.empty:
            for _, row in catalogue_df.iterrows():
                s = max(0, int(row["onset_idx"]))
                e = min(n_times - 1, int(row["end_idx"]))
                # True positive: overlaps with GOES C+ (before expansion)
                if gt_flare[s : e + 1].any():
                    tp_count += 1
                else:
                    fa_count += 1
                    fa_details.append({
                        "day":         day_label,
                        "onset_time":  row.get("onset_time", "?"),
                        "onset_idx":   s,
                        "goes_class":  row.get("goes_class", "?"),
                        "peak_excess_A": row.get("peak_excess_A", np.nan),
                    })

        day_far = fa_count / max(quiet_hours, 1.0 / 3600.0)

        # ── Lead times ────────────────────────────────────────────────────
        if "lead_time_s" in catalogue_df.columns:
            lt = catalogue_df["lead_time_s"].dropna().tolist()
            all_lead_times.extend(lt)
            if "goes_class_int" in catalogue_df.columns:
                lt_mp = catalogue_df.loc[
                    catalogue_df["goes_class_int"] >= 3, "lead_time_s"
                ].dropna().tolist()
                all_lt_mplus.extend(lt_mp)

        per_day.append({
            "day":              day_label,
            "n_detections":     len(catalogue_df),
            "n_tp_events":      tp_count,
            "n_fa_events":      fa_count,
            "quiet_hours":      round(quiet_hours, 2),
            "far_per_hour":     round(day_far, 4),
            "fa_details":       fa_details,
        })

        total_fa    += fa_count
        total_tp    += tp_count
        total_det   += len(catalogue_df)
        total_quiet_h += quiet_hours

    # ── Aggregate ─────────────────────────────────────────────────────────
    if quiet_hours_override is not None:
        total_quiet_h = quiet_hours_override

    far_agg = total_fa / max(total_quiet_h, 1.0 / 3600.0)
    precision_agg = total_tp / max(total_det, 1)

    lt_arr   = np.array(all_lead_times,  dtype=float)
    lt_mp_arr= np.array(all_lt_mplus,    dtype=float)

    lt_p50    = float(np.median(lt_arr))    if len(lt_arr)    > 0 else np.nan
    lt_mp_p50 = float(np.median(lt_mp_arr)) if len(lt_mp_arr) > 0 else np.nan

    if far_agg < 0.5:
        rating = "excellent"
    elif far_agg < 1.0:
        rating = "good"
    elif far_agg < 2.0:
        rating = "marginal"
    else:
        rating = "poor"

    return {
        "per_day":               per_day,
        "total_fa_events":       total_fa,
        "total_tp_events":       total_tp,
        "total_detections":      total_det,
        "total_quiet_hours":     round(total_quiet_h, 2),
        "far_aggregate":         round(far_agg, 4),
        "far_rating":            rating,
        "aggregate_precision":   round(precision_agg, 4),
        "lead_time_all_p50_s":   round(lt_p50,    1) if not np.isnan(lt_p50)    else np.nan,
        "lead_time_Mplus_p50_s": round(lt_mp_p50, 1) if not np.isnan(lt_mp_p50) else np.nan,
    }


def print_false_alarm_report(report: dict) -> None:
    """
    Print the false alarm report in human-readable form.

    Call this after false_alarm_report() and share the output before
    proceeding to Module 4.
    """
    sep = "─" * 60

    print(f"\n{sep}")
    print("  MODULE 3 — FALSE ALARM ANALYSIS")
    print(sep)

    print(f"\n{'Day':<14} {'Det':>4} {'TP':>4} {'FA':>4} {'Quiet h':>8} {'FAR/hr':>8}")
    print("─" * 46)
    for d in report["per_day"]:
        print(
            f"{d['day']:<14} "
            f"{d['n_detections']:>4} "
            f"{d['n_tp_events']:>4} "
            f"{d['n_fa_events']:>4} "
            f"{d['quiet_hours']:>8.2f} "
            f"{d['far_per_hour']:>8.4f}"
        )

    print("─" * 46)
    print(
        f"{'TOTAL':<14} "
        f"{report['total_detections']:>4} "
        f"{report['total_tp_events']:>4} "
        f"{report['total_fa_events']:>4} "
        f"{report['total_quiet_hours']:>8.2f} "
        f"{report['far_aggregate']:>8.4f}"
    )

    print(f"\n  Aggregate FAR : {report['far_aggregate']:.4f} false alerts / hour")
    print(f"  Rating        : {report['far_rating'].upper()}")
    print(f"  Precision     : {report['aggregate_precision']:.4f}")
    print(f"  Lead time p50 : {report['lead_time_all_p50_s']} s (all events)")
    print(f"  Lead time p50 : {report['lead_time_Mplus_p50_s']} s (M+ only)")

    # ISRO benchmark
    far = report["far_aggregate"]
    if far < 0.5:
        verdict = "✓ EXCELLENT — well below 0.5 FA/hr (ISRO best-tier)"
    elif far < 1.0:
        verdict = "✓ GOOD — below 1.0 FA/hr (ISRO acceptable)"
    elif far < 2.0:
        verdict = "⚠ MARGINAL — above 1.0 FA/hr; tighten thresholds"
    else:
        verdict = "✗ POOR — above 2.0 FA/hr; detector needs tuning"

    print(f"\n  ISRO assessment: {verdict}")

    # List individual false alarms
    all_fa = [
        fa
        for d in report["per_day"]
        for fa in d.get("fa_details", [])
    ]
    if all_fa:
        print(f"\n  Individual false alarms ({len(all_fa)} total):")
        for fa in all_fa:
            print(
                f"    {fa['day']}  {fa['onset_time']}  "
                f"excess_A={fa['peak_excess_A']:.3f}"
            )
    else:
        print("\n  No individual false alarms recorded.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_bool_mask(mask: np.ndarray, margin_samples: int) -> np.ndarray:
    """
    Dilate a boolean mask by *margin_samples* on each side (binary dilation).

    Used to create a ±120-s buffer around GOES flare cadences so that the
    detector's early-rise triggers aren't penalised as false alarms.

    Pure-numpy implementation: O(n · margin) for small margins, which is
    fine for margin_samples ≤ 300 and n ≤ 86400.
    """
    if margin_samples <= 0 or not mask.any():
        return mask.copy()

    out = mask.copy()
    indices = np.where(mask)[0]
    for idx in indices:
        lo = max(0, idx - margin_samples)
        hi = min(len(mask), idx + margin_samples + 1)
        out[lo:hi] = True
    return out
