"""
evaluate.py — Evaluation metrics for Module 4
==============================================
Primary metric: TPR @ FAR = 0.5 false alarms / hour
Secondary:
  • Lead time (median cadences before M+ onset in TEST)
  • X-class recall
  • Expected Calibration Error (ECE)
  • ROC-AUC

VAL caveat: 20240510 has 60.9 % M+ prevalence → VAL metrics are optimistic.
Flag this explicitly in any report; rely on TEST for final numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import SolarPatchTST


# ---------------------------------------------------------------------------
# Core threshold-based metric
# ---------------------------------------------------------------------------

def tpr_at_far(
    probs:     np.ndarray,
    y_true:    np.ndarray,
    far_thr:   float = 0.5,
    obs_hours: Optional[float] = None,
    stride_s:  int   = 60,
) -> float:
    """
    Compute TPR at a given false alarm rate threshold.

    The FAR is expressed as false alarms per hour. We sweep the probability
    threshold from high to low, accumulating FP until the FAR budget is
    exhausted, then read off the TPR at that point.

    Parameters
    ----------
    probs      : predicted probabilities for positive class (M+)
    y_true     : binary ground truth (1 = M+, 0 = background)
    far_thr    : false alarm rate budget in alarms / hour (default 0.5)
    obs_hours  : total observation time in hours; if None, derived from len(probs)
    stride_s   : window stride in seconds (default 60)

    Returns
    -------
    TPR in [0, 1]
    """
    if obs_hours is None:
        obs_hours = len(probs) * stride_s / 3600.0

    max_fp = far_thr * obs_hours    # total allowed false positives

    order  = np.argsort(-probs)     # descending by probability
    y_ord  = y_true[order]

    cum_pos = np.cumsum(y_ord)
    cum_neg = np.cumsum(1 - y_ord)

    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0

    # Find the index where we reach the FP budget
    exceeds = np.where(cum_neg > max_fp)[0]
    if len(exceeds) == 0:
        # Under budget even predicting everything positive
        cut = len(y_ord) - 1
    else:
        cut = exceeds[0] - 1    # last index before exceeding budget

    if cut < 0:
        return 0.0

    tpr = cum_pos[cut] / total_pos
    return float(tpr)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def ece(
    probs:   np.ndarray,
    y_true:  np.ndarray,
    n_bins:  int = 10,
) -> float:
    """
    Expected calibration error (equal-width probability bins).
    Lower is better; a perfectly calibrated model returns 0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc  = y_true[mask].mean()
        ece_val += mask.sum() / total * abs(avg_conf - avg_acc)
    return float(ece_val)


# ---------------------------------------------------------------------------
# Full inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:  SolarPatchTST,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run model forward on the full loader, returning numpy arrays.

    Returns
    -------
    {
      "prob_15":   (N,)  — P(M+ in next 15 min)
      "prob_30":   (N,)  — P(M+ in next 30 min)
      "prob_60":   (N,)  — P(M+ in next 60 min)
      "prob_ext":  (N,)  — P(X-class in next 30 min)
      "prob_cmx":  (N,3) — P(C), P(M), P(X) multiclass
      "y_binary":  (N,3) — ground truth M+ at 15/30/60 min
      "y_extreme": (N,)  — ground truth X-class
      "y_class":   (N,)  — GOES class 0–4
    }
    """
    model.eval()
    p15, p30, p60, pext, pcmx = [], [], [], [], []
    yb, yext, yc = [], [], []

    for batch in loader:
        X = batch["X"].to(device)
        lb, le, lmc = model(X)

        pb  = torch.sigmoid(lb)     # (B, 3)
        pe  = torch.sigmoid(le)     # (B, 1)
        pmc = torch.softmax(lmc, dim=-1)  # (B, 3)

        p15.append(pb[:, 0].cpu().numpy())
        p30.append(pb[:, 1].cpu().numpy())
        p60.append(pb[:, 2].cpu().numpy())
        pext.append(pe.squeeze(-1).cpu().numpy())
        pcmx.append(pmc.cpu().numpy())

        yb.append(batch["y_binary"].numpy())
        yext.append(batch["y_extreme"].numpy())
        yc.append(batch["y_class"].numpy())

    return {
        "prob_15":  np.concatenate(p15),
        "prob_30":  np.concatenate(p30),
        "prob_60":  np.concatenate(p60),
        "prob_ext": np.concatenate(pext),
        "prob_cmx": np.concatenate(pcmx, axis=0),
        "y_binary": np.concatenate(yb,   axis=0),
        "y_extreme":np.concatenate(yext),
        "y_class":  np.concatenate(yc),
    }


# ---------------------------------------------------------------------------
# Lead time computation
# ---------------------------------------------------------------------------

def _far_operating_threshold(
    probs:     np.ndarray,
    y_true:    np.ndarray,
    far_thr:   float,
    obs_hours: float,
) -> float:
    """
    Return the probability threshold at the FAR operating point.

    This is the highest threshold before the number of false positives
    first exceeds far_thr * obs_hours — i.e. the same cut used internally
    by tpr_at_far().  Using this threshold for lead-time computation ensures
    lead time is measured at the same operating point as TPR@FAR.
    """
    max_fp  = far_thr * obs_hours
    order   = np.argsort(-probs)
    cum_neg = np.cumsum(1 - y_true[order])
    exceeds = np.where(cum_neg > max_fp)[0]
    cut     = exceeds[0] - 1 if len(exceeds) > 0 else len(probs) - 1
    return float(probs[order[max(cut, 0)]])


def compute_lead_times(
    probs:      np.ndarray,
    y_class:    np.ndarray,
    threshold:  float,
    stride_s:   int = 60,
    window_s:   int = 1800,
) -> Dict[str, float]:
    """
    Approximate lead time: for each TP alert, how many windows before the
    label switches to M+ does the model first trigger?

    Since we have overlapping windows (stride=60 s, window=1800 s), a
    'new flare' event is identified by finding runs of consecutive M+ windows.
    Lead time = (first alert index − onset index) × stride_s / 60 minutes.

    Onsets at idx=0 are skipped: no preceding windows exist to search, so
    lead time is undefined (the sequence starts already in M+ state).

    Also returns ``min_threshold_for_lead`` — the lowest threshold at which
    any onset produces a lead time > 0.  If this equals threshold, lead time
    is non-zero.  If it is far below threshold, the model's pre-onset signal
    exists but is too weak to cross the operating point.

    Returns dict with median/mean lead time, event count, and diagnostics.
    """
    alerts = (probs >= threshold).astype(int)
    m_plus = (y_class >= 3).astype(int)

    # Find onset cadences: transition from 0 → 1 in y_class
    onsets = np.where(np.diff(m_plus, prepend=0) == 1)[0]

    leads          = []
    skipped_boundary = 0

    # Diagnostic: find the minimum threshold that yields any lead time
    min_thr_for_lead: Optional[float] = None

    for onset in onsets:
        # Skip onset at the very start of the sequence — no preceding windows
        # exist, so lead time is undefined (not a model failure).
        if onset == 0:
            skipped_boundary += 1
            continue

        search_start  = max(0, onset - window_s // stride_s)
        pre_probs     = probs[search_start:onset]
        window_alerts = alerts[search_start:onset]

        if window_alerts.sum() > 0:
            first    = search_start + np.argmax(window_alerts)
            lead_min = (onset - first) * stride_s / 60.0
            leads.append(lead_min)
        else:
            # Check what minimum threshold would give lead time for this onset
            if len(pre_probs) > 0:
                max_pre = float(pre_probs.max())
                if max_pre > 0:
                    if min_thr_for_lead is None or max_pre > min_thr_for_lead:
                        min_thr_for_lead = max_pre

    result: Dict[str, object] = {
        "lead_median_min":       0.0,
        "lead_mean_min":         0.0,
        "n_detected":            0,
        "n_onsets_total":        int(len(onsets)),
        "n_onsets_skipped_boundary": int(skipped_boundary),
    }

    if min_thr_for_lead is not None:
        result["min_threshold_for_lead"] = round(min_thr_for_lead, 4)

    if not leads:
        return result

    result["lead_median_min"] = float(np.median(leads))
    result["lead_mean_min"]   = float(np.mean(leads))
    result["n_detected"]      = len(leads)
    return result


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model:       SolarPatchTST,
    loader:      DataLoader,
    device:      torch.device,
    partition:   str   = "TEST",
    stride_s:    int   = 60,
    far_thr:     float = 0.5,
    verbose:     bool  = True,
    val_caveat:  bool  = False,
    horizon_idx: int   = 1,   # 0=15min, 1=30min, 2=60min — primary head to evaluate
) -> Dict[str, float]:
    """
    Full evaluation suite for one data partition.

    horizon_idx selects the binary output head to evaluate:
      1 = 30-min head (default), 2 = 60-min head (use with --horizon 60)

    Returns a flat dict of metrics.
    """
    _PROB_KEYS = ["prob_15", "prob_30", "prob_60"]
    preds  = run_inference(model, loader, device)
    p30    = preds[_PROB_KEYS[horizon_idx]]   # primary horizon probabilities
    yb30   = preds["y_binary"][:, horizon_idx]
    yext   = preds["y_extreme"]
    yc     = preds["y_class"]

    n_windows  = len(p30)
    obs_hours  = n_windows * stride_s / 3600.0

    # Primary metric
    tpr_val = tpr_at_far(p30, yb30, far_thr=far_thr, obs_hours=obs_hours)

    # Calibration
    ece_val = ece(p30, yb30)
    ece_ext = ece(preds["prob_ext"], yext)

    # X-class recall (extreme head)
    x_mask   = yext == 1
    x_recall = preds["prob_ext"][x_mask].mean() if x_mask.any() else float("nan")

    # ROC-AUC (binary 30-min)
    try:
        from sklearn.metrics import roc_auc_score
        auc_val = float(roc_auc_score(yb30, p30)) if yb30.std() > 0 else float("nan")
    except Exception:
        auc_val = float("nan")

    # Operating threshold at FAR budget — used for both TPR and lead time.
    # Using the FAR-based threshold (not F1-optimal) ensures lead time is
    # measured at the same operating point as the headline TPR@FAR metric.
    far_op_t = _far_operating_threshold(p30, yb30, far_thr=far_thr, obs_hours=obs_hours)

    # F1-optimal threshold (kept for reference / opt_threshold field)
    try:
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresh = precision_recall_curve(yb30, p30)
        f1s   = 2 * prec * rec / np.maximum(prec + rec, 1e-6)
        opt_t = float(thresh[np.argmax(f1s[:-1])])
    except Exception:
        opt_t = far_op_t

    lead = compute_lead_times(p30, yc, threshold=far_op_t, stride_s=stride_s)

    def _f(v, n=3):
        """Round and convert to Python float for JSON safety."""
        return round(float(v), n)

    metrics = {
        "partition":               partition,
        "n_windows":               int(n_windows),
        "obs_hours":               _f(obs_hours, 1),
        "prevalence_mplus":        _f(yb30.mean()),
        "far_budget":              float(far_thr),
        f"tpr_at_far{far_thr}":    _f(tpr_val),
        "roc_auc_30min":           _f(auc_val),
        "ece_30min":               _f(ece_val),
        "ece_extreme":             _f(ece_ext),
        "x_recall":                _f(x_recall) if not np.isnan(x_recall) else None,
        "lead_median_min":         _f(lead["lead_median_min"], 1),
        "lead_mean_min":           _f(lead["lead_mean_min"], 1),
        "n_flare_events_detected": int(lead["n_detected"]),
        "n_onsets_total":          int(lead.get("n_onsets_total", 0)),
        "n_onsets_skipped_boundary": int(lead.get("n_onsets_skipped_boundary", 0)),
        "far_op_threshold":        _f(far_op_t),
        "opt_threshold":           _f(opt_t),
    }

    # Add diagnostic: minimum threshold at which any lead time is possible
    if "min_threshold_for_lead" in lead:
        metrics["min_threshold_for_lead"] = lead["min_threshold_for_lead"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluation — {partition}")
        if val_caveat:
            print("  ⚠  VAL caveat: 60.9 % M+ prevalence → metrics are optimistic")
        print(f"{'='*60}")
        print(f"  Windows       : {n_windows}  ({obs_hours:.1f} h)")
        print(f"  M+ prevalence : {100*yb30.mean():.1f} %")
        print(f"  TPR@FAR0.5/hr : {tpr_val:.3f}")
        print(f"  ROC-AUC (30m) : {auc_val:.3f}")
        print(f"  ECE (30m)     : {ece_val:.3f}")
        print(f"  X-class recall: {x_recall:.3f}" if not np.isnan(x_recall) else "  X-class recall: N/A")
        print(f"  Lead time     : median {lead['lead_median_min']:.1f} min  "
              f"mean {lead['lead_mean_min']:.1f} min  "
              f"(events: {lead['n_detected']}/{lead.get('n_onsets_total',0)-lead.get('n_onsets_skipped_boundary',0)} measurable)")
        if lead.get("n_onsets_skipped_boundary", 0) > 0:
            print(f"  ↳ {lead['n_onsets_skipped_boundary']} onset(s) at sequence start skipped "
                  f"(day starts in M+ — no preceding windows)")
        if "min_threshold_for_lead" in lead and lead["lead_median_min"] == 0.0:
            print(f"  ↳ Pre-onset signal max p={lead['min_threshold_for_lead']:.4f} "
                  f"vs FAR threshold {far_op_t:.4f} — signal present but below operating point")
        print(f"  FAR threshold : {far_op_t:.3f}  (opt-F1: {opt_t:.3f})")
        print(f"{'='*60}\n")

    return metrics
