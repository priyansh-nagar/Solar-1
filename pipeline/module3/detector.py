"""
detector.py — Multi-channel adaptive flare nowcaster (Module 3b + 3c)
======================================================================

Design: two-channel voter
--------------------------
  Soft trigger  : excess_A > adaptive_excess_threshold(rolling_stats_of_excess_A)
  Hard trigger  : band_C   > adaptive_band_C_threshold(rolling_stats_of_band_C)

Why Band C as the hard channel (not hardness_ratio Band C / Band A)?
---------------------------------------------------------------------
• During a flare, Band A (1.5-3 keV) rises ~60×, Band C (>3 keV) rises ~10×.
  The ratio Band C / Band A therefore DECREASES during a flare — wrong sign for
  a detection trigger.
• Band C in raw cts/s is zero during quiet periods and clearly non-zero during
  real hard X-ray emission.  Any sustained Band C signal above the rolling noise
  floor is genuine astrophysics.
• A cosmic ray hits Band C for a single cadence; a real flare sustains for ≥30 s.

The two-channel voter:
  detection = sustained(excess_A_trigger AND band_C_trigger, ≥30 s)

Additional physics constraints
-------------------------------
  • 30-second minimum sustain: real flare onsets hold for ≥ 30 s
  • 5-minute minimum gap between distinct events
  • Positive derivative_1s required in the first 60 s of each event
    (real flares always rise; the detector doesn't see them declining first)
  • Quality-flag filter: saturated cadences never count as detections OR misses

Output
------
  List[FlareEvent] where each event has:
    onset_idx    : index into the SoLEXS time array of first trigger
    peak_idx     : index of maximum excess_A in the event
    end_idx      : index of last trigger cadence
    peak_excess_A: float
    goes_class   : "A"/"B"/"C"/"M"/"X"/"?" from GOES cross-match
    goes_class_int: 0-4 / -1
    confidence   : float [0, 1] — fraction of event cadences above both thresholds
    lead_time_s  : float — seconds from onset_idx to peak_idx (pre-peak signal)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .threshold import compute_all_thresholds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NowcasterConfig:
    """Tunable parameters for the nowcaster."""
    # Excess_A threshold parameters
    base_sigma:       float = 3.0    # σ for excess_A threshold
    floor_excess_A:   float = 1.0    # absolute minimum threshold in excess_A units
    excess_A_window_s: int  = 300    # rolling window for excess_A baseline (5 min)

    # Band C (hard channel) threshold parameters
    band_C_sigma:     float = 3.0    # σ for Band C threshold
    band_C_floor:     float = 1.0    # absolute minimum threshold in cts/s
    band_C_window_s:  int   = 900    # rolling window for Band C baseline (15 min)

    # Sustain and gap rules
    min_sustain_s:    int   = 30     # both channels must agree for ≥ 30 s
    min_gap_s:        int   = 300    # 5-min minimum between distinct events

    # Rise verification
    require_positive_deriv:  bool = True   # derivative_1s > 0 in onset window
    deriv_check_window_s:    int  = 60     # look for rise in first N seconds

    # Quality
    skip_saturated:   bool = True    # ignore saturated cadences entirely

    # Detector prefix
    prefix:           str  = "solexs_sdd2"


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class FlareEvent:
    """One detected flare event."""
    onset_idx:      int
    peak_idx:       int
    end_idx:        int
    onset_time:     str           # ISO-8601 UTC string
    peak_time:      str
    end_time:       str
    peak_excess_A:  float
    goes_class:     str           # "A"/"B"/"C"/"M"/"X"/"?"
    goes_class_int: int           # 0-4 / -1
    confidence:     float         # fraction of event cadences above threshold
    lead_time_s:    float         # onset → peak in seconds
    duration_s:     float         # onset → end in seconds
    detection_flags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_flares(
    preprocess_ds,
    goes_class_1s: Optional[np.ndarray] = None,
    empirical_thresholds: Optional[dict] = None,
    config: Optional[NowcasterConfig] = None,
    cadence_s: float = 1.0,
) -> List[FlareEvent]:
    """
    Run the two-channel adaptive nowcaster on a preprocessed Dataset.

    Parameters
    ----------
    preprocess_ds         : xr.Dataset from module2.preprocess_day()
    goes_class_1s         : (n_times,) int8 from CrossMatchResult.goes_class_1s
                            Pass None to use SoLEXS-only labels.
    empirical_thresholds  : dict from CrossMatchResult.thresholds
                            Pass None to use first-principles thresholds.
    config                : NowcasterConfig (uses defaults if None)
    cadence_s             : seconds per sample (default 1.0)

    Returns
    -------
    List[FlareEvent]
    """
    if config is None:
        config = NowcasterConfig()

    # ── Extract arrays ────────────────────────────────────────────────────
    thr_dict = compute_all_thresholds(
        preprocess_ds,
        prefix         = config.prefix,
        base_sigma     = config.base_sigma,
        floor_excess_A = config.floor_excess_A,
        band_C_sigma   = config.band_C_sigma,
        band_C_floor   = config.band_C_floor,
        cadence_s      = cadence_s,
    )

    excess_A  = thr_dict["raw_excess_A"]
    band_C    = thr_dict["raw_band_C"]
    deriv     = thr_dict["raw_deriv"]
    quality   = thr_dict["quality"]
    thr_exc   = thr_dict["excess_A"]
    thr_C     = thr_dict["band_C"]

    n = len(excess_A)

    # ── Build quality mask (usable = in GTI, not saturated) ──────────────
    from pipeline.module1.quality import is_usable, QFlag
    if quality is not None:
        usable = is_usable(quality)
        if config.skip_saturated:
            not_sat = (quality & QFlag.SATURATED) == 0
            usable  = usable & not_sat
    else:
        usable = np.ones(n, dtype=bool)

    # ── Channel triggers ─────────────────────────────────────────────────
    # Soft channel: excess_A
    valid_exc = usable & ~np.isnan(excess_A)
    soft_on   = valid_exc & (excess_A > thr_exc)

    # Hard channel: Band C count rate
    if band_C is not None and thr_C is not None:
        band_C_nn  = np.maximum(band_C, 0.0)           # no negatives
        valid_C    = usable & ~np.isnan(band_C)
        hard_on    = valid_C & (band_C_nn > thr_C)
    else:
        logger.warning(
            "Band C not available — running single-channel (excess_A only). "
            "Expect higher false-alarm rate."
        )
        hard_on = soft_on   # degrade gracefully

    # Both channels must agree
    both_on = soft_on & hard_on

    # ── Find sustained runs (min_sustain_s consecutive both_on cadences) ──
    min_sustain = max(1, int(config.min_sustain_s / cadence_s))
    sustained   = _find_sustained_runs(both_on, min_sustain)

    # ── Merge events within min_gap_s of each other ────────────────────────
    min_gap = max(1, int(config.min_gap_s / cadence_s))
    merged  = _merge_runs(sustained, min_gap)

    # ── Build FlareEvent objects ──────────────────────────────────────────
    times_utc = preprocess_ds.coords["time"].values    # datetime64[ns]
    events: List[FlareEvent] = []

    for (start, end) in merged:
        # Rise verification: at least one positive derivative in first deriv_window
        if config.require_positive_deriv and deriv is not None:
            dw = int(config.deriv_check_window_s / cadence_s)
            check_end = min(start + dw, end + 1)
            rise_ok = np.any(deriv[start:check_end] > 0)
            if not rise_ok:
                logger.debug(
                    "Event [%d, %d] rejected: no positive derivative in onset window",
                    start, end,
                )
                continue

        # Peak: cadence with maximum excess_A in event window
        window_excess = excess_A[start : end + 1].copy()
        window_excess[np.isnan(window_excess)] = -np.inf
        local_peak = int(np.argmax(window_excess))
        peak_idx   = start + local_peak
        peak_val   = float(excess_A[peak_idx]) if not np.isnan(excess_A[peak_idx]) else 0.0

        # Confidence: fraction of event cadences where both channels agree
        event_len  = end - start + 1
        n_both_on  = int(both_on[start : end + 1].sum())
        confidence = n_both_on / max(event_len, 1)

        # GOES class from cross-match
        if goes_class_1s is not None and peak_idx < len(goes_class_1s):
            gc_int = int(goes_class_1s[peak_idx])
        else:
            gc_int = -1

        gc_str = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}.get(gc_int, "?")

        # Fallback: infer class from empirical thresholds
        if gc_int == -1 and empirical_thresholds is not None:
            gc_str, gc_int = _infer_class_from_excess(
                peak_val, empirical_thresholds
            )

        lead_time   = float((peak_idx - start) * cadence_s)
        duration    = float((end - start + 1)  * cadence_s)

        onset_time = _idx_to_iso(times_utc, start)
        peak_time  = _idx_to_iso(times_utc, peak_idx)
        end_time   = _idx_to_iso(times_utc, end)

        events.append(FlareEvent(
            onset_idx      = start,
            peak_idx       = peak_idx,
            end_idx        = end,
            onset_time     = onset_time,
            peak_time      = peak_time,
            end_time       = end_time,
            peak_excess_A  = peak_val,
            goes_class     = gc_str,
            goes_class_int = gc_int,
            confidence     = round(confidence, 4),
            lead_time_s    = lead_time,
            duration_s     = duration,
            detection_flags = {
                "soft_trigger_count": int(soft_on[start:end+1].sum()),
                "hard_trigger_count": int(hard_on[start:end+1].sum()),
                "both_trigger_count": n_both_on,
            },
        ))

    logger.info(
        "Nowcaster: %d flare events detected on %d GTI-valid cadences",
        len(events), int(usable.sum()),
    )

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sustained_runs(
    trigger: np.ndarray,
    min_sustain: int,
) -> List[tuple[int, int]]:
    """
    Find contiguous runs of True in *trigger* that last ≥ min_sustain samples.

    Returns list of (start, end) index pairs (inclusive).
    """
    runs: List[tuple[int, int]] = []
    n = len(trigger)
    i = 0
    while i < n:
        if trigger[i]:
            j = i
            while j < n and trigger[j]:
                j += 1
            if (j - i) >= min_sustain:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _merge_runs(
    runs: List[tuple[int, int]],
    min_gap: int,
) -> List[tuple[int, int]]:
    """
    Merge consecutive runs whose gap is strictly less than min_gap samples.

    Gap is measured as: start[i+1] - end[i].
    Runs separated by exactly min_gap or more are NOT merged.

    Returns merged list of (start, end) pairs.
    """
    if not runs:
        return []
    merged: List[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end          # number of samples between runs
        if gap < min_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _infer_class_from_excess(
    peak_excess_A: float,
    thresholds: dict,
) -> tuple[str, int]:
    """
    Infer GOES class from peak excess_A using empirical thresholds.
    Falls back to first-principles if empirical data is absent.
    """
    # Use p50 thresholds as class boundaries
    boundaries = []
    for cls in ["B", "C", "M", "X"]:
        p50 = thresholds.get(f"{cls}_p50", np.nan)
        if not np.isnan(p50):
            boundaries.append((cls, p50))

    if not boundaries:
        # First principles
        for cls, lo in [("X", 50.0), ("M", 15.0), ("C", 5.0), ("B", 1.0)]:
            if peak_excess_A >= lo:
                return cls, {"B": 1, "C": 2, "M": 3, "X": 4}[cls]
        return "A", 0

    # Walk down from the highest class
    for cls, thr in sorted(boundaries, key=lambda x: -x[1]):
        if peak_excess_A >= thr:
            return cls, {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}[cls]
    return "B", 1


def _idx_to_iso(times_utc: np.ndarray, idx: int) -> str:
    """Convert an index into the SoLEXS time array to an ISO-8601 string."""
    if 0 <= idx < len(times_utc):
        t = times_utc[idx]
        return str(t).replace("T", " ")[:19] + " UTC"
    return "unknown"
