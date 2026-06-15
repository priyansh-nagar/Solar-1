"""
windows.py — Sliding window dataset builder
============================================
Converts the continuous light-curve features into fixed-length windows
suitable for sequence models (LSTM, TCN, Transformer).

Design decisions
----------------
  Window length : 30 minutes (1800 samples at 1-s cadence)
    Captures pre-flare quiet background (~20 min) plus onset/early rise
    (~10 min) within one window.

  Stride        : configurable (default 60 s)
    A 60-s stride gives ~97% window overlap, greatly increasing training
    examples for rare flare classes.

  Label strategy
    TWO modes controlled by label_source:

    label_source="excess_A"  (default, backward-compatible)
      Binary and class labels derived from peak excess_A in the label
      horizon.  Uses first-principles GOES thresholds.  Kept for
      compatibility with Module 2 tests only.  DO NOT use for Module 4.

    label_source="goes_class"  ← USE THIS FOR MODULE 4
      Labels derived from the per-cadence goes_class integer array
      added by module3.goes_crossmatch.build_goes_labels().
      For each window, the label = max(goes_class[horizon]) i.e. the
      highest GOES class in the 30 min after the window ends.
      y_binary = int(max_class >= 2)  (C-class and above = flare)
      This is the ONLY correct labelling: the excess_A proxy had X=0
      (detector saturated) and used wrong threshold values (C=5.0 vs
      empirical C=0.15).

  Class balancing
    Flares are rare (>100:1 ratio).  Imbalance ratio is reported per split.
    Optional oversampling of flare windows in the training partition only.
    VAL and TEST are NEVER resampled — that would inflate reported metrics.

  Leakage prevention
    Windows spanning two split partitions are excluded.
    Windows with > max_nan_frac NaN values are excluded.
    goes_class is stripped from the feature matrix F — it is a label,
    not an input feature.  Including it would be data leakage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .split import TRAIN, VAL, TEST, GAP


WINDOW_S         = 1800
STRIDE_S         = 60
LABEL_HORIZON_S  = 1800
FLARE_EXCESS_THRESHOLD = 5.0
MAX_NAN_FRAC     = 0.30

# First-principles GOES thresholds for excess_A mode (LEGACY — do not use for Module 4)
GOES_THRESHOLDS  = {"B": 1.0, "C": 5.0, "M": 15.0, "X": 50.0}

# GOES class integer for binary label boundary: C-class and above = flare
GOES_FLARE_CLASS_MIN = 2   # C-class

# Feature columns to strip before building F (they are labels, not inputs)
_LABEL_COLS = frozenset({"goes_class"})


@dataclass
class WindowDataset:
    """Container for the sliding window dataset."""
    X             : np.ndarray        # (n_windows, window_len, n_features) float32
    y_binary      : np.ndarray        # (n_windows,) int8
    y_class       : np.ndarray        # (n_windows,) int8
    splits        : np.ndarray        # (n_windows,) uint8
    window_starts : np.ndarray        # (n_windows,) int64
    feature_names : List[str]
    cadence_s     : float = 1.0
    window_s      : int   = WINDOW_S
    label_horizon_s: int  = LABEL_HORIZON_S
    label_source  : str   = "excess_A"
    imbalance_ratio: Dict[str, float] = field(default_factory=dict)

    def get_partition(
        self,
        partition: int,
        balance: bool = False,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y_binary, y_class) for *partition*.

        Parameters
        ----------
        partition : TRAIN / VAL / TEST
        balance   : if True and partition == TRAIN, oversample minority class.
                    Never balance VAL or TEST.
        """
        mask = self.splits == partition
        X_p  = self.X[mask]
        yb_p = self.y_binary[mask]
        yc_p = self.y_class[mask]

        if balance and partition == TRAIN and len(X_p) > 0:
            X_p, yb_p, yc_p = _oversample(X_p, yb_p, yc_p, seed=seed)

        return X_p, yb_p, yc_p

    def class_breakdown(self) -> dict:
        """Return per-class window counts and fractions for each split."""
        cls_names = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}
        result = {}
        for part, name in [(TRAIN, "train"), (VAL, "val"), (TEST, "test")]:
            mask = self.splits == part
            yc = self.y_class[mask]
            counts = {}
            for i in range(5):
                n = int((yc == i).sum())
                if n > 0:
                    counts[cls_names[i]] = n
            result[name] = counts
        return result


def build_windows(
    features: Dict[str, np.ndarray],
    split_array: np.ndarray,
    excess_A: Optional[np.ndarray] = None,
    label_source: Literal["excess_A", "goes_class"] = "excess_A",
    window_s: int          = WINDOW_S,
    stride_s: int          = STRIDE_S,
    label_horizon_s: int   = LABEL_HORIZON_S,
    flare_threshold: float = FLARE_EXCESS_THRESHOLD,
    max_nan_frac: float    = MAX_NAN_FRAC,
    cadence_s: float       = 1.0,
) -> WindowDataset:
    """
    Build a WindowDataset from feature arrays and split labels.

    Parameters
    ----------
    features        : {feature_name: (n_times,) array}
                      Must contain "goes_class" if label_source="goes_class".
                      "goes_class" is automatically stripped from the feature
                      matrix — it is a label, not an input.
    split_array     : (n_times,) uint8 from split.build_split_array()
    excess_A        : (n_times,) normalised excess — used only when
                      label_source="excess_A"
    label_source    : "excess_A" (legacy) or "goes_class" (correct for Mod 4)
    window_s        : window length in seconds
    stride_s        : stride between windows in seconds
    label_horizon_s : look-ahead horizon in seconds
    flare_threshold : excess threshold → binary flare label (excess_A mode only)
    max_nan_frac    : reject windows with more NaN than this fraction
    cadence_s       : seconds per sample
    """
    if label_source not in ("excess_A", "goes_class"):
        raise ValueError(
            f"label_source must be 'excess_A' or 'goes_class', got {label_source!r}"
        )

    window_len  = int(window_s  / cadence_s)
    stride      = max(1, int(stride_s / cadence_s))
    horizon_len = int(label_horizon_s / cadence_s)

    # Strip label columns before building the feature matrix
    feature_names = sorted(k for k in features.keys() if k not in _LABEL_COLS)
    n_times       = len(split_array)

    # Feature matrix (n_times, n_features) in float32
    F = np.stack(
        [features[f].astype(np.float32) for f in feature_names], axis=1
    )

    # ── Label signal ──────────────────────────────────────────────────────
    if label_source == "goes_class":
        _label_array, _label_fn = _setup_goes_class_labels(features, n_times)
    else:
        # Legacy excess_A mode
        if excess_A is None:
            excess_A = features.get("excess_A", np.zeros(n_times, dtype=float))
        excess_A = np.asarray(excess_A, dtype=float)
        _label_array = excess_A
        _label_fn = None

    # Precompute forward maximum label for each position (O(n) deque)
    # For goes_class: max future class in [-1,4]; treat -1 as 0
    if label_source == "goes_class":
        # Replace -1 (unknown) with 0 (A-class background) for the horizon max
        gc_clean = np.where(_label_array < 0, 0, _label_array).astype(float)
        max_future = _rolling_max_future_deque(gc_clean, horizon_len)
    else:
        max_future = _rolling_max_future_deque(_label_array, horizon_len)

    # NaN fraction per window: prefix-sum approach for O(1) per window
    F_isnan = np.isnan(F)
    nan_indicator = F_isnan.mean(axis=1).astype(np.float32)
    cs_nan = np.concatenate([[0.0], np.cumsum(nan_indicator)])

    X_list:  List[np.ndarray] = []
    yb_list: List[int] = []
    yc_list: List[int] = []
    sp_list: List[int] = []
    ws_list: List[int] = []

    max_start = n_times - window_len - horizon_len + 1
    if max_start <= 0:
        raise RuntimeError(
            f"Dataset too short ({n_times} samples) for window_len={window_len} "
            f"+ horizon_len={horizon_len}."
        )

    for start in range(0, max_start, stride):
        end = start + window_len

        # ── Split check ───────────────────────────────────────────────────
        window_splits = split_array[start:end]
        uniq    = np.unique(window_splits)
        non_gap = uniq[uniq != GAP]
        if len(non_gap) != 1:
            continue
        partition = int(non_gap[0])

        # ── NaN fraction check ───────────────────────────────────────────
        window_nan_frac = (cs_nan[end] - cs_nan[start]) / window_len
        if window_nan_frac > max_nan_frac:
            continue

        # ── Label: peak in the horizon after this window ──────────────────
        label_idx = min(end, n_times - 1)
        peak = (
            float(max_future[label_idx])
            if label_idx + horizon_len <= n_times
            else np.nan
        )

        if np.isnan(peak):
            continue

        if label_source == "goes_class":
            y_class  = int(peak)                          # already an int class
            y_binary = int(y_class >= GOES_FLARE_CLASS_MIN)
        else:
            y_binary = int(peak >= flare_threshold)
            y_class  = _goes_class_label(peak)

        X_list.append(F[start:end].copy())
        yb_list.append(y_binary)
        yc_list.append(y_class)
        sp_list.append(partition)
        ws_list.append(start)

    if not X_list:
        raise RuntimeError(
            "No valid windows found. Check that usable data spans at least "
            f"{window_s + label_horizon_s} seconds and split fractions are "
            "appropriate for the dataset length."
        )

    X_arr  = np.stack(X_list,  axis=0)
    yb_arr = np.array(yb_list, dtype=np.int8)
    yc_arr = np.array(yc_list, dtype=np.int8)
    sp_arr = np.array(sp_list, dtype=np.uint8)
    ws_arr = np.array(ws_list, dtype=np.int64)

    return WindowDataset(
        X               = X_arr,
        y_binary        = yb_arr,
        y_class         = yc_arr,
        splits          = sp_arr,
        window_starts   = ws_arr,
        feature_names   = feature_names,
        cadence_s       = cadence_s,
        window_s        = window_s,
        label_horizon_s = label_horizon_s,
        label_source    = label_source,
        imbalance_ratio = _compute_imbalance(yb_arr, sp_arr),
    )


# ---------------------------------------------------------------------------
# goes_class label setup helper
# ---------------------------------------------------------------------------

def _setup_goes_class_labels(
    features: Dict[str, np.ndarray],
    n_times: int,
) -> tuple:
    """
    Validate and extract the goes_class array from features.

    Returns (goes_class_array, None).  Raises if goes_class is missing.
    """
    if "goes_class" not in features:
        raise KeyError(
            "label_source='goes_class' requires 'goes_class' in features dict. "
            "Call module3.goes_crossmatch.build_goes_labels(pds, date) first "
            "to add per-cadence GOES labels to the Dataset."
        )
    gc = np.asarray(features["goes_class"], dtype=np.int8)
    if len(gc) != n_times:
        raise ValueError(
            f"goes_class length {len(gc)} != n_times {n_times}"
        )
    return gc, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_max_future_deque(arr: np.ndarray, horizon: int) -> np.ndarray:
    """
    Compute forward-looking rolling maximum: out[i] = max(arr[i : i+horizon]).
    Uses a monotone deque for O(n) time.

    Positions where the full horizon window extends beyond the array end
    use a partial window.  Positions at index >= n-1 get arr[n-1].
    """
    n   = len(arr)
    out = np.full(n, np.nan)
    safe = np.where(np.isnan(arr), -np.inf, arr)

    dq: deque = deque()   # stores indices, kept in decreasing safe[idx] order

    # Traverse in reverse so we build the window [i, i+horizon)
    for i in range(n - 1, -1, -1):
        # Evict indices that have fallen outside the window [i, i+horizon)
        while dq and dq[-1] >= i + horizon:
            dq.pop()
        # Maintain monotone decreasing order: remove indices with smaller values
        while dq and safe[dq[0]] <= safe[i]:
            dq.popleft()
        dq.appendleft(i)

        front_val = safe[dq[-1]] if dq else -np.inf
        out[i] = front_val if front_val != -np.inf else np.nan

    return out


def _goes_class_label(peak_excess: float) -> int:
    """Map peak excess_A to GOES class integer (excess_A mode only)."""
    if peak_excess >= GOES_THRESHOLDS["X"]: return 4
    if peak_excess >= GOES_THRESHOLDS["M"]: return 3
    if peak_excess >= GOES_THRESHOLDS["C"]: return 2
    if peak_excess >= GOES_THRESHOLDS["B"]: return 1
    return 0


def _compute_imbalance(y_binary: np.ndarray, splits: np.ndarray) -> Dict[str, float]:
    result = {}
    for part, name in [(TRAIN, "train"), (VAL, "val"), (TEST, "test")]:
        mask = splits == part
        n = int(mask.sum())
        if n == 0:
            result[name] = 0.0
            continue
        result[name] = round(float(y_binary[mask].sum()) / n, 4)
    return result


def _oversample(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_class: np.ndarray,
    target_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat minority (flare) windows until flare fraction ≥ target_ratio."""
    rng = np.random.default_rng(seed)
    n_flare = int(y_binary.sum())
    n_quiet = int((y_binary == 0).sum())

    if n_flare == 0 or n_quiet == 0:
        return X, y_binary, y_class

    if n_flare / (n_flare + n_quiet) >= target_ratio:
        return X, y_binary, y_class

    n_target = int(np.ceil(target_ratio * n_quiet / (1.0 - target_ratio)))
    n_extra  = n_target - n_flare
    flare_idx = np.where(y_binary == 1)[0]
    extra_idx = rng.choice(flare_idx, size=n_extra, replace=True)

    X_aug  = np.concatenate([X,        X[extra_idx]])
    yb_aug = np.concatenate([y_binary, y_binary[extra_idx]])
    yc_aug = np.concatenate([y_class,  y_class[extra_idx]])

    perm = rng.permutation(len(X_aug))
    return X_aug[perm], yb_aug[perm], yc_aug[perm]
