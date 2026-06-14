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
    Binary and GOES-class labels derived from the normalised excess (excess_A)
    in the label horizon.  Cross-match against the GOES XRS catalog in
    Module 3 to replace with authoritative labels.

  Class balancing
    Flares are rare (≫100:1 ratio).  Imbalance ratio is reported per split.
    Optional oversampling of flare windows in the training partition only.
    VAL and TEST are NEVER resampled — that would inflate reported metrics.

  Leakage prevention
    Windows spanning two split partitions are excluded.
    Windows with > max_nan_frac NaN values are excluded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .split import TRAIN, VAL, TEST, GAP


WINDOW_S         = 1800
STRIDE_S         = 60
LABEL_HORIZON_S  = 1800
FLARE_EXCESS_THRESHOLD = 5.0
MAX_NAN_FRAC     = 0.30
GOES_THRESHOLDS  = {"B": 1.0, "C": 5.0, "M": 15.0, "X": 50.0}


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


def build_windows(
    features: Dict[str, np.ndarray],
    split_array: np.ndarray,
    excess_A: Optional[np.ndarray] = None,
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
    split_array     : (n_times,) uint8 from split.build_split_array()
    excess_A        : (n_times,) normalised excess (label signal)
    window_s        : window length in seconds
    stride_s        : stride between windows in seconds
    label_horizon_s : look-ahead horizon in seconds
    flare_threshold : excess threshold → binary flare label
    max_nan_frac    : reject windows with more NaN than this fraction
    cadence_s       : seconds per sample
    """
    window_len  = int(window_s  / cadence_s)
    stride      = max(1, int(stride_s / cadence_s))
    horizon_len = int(label_horizon_s / cadence_s)

    feature_names = sorted(features.keys())
    n_features    = len(feature_names)
    n_times       = len(split_array)

    # Feature matrix (n_times, n_features) in float32
    F = np.stack(
        [features[f].astype(np.float32) for f in feature_names], axis=1
    )

    # Label signal
    if excess_A is None:
        excess_A = features.get("excess_A", np.zeros(n_times, dtype=float))
    excess_A = np.asarray(excess_A, dtype=float)

    # Precompute forward maximum excess for labelling (O(n) deque algorithm)
    max_future = _rolling_max_future_deque(excess_A, horizon_len)

    # Precompute per-cadence split and NaN info for fast window rejection
    # NaN fraction per window: sum NaN indicator over window / window_len
    F_isnan = np.isnan(F)  # (n_times, n_features)
    nan_indicator = F_isnan.mean(axis=1).astype(np.float32)  # (n_times,)

    # Prefix sum of NaN indicator for O(1) window NaN query
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

        # ── Split check: window must lie within one non-GAP partition ──
        # Use mode: find majority non-GAP split in the window
        window_splits = split_array[start:end]
        # Fast check using unique values
        uniq = np.unique(window_splits)
        non_gap = uniq[uniq != GAP]
        if len(non_gap) != 1:
            continue
        partition = int(non_gap[0])

        # ── NaN fraction check (O(1) using prefix sums) ──
        window_nan_frac = (cs_nan[end] - cs_nan[start]) / window_len
        if window_nan_frac > max_nan_frac:
            continue

        # ── Label: peak excess in the horizon after this window ──
        # max_future[end-1] = max(excess_A[end-1 : end-1+horizon_len])
        # but we precomputed max_future[i] = max(excess_A[i : i+horizon_len])
        # so we want max_future[end] not end-1 (the label horizon AFTER the window)
        label_idx = min(end, n_times - 1)
        peak = float(max_future[label_idx]) if label_idx + horizon_len <= n_times else np.nan

        if np.isnan(peak):
            continue

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
        imbalance_ratio = _compute_imbalance(yb_arr, sp_arr),
    )


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
