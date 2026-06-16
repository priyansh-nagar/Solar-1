"""
dataset.py — PyTorch Dataset for Module 4 training
====================================================
Memory budget (7470 windows total, 2.1 GB available):
  TRAIN  4885 × 1800 × 29 × 4 B  = 1.02 GB  (loaded once, scales in place)
  VAL    1381 × 1800 × 29 × 4 B  = 0.29 GB
  TEST   1204 × 1800 × 29 × 4 B  = 0.25 GB
  Total  ≈ 1.56 GB  — fits in 2.1 GB

X is loaded from the mmap in 256-window chunks and stored as a torch.Tensor
in memory.  __getitem__ is then O(1) with no disk I/O.
StandardScaler is fitted on a 500-window random subsample of TRAIN.
NaN → 0 fill is applied during loading (neutral = training mean after z-score).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from pipeline.module2.split import TRAIN, VAL, TEST


# ---------------------------------------------------------------------------
# Feature stream definitions  (matched by substring)
# ---------------------------------------------------------------------------

SOFT_FEAT_PATTERNS = (
    "band_A", "total", "residual_A", "excess_A",
    "derivative_1s", "derivative_60s", "rate_of_rise",
    "softness_ratio", "rolling_std", "cumulative_excess",
)
HARD_FEAT_PATTERNS = (
    "band_B", "band_C", "band_D",
    "residual_B", "residual_C", "residual_D",
    "excess_B", "excess_C", "excess_D",
    "hardness_ratio",
)


def resolve_stream_indices(
    feature_names: List[str],
) -> Tuple[List[int], List[int]]:
    """
    Return (soft_indices, hard_indices) for the two attention streams.
    HARD patterns take priority; every feature ends up in exactly one stream.
    """
    soft_idx, hard_idx = [], []
    for i, name in enumerate(feature_names):
        if any(p in name for p in HARD_FEAT_PATTERNS):
            hard_idx.append(i)
        else:
            soft_idx.append(i)
    return soft_idx, hard_idx


# ---------------------------------------------------------------------------
# Bulk-load helper (RAM-safe chunked loading + in-place scaling)
# ---------------------------------------------------------------------------

def _load_partition_x(
    x_path:  Path,
    indices: np.ndarray,
    mean:    np.ndarray,       # (F,) float32
    std:     np.ndarray,       # (F,) float32
    chunk:   int = 256,
) -> torch.Tensor:
    """
    Load a partition's X windows from the mmap file in *chunk*-sized
    blocks, apply NaN fill + z-score in place, and return a float32
    torch.Tensor of shape (N, T, F).

    Peak extra RAM per call: chunk × T × F × 4 bytes ≈ 53 MB (chunk=256).
    """
    X_mmap = np.load(x_path, mmap_mode="r")
    N = len(indices)
    T, F = X_mmap.shape[1], X_mmap.shape[2]

    buf = np.empty((N, T, F), dtype=np.float32)

    for s in range(0, N, chunk):
        e   = min(s + chunk, N)
        raw = np.array(X_mmap[indices[s:e]], dtype=np.float32)   # copy from mmap
        nm  = np.isnan(raw)
        raw[nm] = 0.0
        raw = (raw - mean) / std
        raw[nm] = 0.0
        buf[s:e] = raw

    # torch.from_numpy shares memory with buf — zero copy
    return torch.from_numpy(buf)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SolarFlareDataset(Dataset):
    """
    PyTorch Dataset wrapping one partition of the Module 4 v2 arrays.

    All data lives in RAM after __init__ — __getitem__ is O(1) with no I/O.

    Parameters
    ----------
    x_path        : path to X_v2.npy (read via mmap during init).
                    Pass None only when *_preloaded_X* is provided.
    indices       : 1-D array of global row indices for this partition.
                    Pass None only when *_preloaded_X* is provided.
    y_binary_30   : (N_total,) int8 — M+ at 30 min
    y_class       : (N_total,) int8 — GOES class 0–4
    scaler        : fitted StandardScaler
    feature_names : list of 29 feature names in column order
    y_binary_15   : optional (N_total,) int8 — M+ at 15 min
    y_binary_60   : optional (N_total,) int8 — M+ at 60 min
    y_extreme     : optional (N_total,) int8 — X-class at 30 min
    verbose       : print loading progress
    _preloaded_X  : (N, T, F) float32 torch.Tensor — skip disk loading entirely.
                    Used for unit tests and synthetic-data runs where no .npy file exists.
                    When provided, x_path and indices are ignored.
    """

    def __init__(
        self,
        x_path:        Optional[Path],
        indices:       Optional[np.ndarray],
        y_binary_30:   np.ndarray,
        y_class:       np.ndarray,
        scaler:        StandardScaler,
        feature_names: List[str],
        y_binary_15:   Optional[np.ndarray] = None,
        y_binary_60:   Optional[np.ndarray] = None,
        y_extreme:     Optional[np.ndarray] = None,
        verbose:       bool = True,
        _preloaded_X:  Optional[torch.Tensor] = None,
    ):
        self.feature_names = feature_names
        self.soft_idx, self.hard_idx = resolve_stream_indices(feature_names)

        # ── Per-item scaler arrays (shared for all items — tiny) ─────────
        mean = scaler.mean_.astype(np.float32)
        std  = np.sqrt(scaler.var_).clip(1e-8).astype(np.float32)

        # ── Bulk-load X into RAM (or use preloaded tensor for tests) ─────
        if _preloaded_X is not None:
            # Unit test / synthetic path: X is already in memory, already scaled.
            self.X = _preloaded_X.float()
            N = len(self.X)
            indices = np.arange(N)
        else:
            if x_path is None or indices is None:
                raise ValueError(
                    "Either x_path + indices must be provided, or _preloaded_X must be set."
                )
            n_mb = len(indices) * 1800 * 29 * 4 // (1024 ** 2)
            if verbose:
                print(f"    Loading {len(indices)} windows ({n_mb} MB) …", end=" ", flush=True)
            self.X = _load_partition_x(x_path, indices, mean, std)   # (N, 1800, 29)
            if verbose:
                print("OK")

        # ── Labels (small — plain in-memory tensors) ─────────────────────
        y15  = y_binary_15[indices] if y_binary_15 is not None else y_binary_30[indices]
        y30  = y_binary_30[indices]
        y60  = y_binary_60[indices] if y_binary_60 is not None else y_binary_30[indices]
        yext = y_extreme[indices]   if y_extreme   is not None else (y_class[indices] >= 4).astype(np.int8)
        yc   = y_class[indices]

        yb = np.stack([y15, y30, y60], axis=1).astype(np.float32)
        self.y_binary  = torch.from_numpy(yb)
        self.y_extreme = torch.from_numpy(yext.astype(np.float32))
        self.y_class   = torch.from_numpy(yc.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "X":         self.X[i],
            "y_binary":  self.y_binary[i],
            "y_extreme": self.y_extreme[i],
            "y_class":   self.y_class[i],
        }


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_splits(
    data_dir:     str | Path = "/tmp",
    batch_size:   int        = 32,
    num_workers:  int        = 0,
    scaler_n_bg:  int        = 500,
    seed:         int        = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, List[str]]:
    """
    Build three DataLoaders and a fitted StandardScaler.

    Scaler is fitted on a *scaler_n_bg*-window random subsample of TRAIN
    to avoid loading the full 1.56 GB X array for fitting alone.

    Returns
    -------
    (train_loader, val_loader, test_loader, scaler, feature_names)
    """
    data_dir = Path(data_dir)
    x_path   = data_dir / "X_v2.npy"

    # ── Load small label arrays fully into RAM (< 100 KB each) ───────────
    yb30   = np.load(data_dir / "y_binary_v2.npy")
    yc     = np.load(data_dir / "y_class_v2.npy")
    splits = np.load(data_dir / "splits_v2.npy")
    yb15   = _try_load(data_dir / "y_15min_v2.npy")
    yb60   = _try_load(data_dir / "y_60min_v2.npy")
    yext   = _try_load(data_dir / "y_extreme_v2.npy")

    feat_path = data_dir / "feature_names_v2.json"
    feature_names = (
        json.load(open(feat_path)) if feat_path.exists()
        else [f"feat_{i}" for i in range(29)]
    )

    # ── Partition indices ─────────────────────────────────────────────────
    idx_tr  = np.where(splits == TRAIN)[0]
    idx_val = np.where(splits == VAL)[0]
    idx_te  = np.where(splits == TEST)[0]

    # ── Fit StandardScaler on TRAIN subsample ────────────────────────────
    rng     = np.random.default_rng(seed)
    bg_idx  = rng.choice(idx_tr, min(scaler_n_bg, len(idx_tr)), replace=False)
    X_mmap  = np.load(x_path, mmap_mode="r")

    print(f"Fitting StandardScaler on {len(bg_idx)} TRAIN windows …")
    X_bg    = np.array(X_mmap[bg_idx], dtype=np.float32)      # ≈104 MB
    N_bg, T, F = X_bg.shape
    X_flat  = X_bg.reshape(N_bg * T, F)
    nm      = np.isnan(X_flat)
    X_flat[nm] = 0.0
    scaler  = StandardScaler()
    scaler.fit(X_flat)
    del X_bg, X_flat                                           # free immediately

    # ── Build datasets (bulk-load each partition) ─────────────────────────
    common = dict(
        x_path=x_path, y_binary_30=yb30, y_class=yc,
        scaler=scaler, feature_names=feature_names,
        y_binary_15=yb15, y_binary_60=yb60, y_extreme=yext,
    )

    print("Loading partitions:")
    print("  TRAIN", end=""); ds_tr  = SolarFlareDataset(**common, indices=idx_tr)
    print("  VAL",   end=""); ds_val = SolarFlareDataset(**common, indices=idx_val)
    print("  TEST",  end=""); ds_te  = SolarFlareDataset(**common, indices=idx_te)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    tr_loader  = DataLoader(ds_tr,  shuffle=True,  **kw)
    val_loader = DataLoader(ds_val, shuffle=False, **kw)
    te_loader  = DataLoader(ds_te,  shuffle=False, **kw)

    return tr_loader, val_loader, te_loader, scaler, feature_names


def _try_load(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None
