"""
split.py — Temporal train/validation/test split
=================================================
CRITICAL: All splits are strictly chronological.  Never random.

Why temporal splitting matters for solar physics ML
----------------------------------------------------
Solar activity follows ~11-year cycles and multi-day active-region lifetimes.
A random split would put cadences from the same active region (and the same
flare precursor pattern) into both training and test sets, causing the model
to memorise the specific event rather than learn generalisable precursors.

More subtly: flux values at time t are correlated with flux at t+1, t+2, …
(autocorrelation timescale ~minutes for quiet sun, hours during active periods).
A random split places test samples within the training autocorrelation window,
making the model look better than it will be in production.

Correct approach: train only on the PAST.  Validate on the NEAR FUTURE.
Test on the FAR FUTURE.

Split fractions: 70 / 15 / 15
------------------------------
  70% train  — enough data to learn the full solar-cycle range of backgrounds
  15% val    — large enough to give stable metric estimates
  15% test   — held out until final evaluation; never used for hyperparameter tuning

Inter-split gap
---------------
A short gap (default: 30 minutes) between each split boundary prevents
context leakage when the model uses rolling windows: a 30-min window at the
start of validation cannot peek into the end of training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Integer labels used in the split DataArray
TRAIN = 0
VAL   = 1
TEST  = 2
GAP   = 3    # cadences in the inter-split gap — excluded from all sets


@dataclass
class SplitBoundaries:
    """Index boundaries for each split partition."""
    n_total       : int
    train_end     : int      # exclusive
    gap1_end      : int      # exclusive (train/val gap)
    val_end       : int      # exclusive
    gap2_end      : int      # exclusive (val/test gap)
    test_end      : int      # exclusive (= n_total)
    gap_samples   : int
    train_frac    : float
    val_frac      : float
    test_frac     : float


def compute_split_boundaries(
    n_samples: int,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    test_frac: float  = 0.15,
    gap_s: float      = 1800.0,   # 30-minute inter-split gap
    cadence_s: float  = 1.0,
) -> SplitBoundaries:
    """
    Compute integer index boundaries for a 70/15/15 temporal split.

    Parameters
    ----------
    n_samples   : total number of cadences
    train_frac  : fraction of samples assigned to training
    val_frac    : fraction of samples assigned to validation
    test_frac   : fraction of samples assigned to test
    gap_s       : gap duration between splits in seconds
    cadence_s   : cadence in seconds

    Returns
    -------
    SplitBoundaries with inclusive index ranges for each partition.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
        )

    gap_samples = int(np.ceil(gap_s / cadence_s))

    # Subtract two gap regions from the total before dividing
    usable = n_samples - 2 * gap_samples
    if usable <= 0:
        raise ValueError(
            f"n_samples={n_samples} too small for 2 gaps of {gap_samples} samples each"
        )

    train_n = int(np.floor(usable * train_frac))
    val_n   = int(np.floor(usable * val_frac))
    test_n  = usable - train_n - val_n   # remainder goes to test

    train_end = train_n
    gap1_end  = train_end + gap_samples
    val_end   = gap1_end  + val_n
    gap2_end  = val_end   + gap_samples
    test_end  = n_samples   # cap at total

    return SplitBoundaries(
        n_total     = n_samples,
        train_end   = train_end,
        gap1_end    = gap1_end,
        val_end     = val_end,
        gap2_end    = gap2_end,
        test_end    = test_end,
        gap_samples = gap_samples,
        train_frac  = train_frac,
        val_frac    = val_frac,
        test_frac   = test_frac,
    )


def build_split_array(bounds: SplitBoundaries) -> np.ndarray:
    """
    Build a (n_total,) uint8 array with labels TRAIN / VAL / TEST / GAP.

    Use this array to index into any aligned feature array to select
    the correct partition without risk of temporal leakage.
    """
    split = np.full(bounds.n_total, GAP, dtype=np.uint8)
    split[: bounds.train_end]               = TRAIN
    split[bounds.gap1_end : bounds.val_end] = VAL
    split[bounds.gap2_end : bounds.test_end]= TEST
    return split


def split_summary(split_array: np.ndarray) -> dict:
    """Human-readable partition summary."""
    return {
        "train_samples" : int((split_array == TRAIN).sum()),
        "val_samples"   : int((split_array == VAL).sum()),
        "test_samples"  : int((split_array == TEST).sum()),
        "gap_samples"   : int((split_array == GAP).sum()),
        "total"         : len(split_array),
    }


def mask_for_split(
    split_array: np.ndarray,
    partition: int,
    usable_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a boolean mask selecting cadences in *partition* that are also usable.

    Parameters
    ----------
    split_array  : output of build_split_array()
    partition    : one of TRAIN, VAL, TEST
    usable_mask  : optional boolean mask from quality.is_usable(); if given,
                   only usable cadences within the partition are selected.
    """
    mask = split_array == partition
    if usable_mask is not None:
        mask = mask & usable_mask
    return mask
