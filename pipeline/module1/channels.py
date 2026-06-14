"""
channels.py — Channel-to-energy calibration and science band extraction
========================================================================
SoLEXS is a Silicon Drift Detector (SDD) with 340 channels covering
approximately 1.5–7 keV.  HEL1OS uses CdZnTe and NaI(Tl) detectors in the
~15–150 keV range.

Without a redistribution matrix (RMF) file the calibration is linear:

    E(ch) = E_min + ch × gain     [keV]

where gain = (E_max − E_min) / N_channels.

SCIENCE ENERGY BANDS (4 bands, NUMBAND=4 in .lc header):
  SoLEXS:
    Band A  1.5 – 3.0 keV   soft thermal (quiet-sun & early flare rise)
    Band B  3.0 – 5.0 keV   intermediate
    Band C  5.0 – 7.0 keV   hotter / impulsive phase
    Band D  > 7.0  keV      above nominal range → summed as spillover

  HEL1OS (placeholder — update with actual RMF when available):
    Band A  15  –  30 keV
    Band B  30  –  60 keV
    Band C  60  – 100 keV
    Band D  100 – 150 keV

IMPORTANT: The linear calibration is an approximation.  When an official
response matrix (CALDB RMF) is available, replace _linear_energy_keV() with
a proper MATRIX-based conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Instrument calibration constants
# CAUTION: These are best-estimate values derived from ISSDC documentation
# and the detector physics of Si-SDD (SoLEXS) / CdZnTe (HEL1OS).
# Always cross-check against an official CALDB/RMF when it is available.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorCalibration:
    n_channels: int
    e_min_keV: float       # energy at channel 0
    e_max_keV: float       # energy at last channel
    science_bands: Dict[str, Tuple[float, float]]  # name → (lo_keV, hi_keV)

    @property
    def gain_keV_per_channel(self) -> float:
        return (self.e_max_keV - self.e_min_keV) / self.n_channels

    def channel_to_energy(self, channels: np.ndarray) -> np.ndarray:
        """Return energy in keV for an array of channel numbers (float ok)."""
        return self.e_min_keV + channels.astype(float) * self.gain_keV_per_channel

    def energy_to_channel(self, energy_keV: float) -> int:
        """Return the nearest channel index for a given energy (keV)."""
        ch = (energy_keV - self.e_min_keV) / self.gain_keV_per_channel
        return int(np.clip(round(ch), 0, self.n_channels - 1))

    def band_channel_range(self, band_name: str) -> Tuple[int, int]:
        """Return (ch_lo, ch_hi) inclusive for a named science band."""
        lo_keV, hi_keV = self.science_bands[band_name]
        ch_lo = self.energy_to_channel(lo_keV)
        ch_hi = self.energy_to_channel(hi_keV)
        return ch_lo, ch_hi


# SoLEXS SDD: 340 channels, ~1.5–7.0 keV
# Source: ISRO ISSDC SoLEXS Level-2 documentation (approximate linear fit)
SOLEXS_CALIBRATION = DetectorCalibration(
    n_channels=340,
    e_min_keV=1.5,
    e_max_keV=7.0,
    science_bands={
        "A": (1.5, 3.0),    # soft thermal — quiet-sun & early flare rise
        "B": (3.0, 4.5),    # intermediate
        "C": (4.5, 6.0),    # hotter impulsive phase
        "D": (6.0, 6.95),   # highest SoLEXS channels (capped inside detector range)
    },
)

# HEL1OS: placeholder calibration; update when CALDB available
# Source: ISRO ISSDC HEL1OS documentation (estimate)
HEL1OS_CALIBRATION = DetectorCalibration(
    n_channels=1024,
    e_min_keV=15.0,
    e_max_keV=150.0,
    science_bands={
        "A": (15.0,  30.0),
        "B": (30.0,  60.0),
        "C": (60.0, 100.0),
        "D": (100.0, 150.0),
    },
)


# ---------------------------------------------------------------------------
# Channel array utilities
# ---------------------------------------------------------------------------

def energy_axis(cal: DetectorCalibration) -> np.ndarray:
    """Return the energy (keV) at the centre of each channel."""
    channels = np.arange(cal.n_channels, dtype=float)
    return cal.channel_to_energy(channels)


def extract_band_counts(
    counts: np.ndarray,           # shape (n_times, n_channels)
    band_name: str,
    cal: DetectorCalibration,
) -> np.ndarray:
    """
    Sum counts over all channels belonging to *band_name*.

    Parameters
    ----------
    counts  : (n_times, n_channels) array.  May contain NaNs (bad cadences).
    band_name : one of the keys in cal.science_bands
    cal     : DetectorCalibration for this instrument

    Returns
    -------
    (n_times,) array of total counts in the band.  Rows where all channels
    are NaN remain NaN; rows where only some channels are NaN use nansum
    (missing channels treated as zero — annotated in quality flags).
    """
    ch_lo, ch_hi = cal.band_channel_range(band_name)
    band_slice = counts[:, ch_lo : ch_hi + 1]

    # Fully-NaN rows → NaN output; partial-NaN rows → nansum (conservative)
    all_nan = np.all(np.isnan(band_slice), axis=1)
    result = np.nansum(band_slice, axis=1).astype(float)
    result[all_nan] = np.nan
    return result


def extract_all_bands(
    counts: np.ndarray,
    cal: DetectorCalibration,
) -> Dict[str, np.ndarray]:
    """
    Extract science bands for all named bands in cal.science_bands.

    Returns a dict  band_name → (n_times,) count array.
    """
    return {
        band: extract_band_counts(counts, band, cal)
        for band in cal.science_bands
    }


def total_counts(counts: np.ndarray) -> np.ndarray:
    """Sum over all channels.  NaN rows stay NaN."""
    all_nan = np.all(np.isnan(counts), axis=1)
    result = np.nansum(counts, axis=1).astype(float)
    result[all_nan] = np.nan
    return result


# ---------------------------------------------------------------------------
# Saturation detection
# ---------------------------------------------------------------------------

def flag_saturated_rows(
    counts: np.ndarray,           # (n_times, n_channels)
    saturation_cts_per_channel: float = 1e6,
    saturation_total_fraction: float = 0.80,
) -> np.ndarray:
    """
    Return a boolean mask (True = saturated) based on two criteria:

    1. Any single channel exceeds *saturation_cts_per_channel*
       (detector register overflow; values would wrap or saturate at the ADC
       maximum — for SoLEXS this is instrument-specific but 10^6 cts/s/channel
       is an extremely conservative upper bound).

    2. More than *saturation_total_fraction* of all non-NaN channels carry
       identical counts (pile-up / detector lock-up signature).

    IMPORTANT: These thresholds are conservative defaults.  For X9+ flares
    the SoLEXS count rate can reach ~10^4–10^5 cts/s total; adjust
    saturation_cts_per_channel with empirical detector characterisation.
    """
    # Criterion 1: any channel overflows
    overflow = np.any(counts > saturation_cts_per_channel, axis=1)

    # Criterion 2: pathological uniformity across channels (lock-up)
    n_times, n_ch = counts.shape
    lockup = np.zeros(n_times, dtype=bool)
    for t in range(n_times):
        row = counts[t]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        # If >80% of channels have the same non-zero value → suspect
        vals, cnts = np.unique(valid[valid > 0], return_counts=True)
        if len(cnts) > 0 and cnts.max() / len(valid) > saturation_total_fraction:
            lockup[t] = True

    return overflow | lockup
