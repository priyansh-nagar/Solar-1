"""
features.py — Feature engineering for solar flare detection
=============================================================
Computes physics-motivated features from the preprocessed light curves.
All features are computed on smoothed, background-subtracted arrays.

Feature catalogue
-----------------
  flux_smooth_{B}         Savitzky-Golay smoothed flux for band B
  background_{B}          SNIP background estimate for band B
  residual_{B}            flux_smooth - background (absolute excess)
  excess_{B}              residual / (background + ε)  (normalised)
  derivative_1s           d(flux_smooth_A)/dt at 1-s resolution
  derivative_60s          smoothed first derivative over 60-s window
  rate_of_rise            mean positive derivative over a 5-min window
  hardness_ratio          Band C / (Band A + ε)  — spectral hardness proxy
  softness_ratio          Band A / (Band B + ε)  — spectral softness proxy
  rolling_std_5min        rolling σ of flux_smooth_A over 5-min window
  rolling_std_15min       rolling σ of flux_smooth_A over 15-min window
  cumulative_excess        integral of positive excess (proxy for flare fluence)

Physics rationale
-----------------
  derivative_1s / rate_of_rise:
    Flare onset is characterised by a rapid rise.  The derivative detects this
    even before the absolute count rate reaches a significance threshold.
    Precursor studies (Benz 2008, Warmuth & Mann 2020) show that small-scale
    energy releases 5–30 min before the main flare produce subtle rate-of-rise
    signals in soft X-ray Band A.

  hardness_ratio (Band C / Band A):
    During the impulsive phase, hot plasma (T > 10 MK) brightens Band C
    disproportionately, causing hardness to spike before Band A reaches its
    peak.  This gives ~60-120 s of additional precursor lead time.

  rolling_std:
    Background variance is low during quiet periods.  Elevated variance
    without a clear positive derivative indicates uncertain activity —
    useful as a negative-confidence feature for the classifier.

Performance note
----------------
All rolling operations use fully vectorised numpy cumsum-based implementations.
For n=86400 samples and w=900 window, naive Python loops take ~60 s.
The vectorised versions run in <0.1 s.
"""

from __future__ import annotations

import numpy as np
from typing import Dict


EPSILON = 1e-6
_5MIN_S  = 300
_15MIN_S = 900
_60S     = 60


def compute_features(
    band_smooth: Dict[str, np.ndarray],
    band_background: Dict[str, np.ndarray],
    cadence_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute all features from smoothed and background arrays.

    Parameters
    ----------
    band_smooth     : {band_name: (n_times,) smoothed flux}
    band_background : {band_name: (n_times,) SNIP background}
    cadence_s       : seconds per sample

    Returns
    -------
    features : {feature_name: (n_times,) array}
    """
    feats: Dict[str, np.ndarray] = {}
    n = next(iter(band_smooth.values())).shape[0]

    # ── Per-band residual and normalised excess ──
    for band, smooth in band_smooth.items():
        bg   = band_background.get(band, np.zeros(n))
        resid = smooth - bg
        excess = resid / (np.abs(bg) + EPSILON)
        feats[f"flux_smooth_{band}"]  = smooth
        feats[f"background_{band}"]   = bg
        feats[f"residual_{band}"]     = resid
        feats[f"excess_{band}"]       = excess

    # ── Derivatives on Band A (primary flare channel) ──
    band_a = band_smooth.get("A", np.full(n, np.nan))

    deriv_1s = _safe_diff(band_a, cadence_s)
    feats["derivative_1s"] = deriv_1s

    # 60-second causal rolling mean of the 1-s derivative
    w_60 = max(1, int(_60S / cadence_s))
    feats["derivative_60s"] = _rolling_mean_fast(deriv_1s, w_60)

    # Rate of rise: causal mean of POSITIVE derivative over 5-min window
    w_5m = max(1, int(_5MIN_S / cadence_s))
    feats["rate_of_rise"] = _rate_of_rise_fast(deriv_1s, w_5m)

    # ── Spectral ratios ──
    band_c = band_smooth.get("C", np.full(n, np.nan))
    band_b = band_smooth.get("B", np.full(n, np.nan))

    feats["hardness_ratio"] = np.where(
        np.isnan(band_a) | np.isnan(band_c),
        np.nan,
        band_c / (band_a + EPSILON),
    )
    feats["softness_ratio"] = np.where(
        np.isnan(band_a) | np.isnan(band_b),
        np.nan,
        band_a / (band_b + EPSILON),
    )

    # ── Rolling standard deviations ──
    w_15m = max(3, int(_15MIN_S / cadence_s))
    feats["rolling_std_5min"]  = _rolling_std_fast(band_a, w_5m)
    feats["rolling_std_15min"] = _rolling_std_fast(band_a, w_15m)

    # ── Cumulative excess (running integral of positive excess in Band A) ──
    excess_a = feats["excess_A"]
    pos_excess = np.where(np.isnan(excess_a) | (excess_a < 0), 0.0, excess_a)
    feats["cumulative_excess"] = np.nancumsum(pos_excess) * cadence_s

    return feats


# ---------------------------------------------------------------------------
# Vectorised rolling statistics  (all O(n) time, O(n) memory)
# ---------------------------------------------------------------------------

def _rolling_mean_fast(
    arr: np.ndarray,
    window: int,
    min_frac: float = 0.5,
) -> np.ndarray:
    """
    Causal (backward-looking) rolling mean using the cumsum trick.
    Requires at least min_frac × window valid (non-NaN) samples.

    O(n) time and memory — no Python loops.
    """
    valid = ~np.isnan(arr)
    x   = np.where(valid, arr, 0.0)
    cnt = valid.astype(np.float64)

    # Prefix sums (length n+1, index 0 = 0)
    cs_x   = np.empty(len(arr) + 1); cs_x[0]   = 0.0; np.cumsum(x,   out=cs_x[1:])
    cs_cnt = np.empty(len(arr) + 1); cs_cnt[0] = 0.0; np.cumsum(cnt, out=cs_cnt[1:])

    n     = len(arr)
    ends  = np.arange(1, n + 1)
    starts = np.maximum(0, ends - window)

    sums = cs_x[ends]   - cs_x[starts]
    cnts = cs_cnt[ends] - cs_cnt[starts]

    min_count = max(1, int(window * min_frac))
    out = np.where(cnts >= min_count, sums / np.maximum(cnts, 1.0), np.nan)
    return out.astype(np.float64)


def _rolling_std_fast(
    arr: np.ndarray,
    window: int,
    min_count: int = 3,
) -> np.ndarray:
    """
    Causal rolling standard deviation (Bessel-corrected) via cumsum.
    O(n) time — no Python loops.
    """
    valid = ~np.isnan(arr)
    x   = np.where(valid, arr,    0.0)
    x2  = np.where(valid, arr**2, 0.0)
    cnt = valid.astype(np.float64)

    cs_x   = np.empty(len(arr) + 1); cs_x[0]   = 0.0; np.cumsum(x,   out=cs_x[1:])
    cs_x2  = np.empty(len(arr) + 1); cs_x2[0]  = 0.0; np.cumsum(x2,  out=cs_x2[1:])
    cs_cnt = np.empty(len(arr) + 1); cs_cnt[0] = 0.0; np.cumsum(cnt, out=cs_cnt[1:])

    n     = len(arr)
    ends  = np.arange(1, n + 1)
    starts = np.maximum(0, ends - window)

    sums  = cs_x[ends]   - cs_x[starts]
    sums2 = cs_x2[ends]  - cs_x2[starts]
    cnts  = cs_cnt[ends] - cs_cnt[starts]

    enough = cnts >= min_count
    c      = np.maximum(cnts, 1.0)
    # Population variance via E[X²] − E[X]²
    var_pop = sums2 / c - (sums / c) ** 2
    # Bessel correction: s² = n/(n-1) × σ²
    var     = np.where(enough, var_pop * c / np.maximum(c - 1.0, 1.0), np.nan)
    var     = np.maximum(var, 0.0)          # guard against −ε rounding
    return np.where(enough, np.sqrt(var), np.nan)


def _rate_of_rise_fast(
    deriv: np.ndarray,
    window: int,
    min_data_frac: float = 0.5,
) -> np.ndarray:
    """
    Causal mean of POSITIVE derivative values over a rolling window.

    Returns 0.0 where there are enough data points but no positive derivatives.
    Returns NaN where there are too few data points (< min_data_frac × window).
    O(n) time — no Python loops.
    """
    # Contributions: only positive, non-NaN values
    pos     = np.where(np.isnan(deriv) | (deriv <= 0), 0.0, deriv)
    pos_cnt = (~np.isnan(deriv) & (deriv > 0)).astype(np.float64)
    all_cnt = (~np.isnan(deriv)).astype(np.float64)

    cs_pos  = np.empty(len(deriv) + 1); cs_pos[0]  = 0.0; np.cumsum(pos,     out=cs_pos[1:])
    cs_pcnt = np.empty(len(deriv) + 1); cs_pcnt[0] = 0.0; np.cumsum(pos_cnt, out=cs_pcnt[1:])
    cs_acnt = np.empty(len(deriv) + 1); cs_acnt[0] = 0.0; np.cumsum(all_cnt, out=cs_acnt[1:])

    n      = len(deriv)
    ends   = np.arange(1, n + 1)
    starts = np.maximum(0, ends - window)

    sums   = cs_pos[ends]  - cs_pos[starts]
    pcnts  = cs_pcnt[ends] - cs_pcnt[starts]
    acnts  = cs_acnt[ends] - cs_acnt[starts]

    min_data = max(1, int(window * min_data_frac))
    has_data = acnts >= min_data

    # Mean of positive derivatives; 0.0 if none are positive in the window
    # Guard denominator to avoid divide-by-zero warning (np.where evaluates
    # both branches before selecting — replace zeros with 1 in the divisor).
    safe_pcnts = np.where(pcnts > 0, pcnts, 1.0)
    out = np.where(
        has_data,
        np.where(pcnts > 0, sums / safe_pcnts, 0.0),
        np.nan,
    )
    return out.astype(np.float64)


# ---------------------------------------------------------------------------
# Gradient utility
# ---------------------------------------------------------------------------

def _safe_diff(arr: np.ndarray, cadence_s: float) -> np.ndarray:
    """
    First finite difference with NaN propagation, computed per valid run.
    Uses numpy.gradient (second-order central differences at interior points,
    first-order at edges) applied independently to each contiguous valid run.
    """
    out = np.full_like(arr, np.nan, dtype=float)
    valid = ~np.isnan(arr)

    if valid.sum() < 2:
        return out

    x = np.arange(len(arr), dtype=float) * cadence_s

    # Find contiguous valid runs
    padded = np.concatenate([[False], valid, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where(padded[:-1] & ~padded[1:])[0]   # exclusive

    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        seg = arr[s:e].astype(float)
        t   = x[s:e]
        out[s:e] = np.gradient(seg, t)

    return out
