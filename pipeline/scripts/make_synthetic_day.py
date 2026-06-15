"""
make_synthetic_day.py — generate a synthetic SoLEXS flare day for pipeline testing
====================================================================================
Creates a minimal but physics-faithful FITS directory under /tmp/solexs_data/
that Module 1 ingest_day() and Module 2 preprocess_day() can read without error.

Usage:
    python pipeline/scripts/make_synthetic_day.py [--n-hours 4] [--out-dir /tmp/solexs_data]

The synthetic day has:
  • Background Band A ~30 cts/s (quiet-Sun level)
  • Flare 1 at t=1.0 h: X-class  (peak Band A ~1500 cts/s → excess_A ≈ 50×)
  • Flare 2 at t=2.5 h: M-class  (peak Band A ~450  cts/s → excess_A ≈ 15×)
  • Flare 3 at t=3.5 h: C-class  (peak Band A ~180  cts/s → excess_A ≈  5×)
  • GTI = full range minus one 5-min gap at t=2.0 h (tests gap-fill code)
  • SDD2 present; SDD1 absent (mirrors real post-launch data)
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
from pathlib import Path

import numpy as np
import astropy.io.fits as fits

# ── FITS epoch: MJD 40587 = Unix epoch (1970-01-01 00:00:00 UTC) ──────────
MJD_EPOCH_UNIX = 0.0   # TSTART values are already in Unix seconds in our mock

N_CHANNELS = 340
E_MIN_KEV  = 1.5
E_MAX_KEV  = 7.0
GAIN       = (E_MAX_KEV - E_MIN_KEV) / N_CHANNELS   # keV/channel

# Band A: 1.5–3.0 keV
CH_A_LO = 0
CH_A_HI = int((3.0 - E_MIN_KEV) / GAIN)   # ≈ 93

# Band B: 3.0–4.5 keV
CH_B_LO = CH_A_HI
CH_B_HI = int((4.5 - E_MIN_KEV) / GAIN)   # ≈ 185

# Band C: 4.5–6.0 keV
CH_C_LO = CH_B_HI
CH_C_HI = int((6.0 - E_MIN_KEV) / GAIN)   # ≈ 278

# Band D: 6.0–6.95 keV
CH_D_LO = CH_C_HI
CH_D_HI = int((6.95 - E_MIN_KEV) / GAIN)  # ≈ 339


# ---------------------------------------------------------------------------
# Flare light-curve model (FRED: Fast Rise Exponential Decay)
# ---------------------------------------------------------------------------

def flare_profile(
    t: np.ndarray,
    t_peak: float,
    peak_count_rate: float,
    rise_s: float = 300.0,
    decay_s: float = 900.0,
) -> np.ndarray:
    """Return a FRED profile centred on t_peak, in counts/s."""
    out = np.zeros_like(t, dtype=float)
    # Rise phase: Gaussian half
    rise_mask = t < t_peak
    if rise_mask.any():
        out[rise_mask] = peak_count_rate * np.exp(
            -0.5 * ((t[rise_mask] - t_peak) / rise_s) ** 2
        )
    # Decay phase: exponential
    decay_mask = ~rise_mask
    if decay_mask.any():
        out[decay_mask] = peak_count_rate * np.exp(
            -(t[decay_mask] - t_peak) / decay_s
        )
    return out


# ---------------------------------------------------------------------------
# Synthetic spectrum generator
# ---------------------------------------------------------------------------

def make_spectrum(
    t: np.ndarray,
    bg_band_a: float = 30.0,
    flares: list | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a (n_times, 340) count spectrum.

    Background is distributed ~exponentially across channels (soft spectrum).
    Each flare adds counts mainly to Band A/B (thermal bremsstrahlung).
    """
    rng = np.random.default_rng(seed)
    n = len(t)

    # Background: soft-spectrum allocation across bands
    # Roughly: A gets 45%, B 30%, C 15%, D 5%, rest scattered
    bg_per_channel = np.zeros(N_CHANNELS, dtype=float)
    channels = np.arange(N_CHANNELS, dtype=float)
    energy   = E_MIN_KEV + channels * GAIN

    # Thermal bremsstrahlung-ish decay with photon energy
    weight = np.exp(-energy / 2.5)
    weight /= weight.sum()
    bg_total = bg_band_a / weight[CH_A_LO:CH_A_HI].sum()
    bg_per_channel = weight * bg_total

    # (n_times, 340) Poisson-sampled background
    counts = rng.poisson(
        np.outer(np.ones(n), bg_per_channel)
    ).astype(np.float64)

    # Flare contributions
    if flares:
        for fl in flares:
            profile = flare_profile(t, **fl)    # (n_times,) counts/s for Band A

            # Flare adds proportionally harder spectrum than quiet sun
            fl_weight = np.exp(-energy / 4.0)
            fl_weight /= fl_weight.sum()
            fl_per_channel_at_peak = fl_weight * (fl["peak_count_rate"] / fl_weight[CH_A_LO:CH_A_HI].sum())

            counts += np.outer(profile, fl_per_channel_at_peak / fl["peak_count_rate"])  \
                      * profile[:, None]

        # Recompute: add proper flare counts
        # Reset and redo cleanly
        counts = rng.poisson(
            np.outer(np.ones(n), bg_per_channel)
        ).astype(np.float64)

        for fl in flares:
            profile = flare_profile(t, **fl)    # (n_times,) in cts/s for Band A
            # Scale: fraction of peak that goes to each channel
            fl_weight = np.exp(-energy / 3.5)
            fl_weight /= fl_weight.sum()
            band_a_sum = fl_weight[CH_A_LO:CH_A_HI].sum()
            fl_per_channel = fl_weight / band_a_sum   # normalised so band_A = 1.0

            counts += rng.poisson(
                np.outer(profile, fl_per_channel).clip(0)
            ).astype(np.float64)

    return counts.astype(np.float64)


# ---------------------------------------------------------------------------
# FITS builders
# ---------------------------------------------------------------------------

def build_pi_fits(
    tstart: np.ndarray,
    counts: np.ndarray,
) -> fits.HDUList:
    """Build a .pi HDUList matching the ISSDC SoLEXS Level-1 format."""
    n = len(tstart)

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "SoLEXS"
    primary.header["CONTENT"]  = "Type II PHA file"
    primary.header["FILTER"]   = "SDD2"
    primary.header["TELESCOP"] = "Aditya-L1"
    primary.header["ORIGIN"]   = "SYNTHETIC"

    # Build SPECTRUM extension
    col_tstart   = fits.Column(name="TSTART",   format="D", array=tstart.astype(np.float64))
    col_telapse  = fits.Column(name="TELAPSE",  format="D", array=np.ones(n, dtype=np.float64))
    col_specnum  = fits.Column(name="SPEC_NUM", format="J", array=np.arange(1, n + 1, dtype=np.int32))
    col_channel  = fits.Column(name="CHANNEL",  format=f"{N_CHANNELS}K",
                               array=np.tile(np.arange(N_CHANNELS, dtype=np.int32), (n, 1)))
    col_counts   = fits.Column(name="COUNTS",   format=f"{N_CHANNELS}D", array=counts)
    col_exposure = fits.Column(name="EXPOSURE", format="D", array=np.ones(n, dtype=np.float64))

    spec_hdu = fits.BinTableHDU.from_columns(
        [col_tstart, col_telapse, col_specnum, col_channel, col_counts, col_exposure]
    )
    spec_hdu.name = "SPECTRUM"
    spec_hdu.header["INSTRUME"] = "SoLEXS"
    spec_hdu.header["FILTER"]   = "SDD2"
    spec_hdu.header["CONTENT"]  = "Type II PHA file"
    spec_hdu.header["DETCHANS"] = N_CHANNELS

    return fits.HDUList([primary, spec_hdu])


def build_gti_fits(
    intervals: list[tuple[float, float]],
) -> fits.HDUList:
    """Build a .gti HDUList with one or more GTI intervals."""
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "SoLEXS"
    primary.header["CONTENT"]  = "GOOD TIME INTERVAL"
    primary.header["FILTER"]   = "SDD2"

    starts = np.array([s for s, _ in intervals], dtype=np.float64)
    stops  = np.array([e for _, e in intervals], dtype=np.float64)

    col_start = fits.Column(name="START", format="D", array=starts)
    col_stop  = fits.Column(name="STOP",  format="D", array=stops)

    gti_hdu = fits.BinTableHDU.from_columns([col_start, col_stop])
    gti_hdu.name = "GTI"
    gti_hdu.header["INSTRUME"] = "SoLEXS"
    gti_hdu.header["CONTENT"]  = "GOOD TIME INTERVAL"

    return fits.HDUList([primary, gti_hdu])


def write_fits_gz(hdul: fits.HDUList, path: Path) -> None:
    buf = io.BytesIO()
    hdul.writeto(buf, overwrite=True)
    buf.seek(0)
    with gzip.open(path, "wb") as f:
        f.write(buf.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_day(
    out_dir: Path,
    date_tag: str = "20240222",   # synthetic X6.4 day
    n_hours: int = 4,
    seed: int = 42,
) -> Path:
    """
    Create a synthetic AL1_SLX_L1_{date_tag}_v1.0/ day directory.

    Returns the path to the created day directory.
    """
    n_times = n_hours * 3600
    # TSTART: Unix seconds starting at midnight 2024-02-22
    t0_unix = 1708560000.0   # 2024-02-22 00:00:00 UTC
    tstart  = t0_unix + np.arange(n_times, dtype=np.float64)

    # Background Band A ~30 cts/s
    bg = 30.0

    # Flares scaled so excess_A = (peak_A - bg) / (bg + ε) ≈ target
    # e.g. excess_A ≥ 50 requires peak_A ≥ bg*(50+1) = 1530 cts/s
    flares = [
        {  # X-class at 1.0 h
            "t_peak":          t0_unix + 1.0 * 3600,
            "peak_count_rate": bg * 55,   # excess_A ≈ 54 → X-class label
            "rise_s":  300.0,
            "decay_s": 1200.0,
        },
        {  # M-class at 2.5 h
            "t_peak":          t0_unix + 2.5 * 3600,
            "peak_count_rate": bg * 20,   # excess_A ≈ 19 → M-class label
            "rise_s":  200.0,
            "decay_s":  800.0,
        },
        {  # C-class at 3.5 h
            "t_peak":          t0_unix + 3.5 * 3600,
            "peak_count_rate": bg * 8,    # excess_A ≈  7 → C-class label
            "rise_s":  120.0,
            "decay_s":  400.0,
        },
    ]

    counts = make_spectrum(tstart, bg_band_a=bg, flares=flares, seed=seed)

    # GTI: full range minus a 5-min gap at t=2.0 h (tests gap-handling)
    gap_start = t0_unix + 2.0 * 3600
    gap_end   = gap_start + 300.0
    gti_intervals = [
        (tstart[0],   gap_start - 1),
        (gap_end + 1, tstart[-1]),
    ]

    # Build directory
    day_name = f"AL1_SLX_L1_{date_tag}_v1.0"
    sdd2_dir = out_dir / day_name / "SDD2"
    sdd2_dir.mkdir(parents=True, exist_ok=True)

    base = f"AL1_SLX_SDD2_L1_{date_tag}"
    pi_path  = sdd2_dir / f"{base}.pi.gz"
    gti_path = sdd2_dir / f"{base}.gti.gz"

    pi_hdul  = build_pi_fits(tstart, counts)
    gti_hdul = build_gti_fits(gti_intervals)

    write_fits_gz(pi_hdul,  pi_path)
    write_fits_gz(gti_hdul, gti_path)

    pi_mb  = pi_path.stat().st_size  / 1_048_576
    gti_kb = gti_path.stat().st_size / 1024

    print(f"Created {day_name}/SDD2/")
    print(f"  {pi_path.name:<40} {pi_mb:.1f} MB")
    print(f"  {gti_path.name:<40} {gti_kb:.1f} KB")
    print(f"  n_times={n_times}  n_flares=3  bg_A={bg} cts/s")
    print(f"  Expected excess_A peaks: ~54 (X), ~19 (M), ~7 (C)")

    return out_dir / day_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hours",  type=int,  default=4)
    parser.add_argument("--out-dir",  default="/tmp/solexs_data")
    parser.add_argument("--date-tag", default="20240222")
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_dir = make_day(
        out_dir  = out_dir,
        date_tag = args.date_tag,
        n_hours  = args.n_hours,
        seed     = args.seed,
    )
    print(f"\nRun:  python pipeline/scripts/aggregate_days.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
