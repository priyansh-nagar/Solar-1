"""
formats.py — FITS file format detection and loading
====================================================
Auto-detects SoLEXS vs HEL1OS header structure, discovers all product files
(.pi, .lc, .gti) in a directory, and provides raw HDU access.

Key differences between the two instruments (from ISSDC headers):
  SoLEXS  — INSTRUME = 'SoLEXS',  DETCHANS = 340, energy ~1.5–7 keV
  HEL1OS  — INSTRUME = 'HEL1OS',  energy ~15–150 keV, different channel count
"""

from __future__ import annotations

import enum
import gzip
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import astropy.io.fits as fits


# ---------------------------------------------------------------------------
# Public enums and types
# ---------------------------------------------------------------------------

class Instrument(str, enum.Enum):
    SOLEXS = "SoLEXS"
    HEL1OS = "HEL1OS"
    UNKNOWN = "UNKNOWN"


class ProductKind(str, enum.Enum):
    SPECTRUM = "pi"       # per-second PHA spectrum (340 channels × 1 s)
    LIGHTCURVE = "lc"     # total-flux light curve (1 column)
    GTI = "gti"           # good time intervals


@dataclass
class FITSProduct:
    """Container for one FITS product file (opened lazily)."""
    path: Path
    instrument: Instrument
    detector: str          # e.g. "SDD1", "SDD2", "CdZnTe", ...
    kind: ProductKind
    primary_header: Dict   # copy of PRIMARY HDU header keywords
    _hdul: Optional[fits.HDUList] = field(default=None, repr=False)

    def hdul(self) -> fits.HDUList:
        """Open (and cache) the HDU list.  Works for both plain and .gz files."""
        if self._hdul is None:
            self._hdul = _open_fits(self.path)
        return self._hdul

    def close(self) -> None:
        if self._hdul is not None:
            self._hdul.close()
            self._hdul = None

    def __repr__(self) -> str:
        return (
            f"FITSProduct({self.instrument.value}/{self.detector} "
            f"{self.kind.value} @ {self.path.name})"
        )


# ---------------------------------------------------------------------------
# Instrument detection
# ---------------------------------------------------------------------------

def detect_instrument(header: fits.Header) -> Instrument:
    """
    Determine which Aditya-L1 instrument produced this FITS file.

    Checks PRIMARY HDU header keyword INSTRUME (case-insensitive).
    Falls back to filename heuristics if the keyword is absent.
    """
    instrume = str(header.get("INSTRUME", "")).strip().upper()
    if "SOLEXS" in instrume:
        return Instrument.SOLEXS
    if "HEL1OS" in instrume or "HELIOS" in instrume:
        return Instrument.HEL1OS
    return Instrument.UNKNOWN


def _detect_product_kind(header: fits.Header, path: Path) -> ProductKind:
    """
    Infer product type from the CONTENT keyword or file extension.

    CONTENT values observed in ISSDC Level-1 products:
      'LIGHT CURVE'       → .lc
      'Type II PHA file'  → .pi  (per-second spectrum)
      'OGIP PHA data'     → .pi
      'GOOD TIME INTERVAL'→ .gti
    """
    content = str(header.get("CONTENT", "")).upper()
    if "LIGHT CURVE" in content:
        return ProductKind.LIGHTCURVE
    if "PHA" in content or "SPECTRUM" in content:
        return ProductKind.SPECTRUM
    if "GOOD TIME" in content or "GTI" in content:
        return ProductKind.GTI

    # Fall back to filename stem after stripping .gz
    stem = path.stem.lower()
    if stem.endswith(".pi"):
        return ProductKind.SPECTRUM
    if stem.endswith(".lc"):
        return ProductKind.LIGHTCURVE
    if stem.endswith(".gti"):
        return ProductKind.GTI

    # Last resort: extension
    suffix = path.suffix.lower()
    if suffix == ".pi":
        return ProductKind.SPECTRUM
    if suffix == ".lc":
        return ProductKind.LIGHTCURVE
    if suffix == ".gti":
        return ProductKind.GTI

    raise ValueError(f"Cannot infer product kind from path: {path}")


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_products(directory: str | Path) -> List[FITSProduct]:
    """
    Walk *directory* recursively and return one FITSProduct per FITS file.

    Recognises files ending in .fits, .fit, .pi, .lc, .gti (optionally .gz).
    Sub-directories named SDD1, SDD2, etc. are used to infer detector labels.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    _FITS_SUFFIXES = {".fits", ".fit", ".pi", ".lc", ".gti"}

    products: List[FITSProduct] = []

    for root, _dirs, files in os.walk(directory):
        root_path = Path(root)
        # Detector label from directory name (SDD1, SDD2, CdZnTe, …)
        detector_hint = root_path.name.upper()

        for fname in sorted(files):
            fpath = root_path / fname
            # Strip trailing .gz to get the real extension
            stem_path = Path(fname[:-3]) if fname.endswith(".gz") else Path(fname)
            if stem_path.suffix.lower() not in _FITS_SUFFIXES:
                continue

            try:
                hdul = _open_fits(fpath)
            except Exception as exc:
                import warnings
                warnings.warn(f"Skipping {fpath}: {exc}")
                continue

            primary_hdr = dict(hdul[0].header)
            instrument = detect_instrument(hdul[0].header)

            # Detector: prefer FILTER keyword (set to 'SDD2' etc. in RATE HDU)
            # then fall back to directory hint or PRIMARY header
            detector = str(hdul[0].header.get("FILTER", "")).strip()
            if not detector and len(hdul) > 1:
                detector = str(hdul[1].header.get("FILTER", "")).strip()
            if not detector:
                detector = detector_hint if detector_hint else "UNKNOWN"

            kind = _detect_product_kind(hdul[0].header, stem_path)
            hdul.close()

            products.append(FITSProduct(
                path=fpath,
                instrument=instrument,
                detector=detector,
                kind=kind,
                primary_header=primary_hdr,
            ))

    return products


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_fits(path: Path) -> fits.HDUList:
    """Open a FITS file that may be gzip-compressed."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        return fits.open(io.BytesIO(raw))
    return fits.open(str(path))
