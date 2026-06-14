"""
Module 1 — Data Ingestion
=========================
Reads SoLEXS and HEL1OS Level-1 FITS products from ISSDC/PRADAN, applies
quality filtering, aligns the two instruments to a common time grid, and
returns a research-grade xarray Dataset.

Public API
----------
    from pipeline.module1 import ingest_day

    ds = ingest_day(
        solexs_dir="path/to/AL1_SLX_L1_20260612_v1.0/",
        hel1os_dir=None,          # omit if not available
    )

The returned Dataset has dimensions (time, channel) for per-detector spectra
and pre-computed science energy bands as data variables.
"""

from .ingest import ingest_day
from .formats import detect_instrument, FITSProduct

__all__ = ["ingest_day", "detect_instrument", "FITSProduct"]
