"""
catalogue.py — Flare catalogue construction and I/O
=====================================================
Converts a list of FlareEvent objects into a clean pandas DataFrame and
provides save/load helpers.

Catalogue columns
-----------------
  onset_time      : ISO-8601 UTC string
  peak_time       : ISO-8601 UTC string
  end_time        : ISO-8601 UTC string
  onset_idx       : int — index into SoLEXS time array
  peak_idx        : int
  end_idx         : int
  duration_s      : float — seconds from onset to end
  lead_time_s     : float — seconds from onset to peak (Module 4 headline metric)
  peak_excess_A   : float — maximum excess_A in the event window
  goes_class      : str   — "A"/"B"/"C"/"M"/"X"/"?"
  goes_class_int  : int   — 0-4 / -1
  confidence      : float — fraction of event cadences where both channels fire
  soft_triggers   : int
  hard_triggers   : int
  both_triggers   : int
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .detector import FlareEvent


def build_catalogue(events: List[FlareEvent]) -> "pd.DataFrame":
    """
    Convert a list of FlareEvent objects to a DataFrame.

    Returns an empty DataFrame (with correct columns) if events is empty.
    """
    import pandas as pd

    if not events:
        return pd.DataFrame(columns=[
            "onset_time", "peak_time", "end_time",
            "onset_idx", "peak_idx", "end_idx",
            "duration_s", "lead_time_s", "peak_excess_A",
            "goes_class", "goes_class_int", "confidence",
            "soft_triggers", "hard_triggers", "both_triggers",
        ])

    rows = []
    for ev in events:
        flags = ev.detection_flags or {}
        rows.append({
            "onset_time":    ev.onset_time,
            "peak_time":     ev.peak_time,
            "end_time":      ev.end_time,
            "onset_idx":     ev.onset_idx,
            "peak_idx":      ev.peak_idx,
            "end_idx":       ev.end_idx,
            "duration_s":    ev.duration_s,
            "lead_time_s":   ev.lead_time_s,
            "peak_excess_A": round(ev.peak_excess_A, 4),
            "goes_class":    ev.goes_class,
            "goes_class_int": ev.goes_class_int,
            "confidence":    ev.confidence,
            "soft_triggers": flags.get("soft_trigger_count", -1),
            "hard_triggers": flags.get("hard_trigger_count", -1),
            "both_triggers": flags.get("both_trigger_count", -1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("onset_idx").reset_index(drop=True)
    return df


def save_catalogue(df: "pd.DataFrame", path: str | Path, fmt: str = "csv") -> None:
    """
    Persist the flare catalogue to disk.

    Parameters
    ----------
    df   : output of build_catalogue()
    path : destination file path (.csv or .parquet)
    fmt  : "csv" (default) or "parquet"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_catalogue(path: str | Path) -> "pd.DataFrame":
    """Load a catalogue previously saved by save_catalogue()."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def catalogue_summary(df: "pd.DataFrame") -> dict:
    """Return a human-readable summary dict of the catalogue."""
    if df.empty:
        return {"n_events": 0}

    import pandas as pd

    class_counts = df["goes_class"].value_counts().to_dict()
    lead_times   = df["lead_time_s"].dropna()

    return {
        "n_events":           len(df),
        "by_goes_class":      class_counts,
        "lead_time_median_s": round(float(lead_times.median()), 1) if len(lead_times) else np.nan,
        "lead_time_p25_s":    round(float(lead_times.quantile(0.25)), 1) if len(lead_times) else np.nan,
        "lead_time_p75_s":    round(float(lead_times.quantile(0.75)), 1) if len(lead_times) else np.nan,
        "peak_excess_A_max":  round(float(df["peak_excess_A"].max()), 2),
        "mean_confidence":    round(float(df["confidence"].mean()), 3),
        "mean_duration_s":    round(float(df["duration_s"].mean()), 1),
    }
