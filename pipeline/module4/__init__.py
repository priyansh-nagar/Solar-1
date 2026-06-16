"""
Module 4 — PatchTST Solar Flare Forecaster
==========================================
Dual-stream PatchTST with cross-channel attention, multi-horizon heads,
temperature scaling, and SHAP explainability.
"""

from .model     import SolarPatchTST, SOFT_FEAT_PATTERNS, HARD_FEAT_PATTERNS
from .dataset   import SolarFlareDataset, load_splits
from .train     import train_model
from .evaluate  import evaluate_model, tpr_at_far
from .calibrate import TemperatureScaler
from .explain   import SHAPExplainer

__all__ = [
    "SolarPatchTST",
    "SolarFlareDataset",
    "load_splits",
    "train_model",
    "evaluate_model",
    "tpr_at_far",
    "TemperatureScaler",
    "SHAPExplainer",
    "SOFT_FEAT_PATTERNS",
    "HARD_FEAT_PATTERNS",
]
