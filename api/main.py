"""
SoLEXS Solar Flare Forecasting — FastAPI inference server
==========================================================
POST /predict   — run inference on a 30-min SoLEXS window
GET  /health    — liveness check
GET  /metrics   — stored evaluation metrics from training
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — works both locally and on Render
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_DIR = ROOT / "pipeline" / "checkpoints"

from pipeline.module4.model import SolarPatchTST

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SoLEXS Flare Forecaster",
    description="Real-time M+/X-class solar flare probability from SoLEXS SDD2 light curves",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup
# ---------------------------------------------------------------------------
_model: Optional[SolarPatchTST]  = None
_scaler                           = None
_calibrator                       = None
_feature_names: List[str]         = []
_device                           = torch.device("cpu")


@app.on_event("startup")
def load_model() -> None:
    global _model, _scaler, _calibrator, _feature_names

    model_path = CKPT_DIR / "best_model.pt"
    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found at {model_path}")

    state = torch.load(model_path, map_location=_device)
    _feature_names = state["feature_names"]

    _model = SolarPatchTST(
        feature_names=_feature_names,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    _model.load_state_dict(state["model_state"])
    _model.eval()

    scaler_path = CKPT_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            d = pickle.load(f)
            _scaler = d["scaler"]

    cal_path = CKPT_DIR / "isotonic_calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            d = pickle.load(f)
            _calibrator = d["ir"]

    print(f"Model loaded — {len(_feature_names)} features, device={_device}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    window: List[List[float]] = Field(
        ...,
        description=(
            "2-D array of shape [1800, 29]: 30 minutes of SoLEXS SDD2 "
            "light-curve features at 1-second cadence. "
            "Feature order must match the training feature list (see /features)."
        ),
        min_length=1800,
        max_length=1800,
    )
    calibrate: bool = Field(
        True,
        description="Apply isotonic calibration to reduce ECE (recommended).",
    )


class PredictResponse(BaseModel):
    prob_15min: float = Field(..., description="P(M+ flare within 15 min)")
    prob_30min: float = Field(..., description="P(M+ flare within 30 min)")
    prob_60min: float = Field(..., description="P(M+ flare within 60 min)")
    prob_extreme: float = Field(..., description="P(X-class flare within 30 min)")
    flare_class_probs: dict = Field(..., description="Softmax probs for C / M / X class")
    calibrated: bool = Field(..., description="Whether isotonic calibration was applied")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "calibrator_loaded": _calibrator is not None,
        "n_features": len(_feature_names),
    }


@app.get("/features")
def features() -> dict:
    return {"feature_names": _feature_names, "n_features": len(_feature_names)}


@app.get("/metrics")
def metrics() -> dict:
    path = CKPT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    with open(path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate shape
    arr = np.array(req.window, dtype=np.float32)   # (1800, 29)
    if arr.shape != (1800, len(_feature_names)):
        raise HTTPException(
            status_code=422,
            detail=f"Expected window shape (1800, {len(_feature_names)}), got {arr.shape}",
        )

    # Normalise with training scaler
    if _scaler is not None:
        arr = _scaler.transform(arr)               # (1800, 29)

    # Model forward pass
    x = torch.from_numpy(arr).unsqueeze(0)         # (1, 1800, 29)
    with torch.no_grad():
        logit_binary, logit_extreme, logit_class = _model(x)

    p_binary  = torch.sigmoid(logit_binary).squeeze(0).numpy()          # (3,)
    p_extreme = float(torch.sigmoid(logit_extreme).squeeze())
    p_class   = torch.softmax(logit_class, dim=-1).squeeze(0).numpy()   # (3,)

    p15, p30, p60 = float(p_binary[0]), float(p_binary[1]), float(p_binary[2])

    # Optional isotonic calibration on the 60-min head
    calibrated = False
    if req.calibrate and _calibrator is not None:
        p60 = float(_calibrator.predict(np.array([p60]))[0])
        calibrated = True

    return PredictResponse(
        prob_15min=round(p15, 4),
        prob_30min=round(p30, 4),
        prob_60min=round(p60, 4),
        prob_extreme=round(p_extreme, 4),
        flare_class_probs={
            "C": round(float(p_class[0]), 4),
            "M": round(float(p_class[1]), 4),
            "X": round(float(p_class[2]), 4),
        },
        calibrated=calibrated,
    )
