"""
calibrate.py — Post-hoc calibration for SolarPatchTST
=======================================================
Two calibrators are provided:

TemperatureScaler (Guo et al. 2017)
  A single learnable scalar T is applied to the pre-sigmoid logits:
      p_calibrated = sigmoid(logit / T)
  Best when miscalibration is purely a scale problem.

IsotonicCalibrator
  Fits a non-parametric isotonic regression (sklearn) mapping raw
  sigmoid probabilities → calibrated probabilities on VAL.
  Better when the model's prior doesn't match the deployment prior
  (e.g. TRAIN prevalence 31.8% vs TEST prevalence 19.3%).
  ECE improvement is typically 2-4× larger than temperature scaling
  in the presence of distributional prior mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model    import SolarPatchTST
from .evaluate import ece


class TemperatureScaler(nn.Module):
    """
    Wraps a trained SolarPatchTST and applies temperature scaling
    to the binary (M+) logit only.

    Usage
    -----
    scaler = TemperatureScaler(model).fit(val_loader, device)
    prob   = scaler.predict_30min(x)   # calibrated P(M+ | 30-min horizon)
    """

    def __init__(self, model: SolarPatchTST):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    @torch.no_grad()
    def _collect_logits(
        self,
        loader:      DataLoader,
        device:      torch.device,
        horizon_idx: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect binary logits and ground truth from a loader."""
        self.model.eval()
        logits_list, y_list = [], []
        for batch in loader:
            X = batch["X"].to(device)
            lb, _, _ = self.model(X)
            logits_list.append(lb[:, horizon_idx].cpu())
            y_list.append(batch["y_binary"][:, horizon_idx])
        return torch.cat(logits_list), torch.cat(y_list).float()

    def fit(
        self,
        val_loader:  DataLoader,
        device:      torch.device,
        lr:          float = 0.01,
        max_iter:    int   = 200,
        verbose:     bool  = True,
        horizon_idx: int   = 1,
    ) -> "TemperatureScaler":
        """
        Fit temperature T on VAL NLL (selected binary head).

        Returns self for chaining.
        """
        self.to(device)
        logits, y_true = self._collect_logits(val_loader, device, horizon_idx=horizon_idx)
        logits  = logits.to(device)
        y_true  = y_true.to(device)

        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer     = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def _closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nll_criterion(scaled_logits, y_true)
            loss.backward()
            return loss

        before_ece = self._ece_from_logits(logits, y_true)
        optimizer.step(_closure)
        after_ece  = self._ece_from_logits(logits / self.temperature.detach(), y_true)

        if verbose:
            T_val = self.temperature.item()
            print(
                f"Temperature scaling → T={T_val:.4f}  "
                f"ECE before={before_ece:.4f}  after={after_ece:.4f}"
            )

        return self

    @staticmethod
    def _ece_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
        probs  = torch.sigmoid(logits).detach().cpu().numpy()
        y_np   = y.detach().cpu().numpy()
        return ece(probs, y_np)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with temperature-scaled binary logits.

        Returns (scaled_logit_binary, logit_extreme, logit_class).
        The temperature is applied only to the binary M+ head.
        """
        lb, le, lmc = self.model(x)
        return lb / self.temperature, le, lmc

    @torch.no_grad()
    def predict_30min(
        self,
        loader:      DataLoader,
        device:      torch.device,
        horizon_idx: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Return calibrated M+ probabilities for the selected head."""
        self.eval()
        probs_list, y_list = [], []
        for batch in loader:
            X = batch["X"].to(device)
            lb, _, _ = self(X)
            probs_list.append(torch.sigmoid(lb[:, horizon_idx]).cpu().numpy())
            y_list.append(batch["y_binary"][:, horizon_idx].numpy())
        return {
            "prob_30_calibrated": np.concatenate(probs_list),
            "y_true_30":          np.concatenate(y_list),
        }

    def temperature_value(self) -> float:
        return float(self.temperature.item())

    def save(self, path: str) -> None:
        torch.save({"temperature": self.temperature.data}, path)

    def load(self, path: str) -> "TemperatureScaler":
        d = torch.load(path, map_location="cpu")
        self.temperature.data = d["temperature"]
        return self


# ---------------------------------------------------------------------------
# Isotonic Regression Calibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Non-parametric post-hoc calibrator using isotonic regression (sklearn).

    Unlike temperature scaling (one scalar), isotonic regression fits a
    step function P_raw → P_calibrated directly on VAL probabilities.
    This corrects distributional prior mismatch (e.g. TRAIN prior ≠ TEST
    prior) that temperature scaling cannot fix.

    Usage
    -----
    cal = IsotonicCalibrator(model, horizon_idx=2)
    cal.fit(val_loader, device)
    cal_probs = cal.predict(test_probs)  # numpy array
    cal.save(path)
    """

    def __init__(
        self,
        model:       SolarPatchTST,
        horizon_idx: int = 1,
    ):
        self.model       = model
        self.horizon_idx = horizon_idx
        self._ir         = None   # sklearn IsotonicRegression, set after fit
        self._ece_before: Optional[float] = None
        self._ece_after:  Optional[float] = None

    @torch.no_grad()
    def _collect_probs(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sigmoid probabilities + ground truth for the selected head."""
        self.model.eval()
        probs_list, y_list = [], []
        for batch in loader:
            X  = batch["X"].to(device)
            lb, _, _ = self.model(X)
            pb = torch.sigmoid(lb[:, self.horizon_idx]).cpu().numpy()
            yb = batch["y_binary"][:, self.horizon_idx].numpy()
            probs_list.append(pb)
            y_list.append(yb)
        return np.concatenate(probs_list), np.concatenate(y_list).astype(float)

    def fit(
        self,
        val_loader: DataLoader,
        device:     torch.device,
        verbose:    bool = True,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on VAL probabilities.

        Returns self for chaining.
        """
        from sklearn.isotonic import IsotonicRegression

        probs, y_true = self._collect_probs(val_loader, device)

        self._ece_before = ece(probs, y_true)

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs, y_true)
        self._ir = ir

        cal_probs        = ir.predict(probs)
        self._ece_after  = ece(cal_probs.astype(float), y_true)

        if verbose:
            print(
                f"Isotonic calibration → "
                f"ECE before={self._ece_before:.4f}  after={self._ece_after:.4f}"
            )

        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic regression to a probability array."""
        if self._ir is None:
            raise RuntimeError("IsotonicCalibrator.fit() must be called first.")
        return self._ir.predict(probs).astype(np.float32)

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Run inference + isotonic calibration on a full DataLoader."""
        raw_probs, y_true = self._collect_probs(loader, device)
        cal_probs = self.predict(raw_probs)
        return {
            "prob_30_calibrated": cal_probs,
            "y_true_30":          y_true,
        }

    def save(self, path: str) -> None:
        """Save fitted isotonic regressor to disk (joblib pickle)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"ir": self._ir, "horizon_idx": self.horizon_idx}, f)

    def load(self, path: str) -> "IsotonicCalibrator":
        """Load a previously saved isotonic regressor."""
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._ir         = d["ir"]
        self.horizon_idx = d.get("horizon_idx", self.horizon_idx)
        return self
