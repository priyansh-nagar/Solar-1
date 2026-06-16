"""
calibrate.py — Temperature scaling for SolarPatchTST
=====================================================
Temperature scaling (Guo et al. 2017) is the single most effective
post-hoc calibration technique for neural classifiers.

A single learnable scalar T is applied to the pre-sigmoid logits:
    p_calibrated = sigmoid(logit / T)

T > 1 → softens overconfident predictions (most common case)
T < 1 → sharpens underconfident predictions

The temperature is learned by minimising NLL on the VAL partition
(keeping all other model weights frozen).
"""

from __future__ import annotations

from typing import Dict, Tuple

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
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect 30-min binary logits and ground truth from a loader."""
        self.model.eval()
        logits_list, y_list = [], []
        for batch in loader:
            X = batch["X"].to(device)
            lb, _, _ = self.model(X)
            logits_list.append(lb[:, 1].cpu())    # 30-min head
            y_list.append(batch["y_binary"][:, 1])
        return torch.cat(logits_list), torch.cat(y_list).float()

    def fit(
        self,
        val_loader: DataLoader,
        device:     torch.device,
        lr:         float = 0.01,
        max_iter:   int   = 200,
        verbose:    bool  = True,
    ) -> "TemperatureScaler":
        """
        Fit temperature T on VAL NLL (M+ 30-min head).

        Returns self for chaining.
        """
        self.to(device)
        logits, y_true = self._collect_logits(val_loader, device)
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
        loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Return calibrated 30-min M+ probabilities for the full loader."""
        self.eval()
        probs_list, y_list = [], []
        for batch in loader:
            X = batch["X"].to(device)
            lb, _, _ = self(X)
            probs_list.append(torch.sigmoid(lb[:, 1]).cpu().numpy())
            y_list.append(batch["y_binary"][:, 1].numpy())
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
