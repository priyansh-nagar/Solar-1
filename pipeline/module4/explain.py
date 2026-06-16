"""
explain.py — SHAP-based explainability for SolarPatchTST
=========================================================
Uses SHAP DeepExplainer on the fused representation layer to compute
feature × patch attribution maps for the 30-min M+ head.

Because the full model is (B, 1800, 29) → scalar, we provide a thin
wrapper that exposes only the 30-min binary output.  The SHAP values
are then averaged over the patch dimension to give per-feature importance.

Usage
-----
    explainer = SHAPExplainer(model, train_loader, device, n_bg=64)
    attrs     = explainer.explain(test_loader, n_samples=128)
    explainer.plot_feature_importance(attrs, feature_names, save_path="shap.png")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SolarPatchTST


# ---------------------------------------------------------------------------
# Thin wrapper exposing the 30-min binary output only
# ---------------------------------------------------------------------------

class _BinaryWrapper(nn.Module):
    """Wraps SolarPatchTST to output only P(M+ 30-min) as a scalar per sample."""

    def __init__(self, model: SolarPatchTST):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lb, _, _ = self.model(x)
        return torch.sigmoid(lb[:, 1])   # (B,)


# ---------------------------------------------------------------------------
# Gradient × Input attributions (fast fallback when SHAP is slow)
# ---------------------------------------------------------------------------

def _gradient_x_input(
    model:   SolarPatchTST,
    X:       torch.Tensor,
    device:  torch.device,
    head:    int = 1,         # 0=15min, 1=30min, 2=60min
) -> np.ndarray:
    """
    Gradient × Input attributions for the binary head (fast, deterministic).

    Returns (N, 1800, 29) attribution array.
    """
    model.eval()
    X_dev = X.to(device).requires_grad_(True)
    lb, _, _ = model(X_dev)
    out = torch.sigmoid(lb[:, head])
    out.sum().backward()
    attrs = (X_dev.grad * X_dev).detach().cpu().numpy()
    return attrs


# ---------------------------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    SHAP-based feature attribution for the SolarPatchTST 30-min binary head.

    Falls back to Gradient × Input attribution if SHAP is unavailable or
    the model is too large for DeepExplainer within memory limits.

    Parameters
    ----------
    model          : trained SolarPatchTST
    train_loader   : DataLoader for background sample selection (TRAIN)
    device         : torch device
    n_bg           : number of background samples for SHAP (default 64)
    """

    def __init__(
        self,
        model:        SolarPatchTST,
        train_loader: DataLoader,
        device:       torch.device,
        n_bg:         int = 64,
    ):
        self.model  = model
        self.device = device
        self.n_bg   = n_bg
        self._wrapper = _BinaryWrapper(model).to(device)
        self._shap_values: Optional[np.ndarray] = None

        # Collect background samples
        bg_chunks = []
        for batch in train_loader:
            bg_chunks.append(batch["X"])
            if sum(c.size(0) for c in bg_chunks) >= n_bg:
                break
        self._bg = torch.cat(bg_chunks, dim=0)[: n_bg].to(device)

    def explain(
        self,
        loader:    DataLoader,
        n_samples: int  = 128,
        method:    str  = "auto",
        verbose:   bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature attributions for a set of samples.

        Parameters
        ----------
        loader    : DataLoader with samples to explain
        n_samples : maximum number of samples to explain (expensive!)
        method    : "shap" | "grad_x_input" | "auto" (tries SHAP, falls back)

        Returns
        -------
        dict with keys:
          "attrs"   : (N, 1800, 29)  per-cadence per-feature attribution
          "feat_imp": (29,)           mean |attribution| per feature
          "patch_imp":(60,)           mean |attribution| per patch
          "y_true"  : (N,)            ground truth M+ 30-min label
          "prob_30" : (N,)            predicted M+ probability
        """
        # Collect test samples
        chunks_X, chunks_y = [], []
        for batch in loader:
            chunks_X.append(batch["X"])
            chunks_y.append(batch["y_binary"][:, 1])
            if sum(c.size(0) for c in chunks_X) >= n_samples:
                break
        X_exp  = torch.cat(chunks_X, dim=0)[: n_samples]
        y_true = torch.cat(chunks_y, dim=0)[: n_samples].numpy()

        if verbose:
            print(f"Computing attributions for {X_exp.size(0)} samples …")

        attrs = None

        if method in ("shap", "auto"):
            attrs = self._try_shap(X_exp, verbose)

        if attrs is None:
            if verbose and method == "auto":
                print("  Falling back to Gradient × Input attributions …")
            attrs = self._grad_x_input(X_exp)

        # Probabilities for reference
        with torch.no_grad():
            prob_30 = torch.sigmoid(
                self.model(X_exp.to(self.device))[0][:, 1]
            ).cpu().numpy()

        feat_imp  = np.abs(attrs).mean(axis=(0, 1))   # (29,)
        patch_imp = np.abs(attrs).reshape(
            attrs.shape[0], 60, 30, attrs.shape[2]    # (N,60,30,29)
        ).mean(axis=(0, 2, 3))                         # (60,)

        self._shap_values = attrs

        return {
            "attrs":     attrs,
            "feat_imp":  feat_imp,
            "patch_imp": patch_imp,
            "y_true":    y_true,
            "prob_30":   prob_30,
        }

    def _try_shap(
        self, X_exp: torch.Tensor, verbose: bool
    ) -> Optional[np.ndarray]:
        try:
            import shap
            self._wrapper.eval()
            explainer = shap.DeepExplainer(
                self._wrapper,
                self._bg,
            )
            vals = explainer.shap_values(X_exp.to(self.device))
            if isinstance(vals, list):
                vals = vals[0]
            if verbose:
                print("  SHAP DeepExplainer completed.")
            return np.array(vals)
        except Exception as exc:
            if verbose:
                print(f"  SHAP failed ({exc}); falling back.")
            return None

    def _grad_x_input(self, X_exp: torch.Tensor) -> np.ndarray:
        return _gradient_x_input(self.model, X_exp, self.device)

    def plot_feature_importance(
        self,
        attrs:         Dict[str, np.ndarray],
        feature_names: List[str],
        save_path:     Optional[str] = None,
        top_k:         int           = 15,
    ) -> None:
        """
        Bar chart of mean absolute attribution per feature (top-k).

        Requires matplotlib.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot.")
            return

        feat_imp = attrs["feat_imp"]
        order    = np.argsort(-feat_imp)[:top_k]
        names    = [feature_names[i] for i in order]
        vals     = feat_imp[order]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(range(top_k), vals[::-1], color="#e55c45", edgecolor="none")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Mean |Attribution|", fontsize=10)
        ax.set_title(
            f"Feature Importance — SolarPatchTST 30-min M+ head\n"
            f"(n={len(attrs['y_true'])} samples)",
            fontsize=11,
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_patch_timeline(
        self,
        attrs:     Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Line plot of mean absolute attribution per 30-s patch over the 30-min window.
        Highlights the final 5 patches (most predictive region for onset).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        patch_imp = attrs["patch_imp"]   # (60,)
        t_min     = np.arange(60) * 0.5  # 0 … 29.5 min

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_min, patch_imp, color="#3a7bd5", lw=1.5)
        ax.fill_between(t_min, patch_imp, alpha=0.15, color="#3a7bd5")
        ax.axvspan(27.5, 30.0, color="#e55c45", alpha=0.15, label="Final 5 patches")
        ax.set_xlabel("Window time (min)", fontsize=10)
        ax.set_ylabel("Mean |Attribution|", fontsize=10)
        ax.set_title("Patch-level attribution over 30-min window", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()
