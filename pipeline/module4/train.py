"""
train.py — Training loop for SolarPatchTST
==========================================
Losses:
  total = 1.0 × binary_loss + 0.5 × extreme_loss + 0.3 × multiclass_loss

  binary_loss   : BCEWithLogitsLoss (pos_weight=2.15 for M+ imbalance)
  extreme_loss  : BCEWithLogitsLoss (pos_weight=22.7 for X-class rarity)
  multiclass_loss: CrossEntropyLoss (weights C=1.58, M=3.23, X=23.7)

Primary early-stopping metric: TPR @ FAR = 0.5 false alarms / hour
evaluated on the 30-min binary head output.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .model    import SolarPatchTST
from .evaluate import tpr_at_far


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------

# Class weights for multiclass head (C / M / X)
MULTICLASS_WEIGHTS = torch.tensor([1.58, 3.23, 23.7], dtype=torch.float32)

# TRAIN imbalance factor for M+ binary head
BINARY_POS_WEIGHT  = torch.tensor([2.15], dtype=torch.float32)   # broadcast over 3 outputs

# X-class rarity factor for extreme head
EXTREME_POS_WEIGHT = torch.tensor([22.7], dtype=torch.float32)

# Loss scale factors
LAMBDA_BINARY    = 1.0
LAMBDA_EXTREME   = 0.5
LAMBDA_MULTICLASS = 0.3


# ---------------------------------------------------------------------------
# Multiclass label mapping
#   y_class values 0–4 = A/B/C/M/X
#   multiclass head covers 3 classes: C(2), M(3), X(4)
#   classes 0 and 1 (A, B) are masked out of multiclass loss
# ---------------------------------------------------------------------------

_MULTI_OFFSET = 2   # y_class - 2 → 0=C, 1=M, 2=X


def _compute_loss(
    logit_binary:  torch.Tensor,   # (B, 3)
    logit_extreme: torch.Tensor,   # (B, 1)
    logit_class:   torch.Tensor,   # (B, 3) logits for C/M/X
    y_binary:      torch.Tensor,   # (B, 3)
    y_extreme:     torch.Tensor,   # (B,)
    y_class:       torch.Tensor,   # (B,) values 0–4
    bce_binary:    nn.BCEWithLogitsLoss,
    bce_extreme:   nn.BCEWithLogitsLoss,
    ce_class:      nn.CrossEntropyLoss,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    # Binary M+ loss (all 3 horizon outputs)
    loss_bin = bce_binary(logit_binary, y_binary)

    # Extreme X-class loss
    loss_ext = bce_extreme(logit_extreme.squeeze(-1), y_extreme)

    # Multiclass loss — only on windows with GOES class >= C
    mask_cplus = y_class >= _MULTI_OFFSET
    loss_mc = torch.tensor(0.0, device=logit_class.device)
    if mask_cplus.any():
        yc_offset = (y_class[mask_cplus] - _MULTI_OFFSET).clamp(0, 2)
        loss_mc = ce_class(logit_class[mask_cplus], yc_offset)

    total = (
        LAMBDA_BINARY     * loss_bin +
        LAMBDA_EXTREME    * loss_ext +
        LAMBDA_MULTICLASS * loss_mc
    )
    return total, {
        "loss":     total.item(),
        "bin":      loss_bin.item(),
        "extreme":  loss_ext.item(),
        "mc":       loss_mc.item(),
    }


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     SolarPatchTST,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    bce_bin:   nn.BCEWithLogitsLoss,
    bce_ext:   nn.BCEWithLogitsLoss,
    ce_mc:     nn.CrossEntropyLoss,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {"loss": 0, "bin": 0, "extreme": 0, "mc": 0}
    n_batches = 0

    for batch in loader:
        X         = batch["X"].to(device)
        y_binary  = batch["y_binary"].to(device)
        y_extreme = batch["y_extreme"].to(device)
        y_class   = batch["y_class"].to(device)

        optimizer.zero_grad()
        logit_bin, logit_ext, logit_mc = model(X)
        loss, parts = _compute_loss(
            logit_bin, logit_ext, logit_mc,
            y_binary, y_extreme, y_class,
            bce_bin, bce_ext, ce_mc,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in totals:
            totals[k] += parts[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:   SolarPatchTST,
    loader:  DataLoader,
    device:  torch.device,
    bce_bin: nn.BCEWithLogitsLoss,
    bce_ext: nn.BCEWithLogitsLoss,
    ce_mc:   nn.CrossEntropyLoss,
    stride_s: int = 60,
    far_thr:  float = 0.5,
) -> Dict[str, float]:
    """
    Returns validation loss + TPR @ FAR=0.5/hr on the 30-min binary head.

    Note: VAL partition (20240510) has 60.9% M+ prevalence — TPR will be
    optimistic. Flag this in evaluation reports.
    """
    model.eval()
    all_probs_30  = []
    all_yb30      = []
    all_yext      = []
    totals: Dict[str, float] = {"loss": 0, "bin": 0, "extreme": 0, "mc": 0}
    n_batches = 0

    for batch in loader:
        X         = batch["X"].to(device)
        y_binary  = batch["y_binary"].to(device)
        y_extreme = batch["y_extreme"].to(device)
        y_class   = batch["y_class"].to(device)

        logit_bin, logit_ext, logit_mc = model(X)
        _, parts = _compute_loss(
            logit_bin, logit_ext, logit_mc,
            y_binary, y_extreme, y_class,
            bce_bin, bce_ext, ce_mc,
        )
        for k in totals:
            totals[k] += parts[k]
        n_batches += 1

        all_probs_30.append(torch.sigmoid(logit_bin[:, 1]).cpu().numpy())   # 30-min head
        all_yb30.append(y_binary[:, 1].cpu().numpy())
        all_yext.append(y_extreme.cpu().numpy())

    probs30 = np.concatenate(all_probs_30)
    yb30    = np.concatenate(all_yb30)

    # Observation time in hours for FAR computation
    n_windows  = len(probs30)
    obs_hours  = n_windows * stride_s / 3600.0
    tpr_val    = tpr_at_far(probs30, yb30, far_thr=far_thr, obs_hours=obs_hours)

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics["tpr_at_far"] = tpr_val
    return metrics


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def train_model(
    model:          SolarPatchTST,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    checkpoint_dir: str | Path = "/tmp/module4_ckpt",
    n_epochs:       int        = 30,
    lr:             float      = 1e-4,
    weight_decay:   float      = 1e-4,
    patience:       int        = 7,
    device:         Optional[torch.device] = None,
    verbose:        bool       = True,
) -> Dict[str, list]:
    """
    Full training loop with early stopping on TPR @ FAR=0.5/hr.

    Returns history dict with per-epoch metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    model = model.to(device)

    # ── Losses ──────────────────────────────────────────────────────────────
    bce_bin = nn.BCEWithLogitsLoss(
        pos_weight=BINARY_POS_WEIGHT.expand(3).to(device)
    )
    bce_ext = nn.BCEWithLogitsLoss(
        pos_weight=EXTREME_POS_WEIGHT.to(device)
    )
    ce_mc = nn.CrossEntropyLoss(weight=MULTICLASS_WEIGHTS.to(device))

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history: Dict[str, list] = {
        "tr_loss": [], "tr_bin": [], "tr_ext": [], "tr_mc": [],
        "val_loss": [], "val_tpr": [], "lr": [],
    }

    best_tpr   = -1.0
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_m  = train_one_epoch(
            model, train_loader, optimizer, device, bce_bin, bce_ext, ce_mc
        )
        val_m = validate(
            model, val_loader, device, bce_bin, bce_ext, ce_mc
        )
        scheduler.step()

        history["tr_loss"].append(tr_m["loss"])
        history["tr_bin"].append(tr_m["bin"])
        history["tr_ext"].append(tr_m["extreme"])
        history["tr_mc"].append(tr_m["mc"])
        history["val_loss"].append(val_m["loss"])
        history["val_tpr"].append(val_m["tpr_at_far"])
        history["lr"].append(scheduler.get_last_lr()[0])

        tpr = val_m["tpr_at_far"]
        if tpr > best_tpr:
            best_tpr   = tpr
            no_improve = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_tpr":     best_tpr,
                    "feature_names": model.feature_names,
                    "history":      history,
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if verbose:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"tr_loss={tr_m['loss']:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"TPR@FAR0.5={tpr:.3f}  "
                f"best={best_tpr:.3f}  "
                f"patience={no_improve}/{patience}  "
                f"[{dt:.1f}s]"
            )

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} — best TPR@FAR0.5={best_tpr:.3f}")
            break

    # Load best weights
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if verbose:
            print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, TPR={ckpt['best_tpr']:.3f})")

    return history
