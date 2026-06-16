"""
model.py — SolarPatchTST dual-stream forecaster
================================================
Architecture:
  1. Split 29 features into soft (Band A / primary) and hard (B/C/D / spectral)
  2. Patchify each stream → 60 patches of 30 timesteps → linear embed to d_model
  3. Within-stream Transformer encoder (n_layers, n_heads, dropout)
  4. Cross-stream attention: soft queries hard and vice versa
  5. Mean-pool → concatenate → three parallel heads

Outputs per forward pass (all pre-sigmoid/softmax logits except p_class):
  logit_binary  (B, 3)  — M+ at 15 / 30 / 60 min
  logit_extreme (B, 1)  — X-class at 30 min
  logit_class   (B, 3)  — raw logits for C / M / X (apply softmax externally)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import SOFT_FEAT_PATTERNS, HARD_FEAT_PATTERNS, resolve_stream_indices


# ---------------------------------------------------------------------------
# Soft / hard feature name patterns (re-exported for convenience)
# ---------------------------------------------------------------------------

__all__ = ["SolarPatchTST", "SOFT_FEAT_PATTERNS", "HARD_FEAT_PATTERNS"]

PATCH_SIZE = 30       # seconds per patch
N_PATCHES  = 60       # 1800 / 30


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, fixed)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    (B, T, C) → (B, N_patches, d_model)

    patch_size timesteps are flattened then projected linearly.
    """

    def __init__(self, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_features * patch_size, d_model)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        P = self.patch_size
        N = T // P
        x = x[:, : N * P].reshape(B, N, P * C)   # (B, N_patches, P*C)
        return self.proj(x)                        # (B, N_patches, d_model)


# ---------------------------------------------------------------------------
# Transformer stream encoder
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm — more stable on small datasets
        )
        # enable_nested_tensor is incompatible with norm_first=True; disable explicitly
        # to suppress PyTorch's UserWarning on every instantiation.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)   # (B, N_patches, d_model)


# ---------------------------------------------------------------------------
# Cross-stream attention block
# ---------------------------------------------------------------------------

class CrossStreamAttention(nn.Module):
    """
    Query from stream A, Key/Value from stream B.
    Residual added back to stream A representation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # (B, N, d_model) — from stream A
        kv:    torch.Tensor,    # (B, N, d_model) — from stream B
    ) -> torch.Tensor:
        out, _ = self.attn(query, kv, kv)
        return self.norm(query + self.drop(out))   # (B, N, d_model)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SolarPatchTST(nn.Module):
    """
    Dual-stream PatchTST solar flare forecaster.

    Parameters
    ----------
    feature_names : list of 29 feature names in the column order of X
    patch_size    : timesteps per patch (default 30 s)
    d_model       : embedding dimension (default 64)
    n_heads       : attention heads (default 4)
    n_layers      : Transformer layers per stream (default 2)
    dropout       : dropout rate (default 0.2)
    window_s      : window length in samples (default 1800)
    """

    def __init__(
        self,
        feature_names: List[str],
        patch_size: int   = PATCH_SIZE,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.2,
        window_s:   int   = 1800,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.n_patches     = window_s // patch_size
        self.d_model       = d_model
        self.feature_names = feature_names

        soft_idx, hard_idx = resolve_stream_indices(feature_names)
        self.register_buffer("soft_idx", torch.tensor(soft_idx, dtype=torch.long))
        self.register_buffer("hard_idx", torch.tensor(hard_idx, dtype=torch.long))
        n_soft = len(soft_idx)
        n_hard = len(hard_idx)

        # Stream embeddings
        self.embed_soft = PatchEmbedding(n_soft, patch_size, d_model)
        self.embed_hard = PatchEmbedding(n_hard, patch_size, d_model)

        # Positional encodings
        self.pe_soft = SinusoidalPE(d_model, max_len=self.n_patches + 4)
        self.pe_hard = SinusoidalPE(d_model, max_len=self.n_patches + 4)

        # Within-stream self-attention encoders
        self.enc_soft = StreamEncoder(d_model, n_heads, n_layers, dropout)
        self.enc_hard = StreamEncoder(d_model, n_heads, n_layers, dropout)

        # Cross-stream attention
        self.cross_s2h = CrossStreamAttention(d_model, n_heads, dropout)
        self.cross_h2s = CrossStreamAttention(d_model, n_heads, dropout)

        # Shared fusion norm + projection
        fused_dim = 2 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Heads ──────────────────────────────────────────────────────────
        # Binary M+ head: 3 outputs (15 / 30 / 60 min) — BCEWithLogitsLoss
        self.head_binary = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )
        # Extreme event head: X-class at 30 min — BCEWithLogitsLoss
        self.head_extreme = nn.Sequential(
            nn.Linear(fused_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        # Multiclass head: C / M / X — CrossEntropyLoss (raw logits, no softmax)
        self.head_class = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1800, 29)

        Returns
        -------
        logit_binary  (B, 3)  — pre-sigmoid logits for M+ at 15/30/60 min
        logit_extreme (B, 1)  — pre-sigmoid logit for X-class at 30 min
        logit_class   (B, 3)  — pre-softmax logits for C/M/X
        """
        # ── Split into streams ─────────────────────────────────────────────
        x_soft = x[:, :, self.soft_idx]   # (B, 1800, n_soft)
        x_hard = x[:, :, self.hard_idx]   # (B, 1800, n_hard)

        # ── Patch + embed + positional encoding ───────────────────────────
        s = self.pe_soft(self.embed_soft(x_soft))   # (B, 60, d_model)
        h = self.pe_hard(self.embed_hard(x_hard))   # (B, 60, d_model)

        # ── Within-stream self-attention ──────────────────────────────────
        s_enc = self.enc_soft(s)   # (B, 60, d_model)
        h_enc = self.enc_hard(h)   # (B, 60, d_model)

        # ── Cross-stream attention ────────────────────────────────────────
        s_cross = self.cross_s2h(s_enc, h_enc)   # soft queries hard
        h_cross = self.cross_h2s(h_enc, s_enc)   # hard queries soft

        # ── Mean-pool patches → fuse ──────────────────────────────────────
        s_pool = s_cross.mean(dim=1)              # (B, d_model)
        h_pool = h_cross.mean(dim=1)              # (B, d_model)
        fused  = torch.cat([s_pool, h_pool], dim=-1)  # (B, 2*d_model)
        fused  = self.fusion_proj(self.fusion_norm(fused))

        return (
            self.head_binary(fused),       # (B, 3)
            self.head_extreme(fused),      # (B, 1)
            self.head_class(fused),        # (B, 3)
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
