"""
Standalone SolarPatchTST — no pipeline dependencies.
Copy of pipeline/module4/model.py with feature constants inlined.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

SOFT_FEAT_PATTERNS = (
    "band_A", "total", "residual_A", "excess_A",
    "derivative_1s", "derivative_60s", "rate_of_rise",
    "softness_ratio", "rolling_std", "cumulative_excess",
)
HARD_FEAT_PATTERNS = (
    "band_B", "band_C", "band_D",
    "residual_B", "residual_C", "residual_D",
    "excess_B", "excess_C", "excess_D",
    "hardness_ratio",
)

PATCH_SIZE = 30
N_PATCHES  = 60


def resolve_stream_indices(feature_names: List[str]) -> Tuple[List[int], List[int]]:
    soft_idx, hard_idx = [], []
    for i, name in enumerate(feature_names):
        if any(p in name for p in HARD_FEAT_PATTERNS):
            hard_idx.append(i)
        else:
            soft_idx.append(i)
    return soft_idx, hard_idx


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_features * patch_size, d_model)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        P = self.patch_size
        N = T // P
        x = x[:, : N * P].reshape(B, N, P * C)
        return self.proj(x)


class StreamEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossStreamAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, kv, kv)
        return self.norm(query + self.drop(out))


class SolarPatchTST(nn.Module):
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

        self.embed_soft = PatchEmbedding(n_soft, patch_size, d_model)
        self.embed_hard = PatchEmbedding(n_hard, patch_size, d_model)
        self.pe_soft    = SinusoidalPE(d_model, max_len=self.n_patches + 4)
        self.pe_hard    = SinusoidalPE(d_model, max_len=self.n_patches + 4)
        self.enc_soft   = StreamEncoder(d_model, n_heads, n_layers, dropout)
        self.enc_hard   = StreamEncoder(d_model, n_heads, n_layers, dropout)
        self.cross_s2h  = CrossStreamAttention(d_model, n_heads, dropout)
        self.cross_h2s  = CrossStreamAttention(d_model, n_heads, dropout)

        fused_dim = 2 * d_model
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.head_binary = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 3),
        )
        self.head_extreme = nn.Sequential(
            nn.Linear(fused_dim, 16), nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 1),
        )
        self.head_class = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 3),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_soft = x[:, :, self.soft_idx]
        x_hard = x[:, :, self.hard_idx]
        s = self.pe_soft(self.embed_soft(x_soft))
        h = self.pe_hard(self.embed_hard(x_hard))
        s_enc   = self.enc_soft(s)
        h_enc   = self.enc_hard(h)
        s_cross = self.cross_s2h(s_enc, h_enc)
        h_cross = self.cross_h2s(h_enc, s_enc)
        s_pool  = s_cross.mean(dim=1)
        h_pool  = h_cross.mean(dim=1)
        fused   = torch.cat([s_pool, h_pool], dim=-1)
        fused   = self.fusion_proj(self.fusion_norm(fused))
        return (
            self.head_binary(fused),
            self.head_extreme(fused),
            self.head_class(fused),
        )
