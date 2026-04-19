"""
hybrid_sta.py — HybridSTA-V3 (Improved)
=========================================
IMPROVEMENTS over original:
  1. Learnable 4×4 geometric positional embedding (sensor grid aware)
  2. Pre-norm (LayerNorm before attention) for training stability
  3. Dual output heads: regression (RUL) + auxiliary classification
  4. Stochastic Depth (drop_path) regularisation
  5. Weight-tying between SE fc1/fc2 removed (improves expressivity)
  6. Attention weight caching for XAI (return_attn=True)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DropPath(nn.Module):
    """Stochastic depth regularisation (drops entire sample path)."""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device).floor_() + keep
        return x / keep * random_tensor


class SqueezeExcitation(nn.Module):
    """SE block with independent fc1/fc2 (no weight tying)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1  = nn.Linear(channels, mid, bias=True)
        self.fc2  = nn.Linear(mid, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S)
        s    = self.pool(x).squeeze(-1)          # (B, C)
        attn = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))  # (B, C)
        return x * attn.unsqueeze(-1)


class GeometricPositionalEncoding(nn.Module):
    """
    4×4 sensor-grid-aware positional embedding.
    Each of the 16 sensor positions gets a learned (row, col) embedding.
    """
    def __init__(self, d_model: int, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        n = grid_size * grid_size         # 16 positions
        self.row_emb = nn.Embedding(grid_size, d_model // 2)
        self.col_emb = nn.Embedding(grid_size, d_model // 2)

        rows = torch.arange(n) // grid_size
        cols = torch.arange(n) %  grid_size
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns (1, seq_len, d_model)."""
        r_emb = self.row_emb(self.rows[:seq_len])  # (S, d/2)
        c_emb = self.col_emb(self.cols[:seq_len])  # (S, d/2)
        return torch.cat([r_emb, c_emb], dim=-1).unsqueeze(0)  # (1, S, d)


class PreNormTransformerLayer(nn.Module):
    """Transformer layer with pre-LayerNorm (more stable training)."""
    def __init__(self, d_model: int, n_heads: int, dim_ff: int,
                 dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                return_attn: bool = False) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm  = self.norm1(x)
        attn_out, attn_w = self.attn(x_norm, x_norm, x_norm,
                                      need_weights=return_attn,
                                      average_attn_weights=False)
        if return_attn:
            self._last_attn_weights = attn_w.detach()
        x = x + self.drop_path(attn_out)

        # FFN with pre-norm
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class SpatialTemporalAttention(nn.Module):
    """
    HybridSTA-V3: geometry-aware, pre-norm, dual-head, stochastic depth.

    Architecture:
      1. Feature projection (Conv1d)
      2. Squeeze-and-Excitation channel calibration
      3. Geometric 4×4 positional embedding
      4. Stack of pre-norm transformer layers with DropPath
      5. Global average pool
      6. Dual output heads (RUL regression + damage classification)
    """

    def __init__(
        self,
        n_sensors:    int   = 17,
        signal_length: int  = 16,
        d_model:      int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 3,
        n_classes:    int   = 5,
        task:         str   = "rul",
        dropout:      float = 0.2,
        drop_path:    float = 0.1,
    ):
        super().__init__()
        self.task          = task
        self.signal_length = signal_length
        self.d_model       = d_model

        # 1. Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv1d(n_sensors, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. SE channel calibration
        self.se = SqueezeExcitation(d_model, reduction=4)

        # 3. Geometric positional encoding
        self.pos_enc = GeometricPositionalEncoding(d_model, grid_size=4)

        # 4. Pre-norm transformer stack
        dp_rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, n_heads, d_model * 2, dropout, dp)
            for dp in dp_rates
        ])
        self.norm = nn.LayerNorm(d_model)

        # 5. Aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 6. Output heads
        mid = d_model // 2
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, n_classes),
        )

        self._init_weights()
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, n_sensors, signal_length)
        Returns:
            RUL predictions (B,) for task='rul',
            class logits (B, n_classes) for task='classification'
        """
        # 1. Project
        feat = self.feature_proj(x)          # (B, d_model, S)

        # 2. SE
        feat = self.se(feat)                 # (B, d_model, S)

        # 3. Reshape + positional encoding
        feat = feat.transpose(1, 2)          # (B, S, d_model)
        feat = feat + self.pos_enc(feat.size(1)).to(feat.device)

        # 4. Transformer stack
        for layer in self.layers:
            feat = layer(feat, return_attn=return_attn)
        feat = self.norm(feat)               # (B, S, d_model)

        # 5. Pool
        g = self.pool(feat.transpose(1, 2)).squeeze(-1)   # (B, d_model)

        # 6. Output
        if self.task == "classification":
            return self.cls_head(g)
        else:
            return self.rul_head(g).squeeze(-1)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last layer (for XAI)."""
        last = self.layers[-1]
        return getattr(last, "_last_attn_weights", None)
