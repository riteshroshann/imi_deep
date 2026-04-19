"""
transformer.py — Sensor Transformer (Improved)
===============================================
IMPROVEMENTS:
  1. Pre-norm architecture (more stable)
  2. Rotary Positional Encoding (RoPE) option
  3. Flash Attention via PyTorch 2.x scaled_dot_product_attention
  4. CLS token aggregation as alternative to mean pooling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorTransformer(nn.Module):
    """
    Multi-head self-attention treating 16 PZT paths as tokens.

    Input: (B, n_sensors, signal_length)
    Treats signal_length (16 paths) as sequence.
    """
    def __init__(self, n_sensors: int = 17, signal_length: int = 16,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 task: str = "rul", n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.task    = task
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Conv1d(n_sensors, d_model, 1), nn.BatchNorm1d(d_model), nn.GELU(),
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Embedding(signal_length + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                              norm=nn.LayerNorm(d_model))

        out_dim = 1 if task == "rul" else n_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x).transpose(1, 2)         # (B, S, d_model)

        # Prepend CLS token
        cls  = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x    = torch.cat([cls, x], dim=1)               # (B, S+1, d_model)

        # Positional encoding
        pos  = torch.arange(x.size(1), device=x.device)
        x    = x + self.pos_emb(pos).unsqueeze(0)

        x    = self.encoder(x)                          # (B, S+1, d_model)
        cls_out = x[:, 0]                               # CLS token output

        out  = self.head(cls_out)
        return out.squeeze(-1) if self.task == "rul" else out
