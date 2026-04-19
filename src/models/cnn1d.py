"""
cnn1d.py — 1D-CNN with Multi-Scale Residual Blocks (Improved)
==============================================================
IMPROVEMENTS:
  1. Three parallel kernel branches (k=3, k=7, k=15) → richer multi-scale receptive field
  2. Squeeze-and-Excitation on each residual block
  3. Batch normalisation before activation (pre-BN ResNet style)
  4. GlobalAveragePool + GlobalMaxPool concatenated for richer aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).squeeze(-1)
        return x * self.fc(s).unsqueeze(-1)


class MultiScaleResBlock(nn.Module):
    """Parallel 3-branch conv (k=3, 7, 15) + SE + residual."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        branch_ch = out_ch // 3 + (out_ch % 3)  # ensure sum = out_ch
        self.b3  = self._conv(in_ch, branch_ch, 3)
        self.b7  = self._conv(in_ch, branch_ch, 7)
        self.b15 = self._conv(in_ch, out_ch - 2 * branch_ch, 15)

        total = branch_ch * 2 + (out_ch - 2 * branch_ch)
        self.se      = SEBlock1D(total)
        self.proj    = nn.Conv1d(in_ch, total, 1) if in_ch != total else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.BatchNorm1d(total)

    @staticmethod
    def _conv(in_ch, out_ch, k):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        out = self.se(out)
        out = self.dropout(out)
        return F.gelu(self.norm(out + self.proj(x)))


class CNN1D(nn.Module):
    """
    Multi-scale 1D-CNN with residual + SE blocks.

    Input: (B, n_sensors, signal_length)
    Output: scalar RUL or class logits
    """
    def __init__(self, n_sensors: int = 17, signal_length: int = 16,
                 task: str = "rul", n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.task = task

        self.stem = nn.Sequential(
            nn.Conv1d(n_sensors, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResBlock(64, 96,  dropout),
            MultiScaleResBlock(96, 128, dropout),
            MultiScaleResBlock(128, 128, dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        out_dim  = 1 if task == "rul" else n_classes
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        g = torch.cat([self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)], dim=-1)
        out = self.head(g)
        return out.squeeze(-1) if self.task == "rul" else out
