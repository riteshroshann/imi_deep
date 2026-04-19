"""
tcn.py — Temporal Convolutional Network (Improved)
===================================================
IMPROVEMENTS:
  1. Weight normalisation on conv layers (training stability)
  2. GELU instead of ReLU (smooth gradient flow)
  3. Adaptive receptive field report
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                            padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                            padding=pad, dilation=dilation))
        self.bn1     = nn.BatchNorm1d(out_ch)
        self.bn2     = nn.BatchNorm1d(out_ch)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.chomp   = lambda x: x[:, :, :-pad] if pad > 0 else x
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.chomp(self.conv1(x))))
        out = self.dropout(out)
        out = self.act(self.bn2(self.chomp(self.conv2(out))))
        out = self.dropout(out)
        return self.act(out + self.downsample(x))


class TemporalConvNet(nn.Module):
    """Dilated TCN (d=2^i) for CFRP RUL regression."""
    def __init__(self, n_sensors: int = 17, signal_length: int = 16,
                 task: str = "rul", n_classes: int = 5,
                 n_filters: int = 64, kernel_size: int = 3,
                 n_levels: int = 4, dropout: float = 0.2):
        super().__init__()
        self.task = task

        layers = []
        in_ch  = n_sensors
        for i in range(n_levels):
            out_ch = n_filters * (2 ** min(i, 2))
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)

        out_dim  = 1 if task == "rul" else n_classes
        self.head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(in_ch // 2, out_dim),
        )
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.network(x)
        g   = self.pool(x).squeeze(-1)
        out = self.head(g)
        return out.squeeze(-1) if self.task == "rul" else out
