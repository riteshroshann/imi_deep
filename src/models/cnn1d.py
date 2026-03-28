"""
cnn1d.py — 1D Convolutional Neural Network for Lamb Wave Classification & RUL
===============================================================================
Architecture: Conv1D × 4 → BatchNorm → MaxPool → GlobalAvgPool → Dense

Multi-channel input from all 16 PZT sensors. Supports both damage state
classification (5 classes) and RUL regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv1DBlock(nn.Module):
    """Single 1D convolution block with BatchNorm, activation, and MaxPool.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        pool_size: MaxPool kernel size. Set to 0 to skip pooling.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C_in, T).

        Returns:
            Tensor of shape (B, C_out, T // pool_size).
        """
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    """1D-CNN for multi-channel Lamb wave signal processing.

    Architecture:
        Input (B, 16, T)
          → Conv1D(16→32) → BN → GELU → MaxPool(2)
          → Conv1D(32→64) → BN → GELU → MaxPool(2)
          → Conv1D(64→128) → BN → GELU → MaxPool(2)
          → Conv1D(128→256) → BN → GELU → MaxPool(2)
          → GlobalAvgPool
          → Dense(256→128) → GELU → Dropout
          → Dense(128→output_dim)

    Args:
        n_sensors: Number of input sensor channels (default: 16).
        signal_length: Length of input signals.
        n_classes: Number of output classes (classification) or 1 (regression).
        task: "classification" or "rul" (regression).
        dropout: Dropout probability.
        mc_dropout: If True, apply dropout during inference (MC Dropout).
    """

    def __init__(
        self,
        n_sensors: int = 16,
        signal_length: int = 1024,
        n_classes: int = 5,
        task: str = "classification",
        dropout: float = 0.2,
        mc_dropout: bool = False,
    ):
        super().__init__()
        self.task = task
        self.mc_dropout = mc_dropout
        self.n_classes = n_classes

        # 4 convolutional blocks with increasing channel depth
        self.conv_blocks = nn.Sequential(
            Conv1DBlock(n_sensors, 32, kernel_size=7, pool_size=2, dropout=dropout),
            Conv1DBlock(32, 64, kernel_size=5, pool_size=2, dropout=dropout),
            Conv1DBlock(64, 128, kernel_size=5, pool_size=2, dropout=dropout),
            Conv1DBlock(128, 256, kernel_size=3, pool_size=2, dropout=dropout),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier / Regressor head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 16, T) — multi-channel signals.

        Returns:
            Classification: (B, n_classes) logits
            Regression: (B,) RUL predictions
        """
        # Convolutional feature extraction
        features = self.conv_blocks(x)  # (B, 256, T')

        # Global average pooling
        pooled = self.gap(features).squeeze(-1)  # (B, 256)

        # Output head
        out = self.head(pooled)  # (B, n_classes) or (B, 1)

        if self.task == "rul":
            return out.squeeze(-1)
        return out

    def forward_with_features(self, x: torch.Tensor):
        """Forward pass returning intermediate features for Grad-CAM.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Tuple of (output, conv_features, pooled_features).
        """
        features = self.conv_blocks(x)
        pooled = self.gap(features).squeeze(-1)
        out = self.head(pooled)
        if self.task == "rul":
            out = out.squeeze(-1)
        return out, features, pooled

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        self.mc_dropout = True
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple:
        """Monte Carlo Dropout prediction with uncertainty.

        Args:
            x: Input tensor.
            n_samples: Number of MC forward passes.

        Returns:
            Tuple of (mean_prediction, std_prediction, all_predictions).
        """
        self.enable_mc_dropout()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())

        predictions = np.stack(predictions, axis=0)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std, predictions

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


import numpy as np  # For predict_with_uncertainty
