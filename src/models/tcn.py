"""
tcn.py — Temporal Convolutional Network with Residual Connections
=================================================================
Dilated causal convolutions for long-range temporal dependencies
in fatigue damage progression modeling.

Reference:
    Bai S., Kolter J.Z., Koltun V., "An Empirical Evaluation of
    Generic Convolutional and Recurrent Networks for Sequence Modeling",
    arXiv:1803.01271, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation.

    Ensures no future information leakage by left-padding the input.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal trimming.

        Args:
            x: Tensor of shape (B, C, T).

        Returns:
            Tensor of shape (B, C_out, T).
        """
        out = self.conv(x)
        # Remove right padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNResidualBlock(nn.Module):
    """TCN residual block with two dilated causal convolutions.

    Architecture:
        x → CausalConv → WeightNorm → GELU → Dropout
          → CausalConv → WeightNorm → GELU → Dropout
          + Skip Connection (1×1 conv if channel mismatch)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size for dilated convolutions.
        dilation: Dilation factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Skip connection (1×1 conv for channel adaptation)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Tensor of shape (B, C_in, T).

        Returns:
            Tensor of shape (B, C_out, T).
        """
        residual = self.skip(x)
        out = self.net(x)
        # Match temporal dimensions
        min_len = min(out.shape[-1], residual.shape[-1])
        return F.gelu(out[:, :, :min_len] + residual[:, :, :min_len])


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for CFRP damage sequence modeling.

    Uses exponentially increasing dilation factors for
    receptive field growth: d = 2^i for layer i.

    Architecture:
        Input (B, 16, T)
        → TCNResidualBlock(16→64, d=1)
        → TCNResidualBlock(64→64, d=2)
        → TCNResidualBlock(64→128, d=4)
        → TCNResidualBlock(128→128, d=8)
        → Global Average Pooling
        → Dense → Output

    Args:
        n_sensors: Number of input channels (16).
        signal_length: Signal length.
        channels: List of channel sizes for each TCN block.
        kernel_size: Kernel size for dilated convolutions.
        n_classes: Output classes or 1 for regression.
        task: "classification" or "rul".
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_sensors: int = 17,
        signal_length: int = 16,
        channels: list = None,
        kernel_size: int = 2,
        n_classes: int = 5,
        task: str = "classification",
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 128, 128]
        self.task = task

        # Build TCN blocks with exponential dilation
        layers = []
        in_ch = n_sensors
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Output head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Classification: (B, n_classes) logits
            Regression: (B,) RUL predictions
        """
        features = self.tcn(x)             # (B, C, T')
        pooled = self.gap(features).squeeze(-1)  # (B, C)
        out = self.head(pooled)

        if self.task == "rul":
            return out.squeeze(-1)
        return out

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Monte Carlo Dropout prediction with uncertainty."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(x).cpu().numpy())

        predictions = np.stack(predictions, axis=0)
        return predictions.mean(axis=0), predictions.std(axis=0), predictions

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
