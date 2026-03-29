"""
transformer.py — Multi-Head Transformer Encoder for Sensor Channel Attention
=============================================================================
Architecture: Multi-head self-attention across 16 sensor channels.
    Input: (B, 16, T) — each sensor is a "token" with signal as embedding.
    Positional encoding encodes sensor spatial position.

Configuration: 4 heads, d_model=128, FFN=256, 3 encoder layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SensorPositionalEncoding(nn.Module):
    """Learnable positional encoding for sensor spatial positions.

    Encodes the 4×4 grid position of each PZT sensor using a
    learnable embedding combined with sinusoidal spatial encoding.

    Args:
        n_sensors: Number of sensors (16).
        d_model: Model embedding dimension.
    """

    def __init__(self, n_sensors: int = 16, d_model: int = 128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_sensors, d_model) * 0.02)

        # Sinusoidal positional encoding (fixed component)
        pe = torch.zeros(n_sensors, d_model)
        position = torch.arange(0, n_sensors, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe_fixed", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sensor embeddings.

        Args:
            x: Tensor of shape (B, 16, d_model).

        Returns:
            Tensor of shape (B, 16, d_model) with positional info.
        """
        return x + self.pos_embed + self.pe_fixed


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with multi-head self-attention.

    Implements:
        LayerNorm → MultiHead Self-Attention → Residual
        → LayerNorm → FFN → Residual

    Uses Pre-LayerNorm configuration for training stability.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward network hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional attention weight return.

        Args:
            x: Input tensor of shape (B, S, d_model).
            return_attn: If True, return attention weights.

        Returns:
            Tuple of (output tensor, attention weights or None).
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed,
                                            need_weights=return_attn)
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights


class SensorTransformer(nn.Module):
    """Multi-Head Transformer Encoder for CFRP sensor channel attention.

    Treats each of the 16 PZT sensors as a "token" with its Lamb wave
    signal as the embedding. Self-attention learns inter-sensor
    relationships and damage-induced path changes.

    Architecture:
        Input: (B, 16, T)
        → Signal Embedding: Conv1D(T) → d_model per sensor
        → Positional Encoding (sensor spatial position)
        → 3 × Transformer Encoder Block (4 heads, d_model=128, FFN=256)
        → Global Pooling (mean over sensors)
        → Classification/Regression Head

    Args:
        n_sensors: Number of input sensors (16).
        signal_length: Signal length per sensor.
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        n_layers: Number of encoder layers.
        n_classes: Output classes or 1 for regression.
        task: "classification" or "rul".
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_sensors: int = 17,
        signal_length: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        n_classes: int = 5,
        task: str = "classification",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.n_layers = n_layers

        # Signal embedding: project each sensor’s feature vector to d_model.
        # Input per sensor: (B*n_sensors, signal_length) where signal_length=16.
        # A direct linear projection is architecturally correct for tabular data.
        self.signal_embed = nn.Sequential(
            nn.Linear(signal_length, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = SensorPositionalEncoding(n_sensors, d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Classification: (B, n_classes) logits
            Regression: (B,) RUL predictions
        """
        out, _ = self._forward_impl(x, return_attn=False)
        return out

    def _forward_impl(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Implementation of forward pass with optional attention return.

        Args:
            x: Input (B, 16, T).
            return_attn: Whether to collect attention weights.

        Returns:
            Tuple of (output, list_of_attention_weights or None).
        """
        B, S, T = x.shape
        all_attn = [] if return_attn else None

        # Embed each sensor’s feature vector independently.
        # Reshape: (B, S, T) → (B*S, T) → Linear → (B*S, d_model)
        x_flat = x.reshape(B * S, T)           # (B*17, 16)
        embeddings = self.signal_embed(x_flat)  # (B*17, d_model)
        embeddings = embeddings.reshape(B, S, self.d_model)  # (B, 17, d_model)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Transformer encoder layers
        hidden = embeddings
        for layer in self.encoder_layers:
            hidden, attn_w = layer(hidden, return_attn=return_attn)
            if return_attn and attn_w is not None:
                all_attn.append(attn_w)

        # Final normalization
        hidden = self.final_norm(hidden)  # (B, 16, d_model)

        # Global mean pooling over sensor dimension
        pooled = hidden.mean(dim=1)  # (B, d_model)

        # Output
        out = self.head(pooled)
        if self.task == "rul":
            out = out.squeeze(-1)

        return out, all_attn

    def forward_with_attention(self, x: torch.Tensor):
        """Forward pass returning all layer attention weights.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Tuple of (output, list of attention weight tensors).
            Each attention tensor has shape (B, n_heads, 16, 16).
        """
        return self._forward_impl(x, return_attn=True)

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Extract attention maps for visualization.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            List of attention weight arrays, one per layer.
            Each array has shape (B, n_heads, 16, 16).
        """
        self.eval()
        with torch.no_grad():
            _, attn_maps = self.forward_with_attention(x)
        return [a.cpu().numpy() for a in attn_maps] if attn_maps else []

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Monte Carlo Dropout prediction with uncertainty.

        Args:
            x: Input tensor.
            n_samples: Number of MC forward passes.

        Returns:
            Tuple of (mean, std, all_predictions).
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())

        predictions = np.stack(predictions, axis=0)
        return predictions.mean(axis=0), predictions.std(axis=0), predictions

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
