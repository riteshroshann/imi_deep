"""
bilstm.py — Bi-directional LSTM with Attention Gate
====================================================
Temporal damage progression modeling from Lamb wave signal sequences.

Architecture:
    Input (B, 16, T) → reshape → BiLSTM (2 layers, hidden=128)
    → Attention Gate → Dense → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AttentionGate(nn.Module):
    """Additive attention mechanism over LSTM hidden states.

    Computes context vector as weighted sum of hidden states:
        e_t = v^T · tanh(W·h_t + b)
        α_t = softmax(e_t)
        c = Σ α_t · h_t

    Args:
        hidden_dim: Dimension of hidden states.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            hidden_states: Tensor of shape (B, T, H).

        Returns:
            Tuple of (context_vector (B, H), attention_weights (B, T)).
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (B, T)

        # Softmax normalization
        weights = F.softmax(scores, dim=-1)  # (B, T)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, H)

        return context, weights


class BiLSTMAttention(nn.Module):
    """Bi-directional LSTM with attention gate for temporal damage modeling.

    Architecture:
        Input: (B, 16, T) — multi-channel sensor signals
        → Temporal feature extraction: Conv1D(16→32) for initial encoding
        → BiLSTM (2 layers, hidden=128 per direction, total=256)
        → Attention Gate
        → Dense(256→128) → GELU → Dropout → Output

    Args:
        n_sensors: Number of input sensor channels.
        signal_length: Signal length per sensor.
        hidden_dim: LSTM hidden dimension per direction.
        n_layers: Number of LSTM layers.
        n_classes: Output classes (classification) or 1 (regression).
        task: "classification" or "rul".
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_sensors: int = 16,
        signal_length: int = 1024,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = 5,
        task: str = "classification",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task = task
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Initial temporal feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_sensors, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
        )

        # Downsampled length after pooling
        self.seq_len = signal_length // 4

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Attention gate over BiLSTM outputs
        self.attention = AttentionGate(hidden_dim * 2)  # *2 for bidirectional

        # Output head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM with orthogonal weight initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Classification: (B, n_classes) logits
            Regression: (B,) RUL predictions
        """
        # Initial convolution for temporal feature extraction
        features = self.input_conv(x)  # (B, 32, T//4)

        # Transpose for LSTM: (B, T//4, 32)
        features = features.permute(0, 2, 1)

        # BiLSTM processing
        lstm_out, _ = self.lstm(features)  # (B, T//4, hidden*2)

        # Attention-weighted context
        context, _ = self.attention(lstm_out)  # (B, hidden*2)

        # Output
        out = self.head(context)

        if self.task == "rul":
            return out.squeeze(-1)
        return out

    def forward_with_attention(self, x: torch.Tensor):
        """Forward pass returning attention weights for visualization.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Tuple of (output, attention_weights).
        """
        features = self.input_conv(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        context, attn_weights = self.attention(lstm_out)
        out = self.head(context)
        if self.task == "rul":
            out = out.squeeze(-1)
        return out, attn_weights

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
        # Enable dropout during inference
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
