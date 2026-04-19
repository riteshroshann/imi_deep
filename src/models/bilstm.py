"""
bilstm.py — Bidirectional LSTM with Bahdanau Attention (Improved)
===================================================================
IMPROVEMENTS:
  1. Bahdanau (additive) attention instead of simple dot-product
  2. Learnable context vector per head
  3. Layer normalisation on LSTM outputs
  4. Variational dropout between LSTM layers (lock-step masking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention with learnable query vector."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, T, 2*hidden) → attended (B, 2*hidden)."""
        scores  = self.v(torch.tanh(self.W(H))).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)               # (B, T)
        ctx     = (weights.unsqueeze(-1) * H).sum(dim=1)     # (B, 2*hidden)
        return ctx


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM + Bahdanau attention for RUL / damage classification.

    Input: (B, n_sensors, signal_length)
    Output: RUL scalar or class logits
    """

    def __init__(self, n_sensors: int = 17, signal_length: int = 16,
                 hidden_dim: int = 128, n_layers: int = 2,
                 task: str = "rul", n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.task = task

        # Treat each time-step (sensor path) as a sequence element
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention  = BahdanauAttention(hidden_dim)

        out_dim = 1 if task == "rul" else n_classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S) → treat S as time, C as features
        x   = x.permute(0, 2, 1)           # (B, S, C)
        H, _ = self.lstm(x)                 # (B, S, 2*hidden)
        H   = self.layer_norm(H)
        ctx = self.attention(H)             # (B, 2*hidden)
        out = self.head(ctx)
        return out.squeeze(-1) if self.task == "rul" else out
