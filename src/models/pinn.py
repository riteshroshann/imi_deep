"""
pinn.py — Physics-Informed Neural Network for Property Degradation
===================================================================
Embeds Paris Law (da/dN = C·ΔK^m) as a physics constraint in the
loss function to predict stiffness degradation E(N)/E₀ and residual
strength σ_r(N)/σ₀ using Continuum Damage Mechanics (CDM) coupling.

The physics loss regularizes extrapolation beyond the training regime,
enforcing physically consistent degradation curves.

References:
    Raissi M., Perdikaris P., Karniadakis G.E., "Physics-informed neural
    networks: A deep learning framework for solving forward and inverse
    problems involving nonlinear partial differential equations",
    J. Computational Physics, 378, 686–707, 2019.
    
    Paris P., Erdogan F., "A critical analysis of crack growth laws",
    J. Basic Engineering, 85(4), 528–533, 1963.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class PINNBlock(nn.Module):
    """Residual block for the PINN backbone.

    Args:
        dim: Input and output dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return torch.tanh(x + self.net(x))


class PhysicsInformedNet(nn.Module):
    """Physics-Informed Neural Network for CFRP property degradation.

    Predicts stiffness ratio E(N)/E₀ and strength ratio σ_r(N)/σ₀
    as functions of:
        - Fatigue cycle count N
        - Accumulated damage features
        - Layup configuration

    The loss function combines data-driven and physics terms:
        L_total = L_data + λ · L_physics

    where:
        L_data = MSE(predicted_property, actual_property)
        L_physics = MSE(dE/dN_predicted, Paris_Law_prediction)

    Paris Law: da/dN = C · (ΔK)^m
    where C and m are material constants, ΔK is the stress intensity
    factor range. The crack growth rate is related to stiffness
    degradation through CDM coupling:
        dE/dN = -E₀ · f(D) · da/dN

    Args:
        input_dim: Number of input features (cycles + damage features).
        hidden_dim: Hidden layer dimension.
        n_blocks: Number of residual blocks.
        output_dim: Number of outputs (2: stiffness + strength ratios).
        dropout: Dropout probability.
        paris_C: Paris Law constant C (log scale).
        paris_m: Paris Law exponent m.
        lambda_physics: Physics loss weight.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        output_dim: int = 2,
        dropout: float = 0.1,
        paris_C: float = 1e-10,
        paris_m: float = 3.0,
        lambda_physics: float = 0.1,
    ):
        super().__init__()
        self.paris_C = paris_C
        self.paris_m = paris_m
        self.lambda_physics = lambda_physics
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Residual backbone (Tanh activation for smoothness constraint)
        self.blocks = nn.ModuleList([
            PINNBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        # Output heads
        self.stiffness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # E(N)/E₀ ∈ [0, 1]
        )

        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # σ_r(N)/σ₀ ∈ [0, 1]
        )

        # Learnable Paris Law parameters (for fine-tuning)
        self.log_C = nn.Parameter(torch.tensor(np.log(paris_C)))
        self.m_param = nn.Parameter(torch.tensor(paris_m))

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for smooth initial predictions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting stiffness and strength ratios.

        Args:
            x: Input features of shape (B, input_dim).
                Expected: [N/N_max, DI_features..., layup_encoding...]

        Returns:
            Tuple of (stiffness_ratio, strength_ratio), each (B,).
        """
        hidden = self.input_proj(x)

        for block in self.blocks:
            hidden = block(hidden)

        stiffness = self.stiffness_head(hidden).squeeze(-1)  # (B,)
        strength = self.strength_head(hidden).squeeze(-1)     # (B,)

        return stiffness, strength

    def physics_loss(
        self,
        x: torch.Tensor,
        stiffness_pred: torch.Tensor,
        cycles_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics-based loss using Paris Law constraint.

        Paris Law: da/dN = C · (ΔK)^m
        CDM coupling: dE/dN ∝ -D_rate ~ -C·(ΔK)^m

        We enforce that the predicted stiffness degradation rate
        is consistent with the Paris Law prediction.

        The derivative dE/dN is computed via automatic differentiation.

        Args:
            x: Input features (B, input_dim).
            stiffness_pred: Predicted E(N)/E₀ of shape (B,).
            cycles_normalized: Normalized cycle count N/N_max of shape (B,).

        Returns:
            Physics loss scalar.
        """
        # Compute dE/dN via automatic differentiation
        cycles_normalized.requires_grad_(True)

        # Create modified input with grad-tracked cycles
        x_grad = x.clone()
        x_grad[:, 0] = cycles_normalized

        stiff_pred, _ = self.forward(x_grad)

        # Automatic differentiation: ∂E/∂(N/N_max)
        dE_dN = torch.autograd.grad(
            outputs=stiff_pred,
            inputs=cycles_normalized,
            grad_outputs=torch.ones_like(stiff_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Paris Law prediction for degradation rate
        # ΔK is approximated from current damage state
        C = torch.exp(self.log_C)
        m = self.m_param

        # Stress intensity factor (approximated from stiffness ratio)
        delta_K = (1.0 - stiff_pred + 0.1) * 10.0  # Normalized SIF
        paris_rate = -C * torch.pow(delta_K.abs() + 1e-8, m)

        # Physics loss: predicted rate should match Paris Law rate
        physics_loss = F.mse_loss(dE_dN, paris_rate)

        # Additional constraint: monotonic degradation (dE/dN ≤ 0)
        monotonic_penalty = F.relu(dE_dN).mean()

        return physics_loss + 0.5 * monotonic_penalty

    def total_loss(
        self,
        x: torch.Tensor,
        stiffness_true: torch.Tensor,
        strength_true: torch.Tensor,
        cycles_normalized: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss combining data-driven and physics terms.

        L_total = L_data_stiffness + L_data_strength + λ · L_physics

        Args:
            x: Input features (B, input_dim).
            stiffness_true: True E(N)/E₀ of shape (B,).
            strength_true: True σ_r(N)/σ₀ of shape (B,).
            cycles_normalized: Normalized cycle count of shape (B,).

        Returns:
            Dictionary with {total, data, physics} loss values.
        """
        stiffness_pred, strength_pred = self.forward(x)

        # Data loss
        loss_stiffness = F.mse_loss(stiffness_pred, stiffness_true)
        loss_strength = F.mse_loss(strength_pred, strength_true)
        data_loss = loss_stiffness + loss_strength

        # Physics loss
        try:
            phys_loss = self.physics_loss(x, stiffness_pred, cycles_normalized)
        except RuntimeError:
            # Fallback if autograd fails
            phys_loss = torch.tensor(0.0, device=x.device)

        # Total loss
        total = data_loss + self.lambda_physics * phys_loss

        return {
            "total": total,
            "data": data_loss,
            "physics": phys_loss,
            "stiffness_loss": loss_stiffness,
            "strength_loss": loss_strength,
        }

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PINNTrainer:
    """Dedicated trainer for the PINN with physics loss scheduling.

    Implements curriculum learning: physics loss weight increases
    over training to first learn data patterns, then enforce physics.

    Args:
        model: PhysicsInformedNet instance.
        lr: Learning rate.
        weight_decay: L2 regularization.
        lambda_schedule: "constant", "linear", or "cosine" for physics weight.
        max_epochs: Total number of training epochs.
    """

    def __init__(
        self,
        model: PhysicsInformedNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_schedule: str = "linear",
        max_epochs: int = 500,
    ):
        self.model = model
        self.lambda_schedule = lambda_schedule
        self.max_epochs = max_epochs
        self.initial_lambda = model.lambda_physics

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        self.history = {
            "total_loss": [], "data_loss": [], "physics_loss": [],
            "val_loss": [], "lr": [],
        }

    def _get_lambda(self, epoch: int) -> float:
        """Compute physics loss weight for current epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Physics loss weight λ.
        """
        frac = epoch / max(1, self.max_epochs)

        if self.lambda_schedule == "constant":
            return self.initial_lambda
        elif self.lambda_schedule == "linear":
            # Linearly increase from 0.01 to initial_lambda
            return 0.01 + (self.initial_lambda - 0.01) * frac
        elif self.lambda_schedule == "cosine":
            # Cosine warmup
            return self.initial_lambda * (1 - np.cos(np.pi * frac)) / 2
        return self.initial_lambda

    def train_epoch(
        self,
        train_data: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_data: Dictionary with {x, stiffness, strength, cycles}.
            epoch: Current epoch number.

        Returns:
            Dictionary of loss values.
        """
        self.model.train()

        # Update physics weight
        self.model.lambda_physics = self._get_lambda(epoch)

        x = train_data["x"]
        stiffness = train_data["stiffness"]
        strength = train_data["strength"]
        cycles = train_data["cycles"]

        self.optimizer.zero_grad()
        losses = self.model.total_loss(x, stiffness, strength, cycles)
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Record history
        result = {k: v.item() for k, v in losses.items()}
        self.history["total_loss"].append(result["total"])
        self.history["data_loss"].append(result["data"])
        self.history["physics_loss"].append(result.get("physics", 0.0))
        self.history["lr"].append(self.scheduler.get_last_lr()[0])

        return result
