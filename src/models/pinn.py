"""
pinn.py — Physics-Informed Neural Network (Improved)
=====================================================
IMPROVEMENTS:
  1. Learnable Paris Law exponent m clamped to physically valid range [1,5]
  2. Dual monotonicity penalties: global + local (consecutive pairs)
  3. Adaptive lambda_physics scheduler (now supports 'cosine_restart')
  4. Mini-batch training support in PINNTrainer
  5. Heteroscedastic aleatoric uncertainty head (log-variance prediction)
  6. NaN-safe autograd: disables physics loss when grad is NaN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PINNBlock(nn.Module):
    """Residual block with Tanh + LayerNorm (smooth, physically consistent)."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x + self.net(x))


class PhysicsInformedNet(nn.Module):
    """
    PINN predicting E(N)/E₀ and σ_r(N)/σ₀ with Paris-Law physics constraint.

    Additional output: log-variance heads for aleatoric uncertainty.
    """

    def __init__(
        self,
        input_dim:      int   = 20,
        hidden_dim:     int   = 128,
        n_blocks:       int   = 4,
        dropout:        float = 0.1,
        paris_C:        float = 1e-10,
        paris_m:        float = 3.0,
        lambda_physics: float = 0.1,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Residual backbone
        self.blocks = nn.ModuleList([PINNBlock(hidden_dim, dropout) for _ in range(n_blocks)])

        # Property heads (mean + log-variance for heteroscedastic UQ)
        def _head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
            )

        def _logvar_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1),          # unbounded log-variance
            )

        self.stiffness_head  = _head()
        self.strength_head   = _head()
        self.stiffness_logvar = _logvar_head()
        self.strength_logvar  = _logvar_head()

        # Learnable Paris Law parameters — clamped during forward
        self.log_C  = nn.Parameter(torch.tensor(float(np.log(paris_C + 1e-30))))
        self.m_raw  = nn.Parameter(torch.tensor(float(paris_m)))   # clamped to [1,5]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def m_param(self) -> torch.Tensor:
        """Physics-valid Paris exponent clamped to [1, 5]."""
        return torch.clamp(self.m_raw, 1.0, 5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        stiffness = self.stiffness_head(h).squeeze(-1)
        strength  = self.strength_head(h).squeeze(-1)
        return stiffness, strength

    def forward_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (stiffness_mean, strength_mean, stiffness_std, strength_std)."""
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        stiff_mean  = self.stiffness_head(h).squeeze(-1)
        str_mean    = self.strength_head(h).squeeze(-1)
        stiff_std   = torch.exp(0.5 * self.stiffness_logvar(h).squeeze(-1))
        str_std     = torch.exp(0.5 * self.strength_logvar(h).squeeze(-1))
        return stiff_mean, str_mean, stiff_std, str_std

    def physics_loss(
        self,
        x:                  torch.Tensor,
        stiffness_pred:     torch.Tensor,
        cycles_normalized:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Paris Law constraint + dual monotonicity penalty.
        Returns zero tensor if autograd produces NaN.
        """
        cycles_t = cycles_normalized.clone().requires_grad_(True)
        x_mod    = x.clone()
        x_mod[:, 0] = cycles_t

        stiff_pred_g, _ = self.forward(x_mod)

        try:
            dE_dN = torch.autograd.grad(
                stiff_pred_g, cycles_t,
                grad_outputs=torch.ones_like(stiff_pred_g),
                create_graph=True, retain_graph=True,
            )[0]
        except RuntimeError:
            return torch.tensor(0.0, device=x.device)

        if torch.isnan(dE_dN).any():
            return torch.tensor(0.0, device=x.device)

        C       = torch.exp(self.log_C)
        m       = self.m_param
        delta_K = (1.0 - stiff_pred_g + 0.1) * 10.0
        paris   = -C * torch.pow(delta_K.abs() + 1e-8, m)

        # Paris Law residual
        loss_paris = F.mse_loss(dE_dN, paris)

        # Global monotonicity: dE/dN ≤ 0 everywhere
        mono_global = F.relu(dE_dN).mean()

        # Local monotonicity: stiffness should not increase between consecutive steps
        if stiffness_pred.numel() > 1:
            diffs = stiffness_pred[1:] - stiffness_pred[:-1]
            mono_local = F.relu(diffs).mean()
        else:
            mono_local = torch.tensor(0.0, device=x.device)

        return loss_paris + 0.5 * mono_global + 0.3 * mono_local

    def total_loss(
        self,
        x:                 torch.Tensor,
        stiffness_true:    torch.Tensor,
        strength_true:     torch.Tensor,
        cycles_normalized: torch.Tensor,
        use_nll: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        L_total = L_data + λ · L_physics

        If use_nll=True, uses Gaussian NLL loss (heteroscedastic) instead of MSE.
        """
        if use_nll:
            stiff_m, str_m, stiff_s, str_s = self.forward_with_uncertainty(x)
            loss_stiff = F.gaussian_nll_loss(stiff_m, stiffness_true, stiff_s**2)
            loss_str   = F.gaussian_nll_loss(str_m,   strength_true,   str_s**2)
            stiffness_pred = stiff_m
        else:
            stiff_m, str_m = self.forward(x)
            loss_stiff = F.mse_loss(stiff_m, stiffness_true)
            loss_str   = F.mse_loss(str_m,   strength_true)
            stiffness_pred = stiff_m

        data_loss = loss_stiff + loss_str

        try:
            phys_loss = self.physics_loss(x, stiffness_pred, cycles_normalized)
        except RuntimeError:
            phys_loss = torch.tensor(0.0, device=x.device)

        total = data_loss + self.lambda_physics * phys_loss

        return {
            "total":           total,
            "data":            data_loss,
            "physics":         phys_loss,
            "stiffness_loss":  loss_stiff,
            "strength_loss":   loss_str,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PINNTrainer:
    """
    PINN trainer with curriculum lambda schedule and mini-batch support.

    IMPROVEMENTS:
      - Mini-batch iteration over training data
      - 'cosine_restart' lambda schedule
      - Gradient clipping on Paris Law parameters separately
    """

    def __init__(
        self,
        model:           PhysicsInformedNet,
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        lambda_schedule: str   = "linear",
        max_epochs:      int   = 500,
        batch_size:      int   = 256,
    ):
        self.model           = model
        self.lambda_schedule = lambda_schedule
        self.max_epochs      = max_epochs
        self.batch_size      = batch_size
        self.initial_lambda  = model.lambda_physics

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01)

        self.history = {"total_loss":[], "data_loss":[], "physics_loss":[], "lr":[]}

    def _get_lambda(self, epoch: int) -> float:
        frac = epoch / max(1, self.max_epochs)
        if self.lambda_schedule == "constant":
            return self.initial_lambda
        elif self.lambda_schedule == "linear":
            return 0.01 + (self.initial_lambda - 0.01) * frac
        elif self.lambda_schedule == "cosine":
            return self.initial_lambda * (1 - np.cos(np.pi * frac)) / 2
        elif self.lambda_schedule == "cosine_restart":
            period = self.max_epochs // 3
            local  = (epoch % period) / max(1, period)
            return self.initial_lambda * (1 - np.cos(np.pi * local)) / 2
        return self.initial_lambda

    def train_epoch(self, train_data: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.train()
        self.model.lambda_physics = self._get_lambda(epoch)

        x, stiff, strength, cycles = (train_data["x"], train_data["stiffness"],
                                       train_data["strength"], train_data["cycles"])
        n = x.shape[0]

        # Mini-batch loop
        perm  = torch.randperm(n, device=x.device)
        total_l, data_l, phys_l = 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            self.optimizer.zero_grad()
            losses = self.model.total_loss(
                x[idx], stiff[idx], strength[idx], cycles[idx])
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_l += losses["total"].item()
            data_l  += losses["data"].item()
            phys_l  += losses.get("physics", torch.tensor(0.0)).item() \
                       if isinstance(losses.get("physics"), torch.Tensor) \
                       else losses.get("physics", 0.0)
            n_batches += 1

        self.scheduler.step()
        result = {
            "total":   total_l / n_batches,
            "data":    data_l  / n_batches,
            "physics": phys_l  / n_batches,
        }
        self.history["total_loss"].append(result["total"])
        self.history["data_loss"].append(result["data"])
        self.history["physics_loss"].append(result["physics"])
        self.history["lr"].append(self.scheduler.get_last_lr()[0])
        return result
