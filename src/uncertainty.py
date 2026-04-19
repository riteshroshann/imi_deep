"""
uncertainty.py — Uncertainty Quantification (Improved)
=======================================================
IMPROVEMENTS:
  1. MC Dropout uses eval-mode BN but keeps dropout active (correct Gal & Ghahramani)
  2. ConformalPredictor uses split-conformal (faster, fewer assumptions)
  3. comprehensive_rul_metrics adds NASA PHM scoring function
  4. compute_calibration_curve now returns Expected Calibration Error (ECE)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


# ── Metrics ────────────────────────────────────────────────────────────────────
def comprehensive_rul_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Standard regression metrics + NASA PHM scoring function.

    NASA score (lower = better):
        S = Σ exp(-d_i / 13) - 1  if d_i < 0  (early prediction)
        S = Σ exp( d_i / 10) - 1  if d_i ≥ 0  (late prediction)
    where d_i = ŷ_i - y_i (prediction error in RUL units).
    """
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    valid  = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred, y_true = y_pred[valid], y_true[valid]

    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "nasa_score": np.nan}

    residuals = y_pred - y_true
    mse  = np.mean(residuals ** 2)
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1.0 - ss_res / (ss_tot + 1e-12))

    # NASA PHM score
    nasa_score = float(np.sum(
        np.where(residuals < 0,
                 np.exp(-residuals / 13.0) - 1.0,
                 np.exp( residuals / 10.0) - 1.0)
    ))

    return {"rmse": rmse, "mae": mae, "r2": r2, "nasa_score": nasa_score}


# ── MC Dropout ─────────────────────────────────────────────────────────────────
def _enable_dropout(model: nn.Module):
    """Set dropout layers to train mode while keeping BN in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 50,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    MC Dropout epistemic uncertainty estimate.
    BN layers remain in eval; only Dropout layers are active.
    """
    model.eval()
    _enable_dropout(model)

    preds = []
    x     = x.to(device)
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            preds.append(out.cpu().numpy())

    preds = np.stack(preds, axis=0)   # (T, N)
    return {
        "mean": preds.mean(axis=0),
        "std":  preds.std(axis=0),
        "samples": preds,
    }


# ── Conformal Prediction ───────────────────────────────────────────────────────
class ConformalPredictor:
    """
    Split-conformal prediction intervals.

    Fits on a calibration split, then produces distribution-free
    coverage-guaranteed intervals at inference time.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha      = alpha
        self._q_hat: Optional[float] = None

    def calibrate(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Compute the conformal score quantile on a held-out calibration set."""
        scores      = np.abs(y_pred - y_true)
        n           = len(scores)
        q_level     = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level     = min(q_level, 1.0)
        self._q_hat = float(np.quantile(scores, q_level))

    def predict_interval(
        self, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction intervals."""
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        lo = y_pred - self._q_hat
        hi = y_pred + self._q_hat
        return lo, hi

    def evaluate_coverage(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        lo, hi = self.predict_interval(y_pred)
        covered = ((y_true >= lo) & (y_true <= hi)).mean()
        avg_width = float((hi - lo).mean())
        return {
            "empirical_coverage": float(covered),
            "target_coverage":    1.0 - self.alpha,
            "avg_interval_width": avg_width,
            "q_hat":              self._q_hat or 0.0,
        }


def compute_calibration_curve(
    y_pred: np.ndarray,
    y_std:  np.ndarray,
    y_true: np.ndarray,
    confidence_levels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration curve and ECE for Gaussian predictive distribution.
    """
    if confidence_levels is None:
        confidence_levels = np.linspace(0.1, 0.99, 20)

    observed = []
    for conf in confidence_levels:
        z   = abs(y_pred - y_true) / (y_std + 1e-8)
        # Gaussian quantile for this confidence level
        from scipy.stats import norm
        threshold = norm.ppf((1 + conf) / 2)
        observed.append(float((z <= threshold).mean()))

    observed = np.array(observed)
    ece      = float(np.mean(np.abs(observed - confidence_levels)))

    return {
        "expected_coverage": confidence_levels,
        "observed_coverage": observed,
        "ece":               ece,
    }
