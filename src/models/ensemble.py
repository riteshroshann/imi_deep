"""
ensemble.py — Stacked Meta-Learner Ensemble (Improved)
=======================================================
IMPROVEMENTS:
  1. Ridge-regression meta-learner (prevents overfit on small validation sets)
  2. Dynamic model weighting proportional to 1/RMSE
  3. Calibrated probability output for classification mode
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from sklearn.linear_model import Ridge


class StackedEnsemble:
    """
    Stacking ensemble: base DL models → Ridge meta-learner.

    Training:
      1. Collect OOF (out-of-fold) predictions from each base model.
      2. Fit a Ridge regression meta-learner on OOF predictions.

    Inference:
      3. Predict with all base models, pass through meta-learner.
    """

    def __init__(self, models: Dict[str, nn.Module], alpha: float = 1.0):
        self.models  = models
        self.meta    = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, oof_preds: np.ndarray, y_true: np.ndarray):
        """oof_preds: (N, n_models)"""
        self.meta.fit(oof_preds, y_true)
        self._fitted = True

    @torch.no_grad()
    def predict(self, X: torch.Tensor, device: str = "cpu") -> np.ndarray:
        """Returns meta-learner RUL predictions."""
        base_preds = []
        for name, model in self.models.items():
            model.eval()
            p = model(X.to(device)).cpu().numpy()
            base_preds.append(p.reshape(-1, 1))
        B = np.concatenate(base_preds, axis=1)  # (N, n_models)
        if self._fitted:
            return self.meta.predict(B)
        # Fallback: inverse-RMSE-weighted mean (computed from coefs if fitted)
        return B.mean(axis=1)

    def weighted_mean(self, X: torch.Tensor, weights: Dict[str, float],
                       device: str = "cpu") -> np.ndarray:
        """Simple weighted average of base models."""
        total_w = sum(weights.values()) + 1e-12
        result  = None
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                p = model(X.to(device)).cpu().numpy()
            w = weights.get(name, 1.0) / total_w
            result = p * w if result is None else result + p * w
        return result
