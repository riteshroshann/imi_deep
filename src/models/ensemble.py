"""
ensemble.py — Stacked Ensemble (CNN + BiLSTM + Transformer)
============================================================
Meta-learner: Ridge regression stacking of Model A (CNN), 
Model B (BiLSTM), and Model D (Transformer) predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import RidgeCV


class StackedEnsemble(nn.Module):
    """Stacked ensemble combining multiple deep learning models.

    Level 0: CNN1D, BiLSTMAttention, SensorTransformer (base learners)
    Level 1: Ridge regression meta-learner over base predictions

    The meta-learner is trained on out-of-fold predictions from the
    base models to avoid overfitting.

    Args:
        base_models: Dictionary of {name: model} base learners.
        task: "classification" or "rul".
        n_classes: Number of output classes (classification only).
        meta_alpha: Ridge regularization parameter range for CV.
    """

    def __init__(
        self,
        base_models: Dict[str, nn.Module],
        task: str = "rul",
        n_classes: int = 5,
        meta_alpha: list = None,
    ):
        super().__init__()
        self.base_models = nn.ModuleDict(base_models)
        self.task = task
        self.n_classes = n_classes
        self.meta_alpha = meta_alpha or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.meta_learner = None  # Fitted after base model training
        self._is_meta_fitted = False

    def fit_meta_learner(
        self,
        base_predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """Fit the Ridge meta-learner on base model predictions.

        Args:
            base_predictions: Array of shape (N, n_models) for regression
                             or (N, n_models * n_classes) for classification.
            targets: True target values of shape (N,).
        """
        self.meta_learner = RidgeCV(
            alphas=self.meta_alpha,
            cv=5,
            scoring="neg_mean_squared_error",
        )
        self.meta_learner.fit(base_predictions, targets)
        self._is_meta_fitted = True

    def get_base_predictions(
        self, x: torch.Tensor
    ) -> np.ndarray:
        """Get predictions from all base models.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Array of shape (B, n_models) for regression.
        """
        predictions = []
        for name, model in self.base_models.items():
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if self.task == "classification":
                    pred = torch.softmax(pred, dim=-1)
                predictions.append(pred.cpu().numpy())

        if self.task == "rul":
            return np.column_stack(predictions)
        else:
            return np.concatenate(predictions, axis=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        If meta-learner is fitted, uses stacking.
        Otherwise, averages base model predictions.

        Args:
            x: Input tensor of shape (B, 16, T).

        Returns:
            Ensemble prediction tensor.
        """
        base_preds = self.get_base_predictions(x)

        if self._is_meta_fitted and self.meta_learner is not None:
            meta_pred = self.meta_learner.predict(base_preds)
            return torch.FloatTensor(meta_pred)
        else:
            # Simple averaging fallback
            if self.task == "rul":
                return torch.FloatTensor(base_preds.mean(axis=1))
            else:
                # Average class probabilities
                n_models = len(self.base_models)
                avg_probs = np.zeros((base_preds.shape[0], self.n_classes))
                for i in range(n_models):
                    start = i * self.n_classes
                    avg_probs += base_preds[:, start:start + self.n_classes]
                avg_probs /= n_models
                return torch.FloatTensor(avg_probs)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensemble uncertainty from base model disagreement + MC Dropout.

        Args:
            x: Input tensor.
            n_samples: MC Dropout samples per base model.

        Returns:
            Tuple of (mean, std, all_predictions).
        """
        all_mc_preds = []

        for name, model in self.base_models.items():
            if hasattr(model, "predict_with_uncertainty"):
                _, _, mc_preds = model.predict_with_uncertainty(x, n_samples)
                all_mc_preds.append(mc_preds)  # (n_samples, B)

        if all_mc_preds:
            combined = np.concatenate(all_mc_preds, axis=0)  # (n_total, B)
            mean = combined.mean(axis=0)
            std = combined.std(axis=0)
            return mean, std, combined

        # Fallback: use base prediction variance
        base_preds = self.get_base_predictions(x)
        mean = base_preds.mean(axis=1)
        std = base_preds.std(axis=1)
        return mean, std, base_preds.T

    @property
    def num_parameters(self) -> int:
        """Total parameters across all base models."""
        return sum(
            sum(p.numel() for p in m.parameters() if p.requires_grad)
            for m in self.base_models.values()
        )
