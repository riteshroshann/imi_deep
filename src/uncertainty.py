"""
uncertainty.py — Uncertainty Quantification for RUL Prediction
===============================================================
- Monte Carlo Dropout ensemble for epistemic uncertainty
- Conformal Prediction for distribution-free coverage guarantees
- Calibration curves and reliability diagrams
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO DROPOUT
# ═══════════════════════════════════════════════════════════════════════════════


def mc_dropout_predict(
    model,
    x,
    n_samples: int = 100,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Perform Monte Carlo Dropout inference for epistemic uncertainty.

    Runs n_samples stochastic forward passes with dropout enabled,
    then computes prediction mean and variance.

    Epistemic uncertainty (model uncertainty):
        σ²_epistemic = (1/T) Σ_t (ŷ_t - ȳ)²

    Args:
        model: PyTorch model with dropout layers.
        x: Input tensor or numpy array.
        n_samples: Number of stochastic forward passes.
        device: Compute device.

    Returns:
        Dictionary with mean, std, lower, upper, all_predictions.
    """
    import torch

    model.eval()
    # Enable dropout during inference
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            if pred.dim() > 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)  # (n_samples, N)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)

    return {
        "mean": mean,
        "std": std,
        "lower_95": mean - 1.96 * std,
        "upper_95": mean + 1.96 * std,
        "lower_90": mean - 1.645 * std,
        "upper_90": mean + 1.645 * std,
        "all_predictions": predictions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════


class ConformalPredictor:
    """Split Conformal Prediction for distribution-free coverage.

    Provides prediction intervals with finite-sample coverage guarantees:
        P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α

    Theorem (Vovk et al., 2005):
        For exchangeable data, split conformal prediction achieves
        coverage ≥ 1 - α with probability 1, regardless of the
        underlying distribution.

    Args:
        model: Fitted prediction model.
        alpha: Miscoverage rate (default 0.1 for 90% coverage).
    """

    def __init__(self, model=None, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None

    def calibrate(
        self,
        cal_predictions: np.ndarray,
        cal_targets: np.ndarray,
    ):
        """Calibrate conformal predictor on calibration set.

        Computes nonconformity scores (absolute residuals) and
        determines the (1-α)(1 + 1/n)-th quantile.

        Args:
            cal_predictions: Model predictions on calibration set.
            cal_targets: True values for calibration set.
        """
        # Nonconformity scores: |ŷ - y|
        self.calibration_scores = np.abs(cal_predictions - cal_targets)

        # Adjusted quantile for finite-sample guarantee
        n = len(self.calibration_scores)
        adjusted_alpha = np.ceil((n + 1) * (1 - self.alpha)) / n
        adjusted_alpha = min(adjusted_alpha, 1.0)

        self.quantile = np.quantile(self.calibration_scores, adjusted_alpha)
        logger.info("Conformal calibration: n=%d, α=%.3f, quantile=%.4f",
                    n, self.alpha, self.quantile)

    def predict_interval(
        self,
        predictions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute prediction intervals.

        Args:
            predictions: Point predictions of shape (N,).

        Returns:
            Dictionary with lower, upper bounds and width.
        """
        if self.quantile is None:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        lower = predictions - self.quantile
        upper = predictions + self.quantile

        return {
            "lower": lower,
            "upper": upper,
            "center": predictions,
            "width": np.full_like(predictions, 2 * self.quantile),
        }

    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate empirical coverage and interval metrics.

        Args:
            predictions: Point predictions.
            targets: True values.

        Returns:
            Dictionary of coverage metrics.
        """
        intervals = self.predict_interval(predictions)
        covered = (targets >= intervals["lower"]) & (targets <= intervals["upper"])

        return {
            "empirical_coverage": covered.mean(),
            "target_coverage": 1 - self.alpha,
            "mean_width": intervals["width"].mean(),
            "median_width": np.median(intervals["width"]),
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """Locally-adaptive conformal prediction with heteroscedastic widths.

    Uses a separate model (e.g., uncertainty from MC Dropout) to
    modulate interval widths based on local difficulty.

    Args:
        alpha: Miscoverage rate.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__(alpha=alpha)
        self.scale_scores = None

    def calibrate_adaptive(
        self,
        cal_predictions: np.ndarray,
        cal_targets: np.ndarray,
        cal_uncertainties: np.ndarray,
    ):
        """Calibrate with uncertainty-scaled nonconformity scores.

        Args:
            cal_predictions: Predictions on calibration set.
            cal_targets: True values.
            cal_uncertainties: Uncertainty estimates (e.g., MC Dropout std).
        """
        # Normalized residuals
        residuals = np.abs(cal_predictions - cal_targets)
        scales = cal_uncertainties + 1e-8
        self.scale_scores = residuals / scales

        # Quantile computation
        n = len(self.scale_scores)
        adjusted_alpha = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(self.scale_scores, adjusted_alpha)

    def predict_interval_adaptive(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute adaptive prediction intervals.

        Args:
            predictions: Point predictions.
            uncertainties: Local uncertainty estimates.

        Returns:
            Dictionary with lower, upper bounds and widths.
        """
        if self.quantile is None:
            raise RuntimeError("Must call calibrate_adaptive() first")

        widths = self.quantile * (uncertainties + 1e-8)
        return {
            "lower": predictions - widths,
            "upper": predictions + widths,
            "center": predictions,
            "width": 2 * widths,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_calibration_curve(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    confidence_levels: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Compute calibration/reliability curve data.

    For each confidence level, computes expected vs observed coverage:
        Expected coverage = confidence level
        Observed coverage = fraction of targets within predicted interval

    Args:
        predictions: Point predictions of shape (N,).
        uncertainties: Standard deviation estimates of shape (N,).
        targets: True values of shape (N,).
        confidence_levels: Array of confidence levels to evaluate.

    Returns:
        Dictionary with expected_coverage, observed_coverage, and ECE.
    """
    if confidence_levels is None:
        confidence_levels = np.array([0.50, 0.60, 0.70, 0.75, 0.80,
                                       0.85, 0.90, 0.95, 0.97, 0.99])

    observed = []
    for conf in confidence_levels:
        z = norm.ppf(0.5 + conf / 2)
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        covered = ((targets >= lower) & (targets <= upper)).mean()
        observed.append(covered)

    observed = np.array(observed)

    # Expected Calibration Error
    ece = np.mean(np.abs(observed - confidence_levels))

    return {
        "expected_coverage": confidence_levels,
        "observed_coverage": observed,
        "ece": ece,
    }


def compute_sharpness(uncertainties: np.ndarray) -> Dict[str, float]:
    """Compute sharpness metrics for uncertainty estimates.

    Sharpness measures how tight the prediction intervals are.
    Lower is better (conditional on good coverage).

    Args:
        uncertainties: Standard deviation estimates.

    Returns:
        Dictionary of sharpness metrics.
    """
    return {
        "mean_std": uncertainties.mean(),
        "median_std": np.median(uncertainties),
        "std_of_std": uncertainties.std(),
        "cv_std": uncertainties.std() / (uncertainties.mean() + 1e-8),
    }


def compute_picp_mpiw(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP)
    and Mean Prediction Interval Width (MPIW).

    Args:
        lower: Lower bounds of prediction intervals.
        upper: Upper bounds of prediction intervals.
        targets: True values.

    Returns:
        Dictionary with PICP and MPIW metrics.
    """
    covered = (targets >= lower) & (targets <= upper)
    widths = upper - lower

    return {
        "picp": covered.mean(),
        "mpiw": widths.mean(),
        "mpiw_covered": widths[covered].mean() if covered.any() else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NASA PHM SCORE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def nasa_phm_score(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute the NASA PHM Challenge scoring function.

    Asymmetric scoring that penalizes late predictions (optimistic RUL)
    more heavily than early predictions (conservative RUL).

    Score = Σ_i s_i, where:
        s_i = exp(-d_i/13) - 1,  if d_i < 0  (early prediction)
        s_i = exp(d_i/10) - 1,   if d_i ≥ 0  (late prediction)

    d_i = pred_i - true_i

    Lower score is better.

    Args:
        predictions: Predicted RUL values.
        targets: True RUL values.

    Returns:
        NASA PHM score (lower is better).
    """
    d = predictions - targets
    score = np.where(
        d < 0,
        np.exp(-d / 13) - 1,
        np.exp(d / 10) - 1,
    )
    return score.sum()


def comprehensive_rul_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive RUL prediction metrics.

    Args:
        predictions: Predicted RUL values.
        targets: True RUL values.
        uncertainties: Optional uncertainty estimates.

    Returns:
        Dictionary of all RUL metrics.
    """
    residuals = predictions - targets

    metrics = {
        "rmse": np.sqrt(np.mean(residuals ** 2)),
        "mae": np.mean(np.abs(residuals)),
        "mape": np.mean(np.abs(residuals) / (np.abs(targets) + 1e-8)) * 100,
        "r2": 1 - np.sum(residuals ** 2) / (np.sum((targets - targets.mean()) ** 2) + 1e-8),
        "nasa_score": nasa_phm_score(predictions, targets),
        "mean_bias": np.mean(residuals),
        "max_error": np.max(np.abs(residuals)),
    }

    if uncertainties is not None:
        cal = compute_calibration_curve(predictions, uncertainties, targets)
        metrics["ece"] = cal["ece"]
        sharpness = compute_sharpness(uncertainties)
        metrics.update(sharpness)

    return metrics
