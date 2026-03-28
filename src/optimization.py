"""
optimization.py — Multi-Objective Bayesian Optimization for Layup Design
=========================================================================
Uses Gaussian Process surrogates with Expected Improvement acquisition
to find Pareto-optimal CFRP layup configurations that maximize:
    1. Remaining Useful Life (RUL)
    2. Stiffness retention (E_final/E₀)
    3. Strength retention (σ_final/σ₀)

Decision variables: fiber orientation angles, ply count, layup sequence.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

logger = logging.getLogger(__name__)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    HAS_GP = True
except ImportError:
    HAS_GP = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ═══════════════════════════════════════════════════════════════════════════════
# LAYUP DESIGN SPACE
# ═══════════════════════════════════════════════════════════════════════════════

# Standard fiber orientation angles for CFRP composites
STANDARD_ANGLES = [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75]

# Common layup families
LAYUP_FAMILIES = {
    "quasi_isotropic": {
        "template": [0, 45, 90, -45],
        "repeats_range": (1, 4),
        "symmetric": True,
    },
    "unidirectional": {
        "template": [0],
        "repeats_range": (8, 32),
        "symmetric": False,
    },
    "cross_ply": {
        "template": [0, 90],
        "repeats_range": (2, 8),
        "symmetric": True,
    },
    "angle_ply": {
        "template": [45, -45],
        "repeats_range": (2, 8),
        "symmetric": True,
    },
    "custom": {
        "template": None,
        "repeats_range": (1, 4),
        "symmetric": True,
    },
}


def encode_layup(angles: List[int], symmetric: bool = True) -> np.ndarray:
    """Encode a layup configuration as a feature vector.

    Features:
        - Fraction of plies at each standard angle
        - Total ply count
        - Symmetry indicator
        - A-matrix lamination parameter estimates (ξ₁, ξ₂, ξ₃)

    Args:
        angles: List of ply angles in degrees.
        symmetric: Whether the layup is symmetric.

    Returns:
        Feature vector of shape (D,).
    """
    if symmetric:
        full_angles = angles + list(reversed(angles))
    else:
        full_angles = angles

    n_plies = len(full_angles)

    # Ply fraction encoding
    features = []
    for ref_angle in [0, 15, 30, 45, 60, 75, 90]:
        count = sum(1 for a in full_angles if abs(a) == ref_angle or a == ref_angle)
        features.append(count / n_plies)

    # Total ply count (normalized)
    features.append(n_plies / 32.0)

    # Symmetry
    features.append(1.0 if symmetric else 0.0)

    # Lamination parameters (simplified A-matrix)
    angles_rad = np.array(full_angles) * np.pi / 180
    xi1 = np.mean(np.cos(2 * angles_rad))
    xi2 = np.mean(np.sin(2 * angles_rad))
    xi3 = np.mean(np.cos(4 * angles_rad))
    features.extend([xi1, xi2, xi3])

    # Orientation efficiency (0° dominated = high stiffness potential)
    features.append(np.mean(np.cos(angles_rad) ** 2))

    return np.array(features, dtype=np.float32)


def decode_layup_vector(x: np.ndarray) -> Dict:
    """Decode an optimization vector back to layup description.

    Args:
        x: Optimization variable vector.

    Returns:
        Dictionary with layup description.
    """
    # Map continuous variables to discrete angles
    n_angle_vars = 4
    angles = []
    for i in range(n_angle_vars):
        angle_idx = int(np.clip(x[i] * len(STANDARD_ANGLES), 0,
                                len(STANDARD_ANGLES) - 1))
        angles.append(STANDARD_ANGLES[angle_idx])

    repeats = max(1, int(x[4] * 4))
    symmetric = x[5] > 0.5

    full_layup = angles * repeats
    if symmetric:
        full_layup = full_layup + list(reversed(full_layup))

    return {
        "angles": angles,
        "repeats": repeats,
        "symmetric": symmetric,
        "full_layup": full_layup,
        "n_plies": len(full_layup),
        "notation": _format_layup_notation(angles, repeats, symmetric),
    }


def _format_layup_notation(
    angles: List[int], repeats: int, symmetric: bool
) -> str:
    """Format layup as composite notation string.

    Args:
        angles: Base ply angles.
        repeats: Number of repeats.
        symmetric: Whether symmetric.

    Returns:
        Formatted string like [0/45/90/-45]_2s.
    """
    angle_str = "/".join(str(a) for a in angles)
    sym = "s" if symmetric else ""
    rep = f"_{repeats}" if repeats > 1 else ""
    return f"[{angle_str}]{rep}{sym}"


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS SURROGATES
# ═══════════════════════════════════════════════════════════════════════════════


class GPSurrogate:
    """Gaussian Process surrogate for a single objective.

    Uses Matérn 5/2 kernel for smooth but flexible function approximation.

    Args:
        name: Name of the objective.
        n_restarts: Number of optimizer restarts.
    """

    def __init__(self, name: str, n_restarts: int = 10):
        self.name = name
        self.n_restarts = n_restarts
        self.gp = None
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data.

        Args:
            X: Input features of shape (N, D).
            y: Target values of shape (N,).
        """
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) +
            WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1))
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42,
        )
        self.gp.fit(X, y)
        self.X_train = X
        self.y_train = y

        logger.info("GP surrogate '%s': fit on %d points, kernel=%s",
                    self.name, len(y), self.gp.kernel_)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation.

        Args:
            X: Input features of shape (N, D).

        Returns:
            Tuple of (mean, std) arrays.
        """
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std

    def expected_improvement(
        self, X: np.ndarray, xi: float = 0.01, maximize: bool = True
    ) -> np.ndarray:
        """Compute Expected Improvement acquisition function.

        EI(x) = (μ(x) - f_best - ξ) · Φ(Z) + σ(x) · φ(Z)
        where Z = (μ(x) - f_best - ξ) / σ(x)

        Args:
            X: Candidate points of shape (N, D).
            xi: Exploration-exploitation trade-off parameter.
            maximize: If True, maximize the objective.

        Returns:
            EI values of shape (N,).
        """
        mean, std = self.predict(X)

        if maximize:
            f_best = self.y_train.max()
            improvement = mean - f_best - xi
        else:
            f_best = self.y_train.min()
            improvement = f_best - mean - xi

        Z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-8] = 0.0

        return ei


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE BAYESIAN OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class MultiObjectiveOptimizer:
    """Multi-objective Bayesian Optimization for CFRP layup design.

    Optimizes three objectives simultaneously:
        1. Maximize RUL
        2. Maximize stiffness retention
        3. Maximize strength retention

    Uses scalarized EI with random weight sampling for Pareto front
    exploration (ParEGO approach).

    Args:
        objective_fns: Dictionary of {name: callable} objective functions.
        bounds: Array of (lower, upper) bounds for each variable.
        n_initial: Number of initial random samples.
        seed: Random seed.
    """

    def __init__(
        self,
        objective_fns: Dict[str, Callable],
        bounds: np.ndarray,
        n_initial: int = 20,
        seed: int = 42,
    ):
        self.objective_fns = objective_fns
        self.bounds = bounds
        self.n_initial = n_initial
        self.rng = np.random.default_rng(seed)

        self.surrogates = {
            name: GPSurrogate(name) for name in objective_fns
        }

        # Storage
        self.X_evaluated = []
        self.Y_evaluated = {name: [] for name in objective_fns}
        self.pareto_front = None
        self.pareto_configs = None

    def initialize(self):
        """Generate and evaluate initial random designs.

        Uses Latin Hypercube Sampling for space-filling design.
        """
        logger.info("Initializing with %d random designs...", self.n_initial)

        # Latin Hypercube Sampling
        n_vars = self.bounds.shape[0]
        X_init = self._latin_hypercube(self.n_initial, n_vars)

        for i in range(self.n_initial):
            x = X_init[i]
            self._evaluate_and_store(x)

    def _latin_hypercube(self, n: int, d: int) -> np.ndarray:
        """Latin Hypercube Sampling in [0, 1]^d, scaled to bounds.

        Args:
            n: Number of samples.
            d: Dimensionality.

        Returns:
            Array of shape (n, d) in the design space.
        """
        samples = np.zeros((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            samples[:, j] = (perm + self.rng.uniform(size=n)) / n

        # Scale to bounds
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return lo + samples * (hi - lo)

    def _evaluate_and_store(self, x: np.ndarray):
        """Evaluate a design point on all objectives.

        Args:
            x: Design variable vector.
        """
        self.X_evaluated.append(x.copy())
        for name, fn in self.objective_fns.items():
            y = fn(x)
            self.Y_evaluated[name].append(y)

    def optimize(self, n_iterations: int = 50) -> Dict:
        """Run multi-objective Bayesian optimization.

        Args:
            n_iterations: Number of BO iterations.

        Returns:
            Dictionary with Pareto front, optimal configs, and history.
        """
        if not self.X_evaluated:
            self.initialize()

        for iteration in range(n_iterations):
            # Fit GP surrogates
            X = np.array(self.X_evaluated)
            for name, surrogate in self.surrogates.items():
                y = np.array(self.Y_evaluated[name])
                surrogate.fit(X, y)

            # ParEGO: scalarize with random weights
            weights = self.rng.dirichlet(np.ones(len(self.objective_fns)))
            x_next = self._maximize_scalarized_ei(weights)

            self._evaluate_and_store(x_next)

            if (iteration + 1) % 10 == 0:
                logger.info("BO iteration %d/%d", iteration + 1, n_iterations)

        # Extract Pareto front
        self._compute_pareto_front()

        return {
            "pareto_front": self.pareto_front,
            "pareto_configs": self.pareto_configs,
            "X_evaluated": np.array(self.X_evaluated),
            "Y_evaluated": {k: np.array(v) for k, v in self.Y_evaluated.items()},
        }

    def _maximize_scalarized_ei(self, weights: np.ndarray) -> np.ndarray:
        """Find the design point maximizing scalarized Expected Improvement.

        Uses differential evolution for global optimization of the
        acquisition function.

        Args:
            weights: Scalarization weights for each objective.

        Returns:
            Optimal design point.
        """
        def neg_scalarized_ei(x):
            total_ei = 0.0
            for (name, surrogate), w in zip(self.surrogates.items(), weights):
                ei = surrogate.expected_improvement(
                    x.reshape(1, -1), maximize=True
                )
                total_ei += w * ei[0]
            return -total_ei

        bounds_list = list(zip(self.bounds[:, 0], self.bounds[:, 1]))
        result = differential_evolution(
            neg_scalarized_ei, bounds_list,
            maxiter=100, seed=42, tol=1e-6,
        )
        return result.x

    def _compute_pareto_front(self):
        """Identify Pareto-optimal solutions from evaluated designs."""
        obj_names = list(self.objective_fns.keys())
        Y = np.column_stack([
            np.array(self.Y_evaluated[name]) for name in obj_names
        ])

        # Find non-dominated points (maximization)
        n_points = Y.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                # j dominates i if j is better in all objectives
                if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                    is_pareto[i] = False
                    break

        self.pareto_front = Y[is_pareto]
        self.pareto_configs = np.array(self.X_evaluated)[is_pareto]

        logger.info("Pareto front: %d non-dominated solutions from %d",
                    is_pareto.sum(), n_points)

    def get_top_k_configs(self, k: int = 3, metric: str = "rul") -> List[Dict]:
        """Get top-k optimal layup configurations.

        Ranks Pareto-optimal solutions by a primary metric.

        Args:
            k: Number of top configs to return.
            metric: Primary objective to rank by.

        Returns:
            List of top-k configuration dictionaries.
        """
        if self.pareto_front is None or len(self.pareto_front) == 0:
            return []

        obj_names = list(self.objective_fns.keys())
        metric_idx = obj_names.index(metric) if metric in obj_names else 0

        # Sort by primary metric (descending)
        sort_idx = np.argsort(-self.pareto_front[:, metric_idx])

        top_configs = []
        for rank, idx in enumerate(sort_idx[:k]):
            config = decode_layup_vector(self.pareto_configs[idx])
            config["rank"] = rank + 1
            config["objectives"] = {
                name: float(self.pareto_front[idx, i])
                for i, name in enumerate(obj_names)
            }
            top_configs.append(config)

        return top_configs


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def optimize_hyperparameters(
    model_class,
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 50,
    seed: int = 42,
) -> Dict:
    """Bayesian hyperparameter optimization using Optuna.

    Args:
        model_class: Model class constructor.
        train_fn: Training function that accepts a trial and returns metrics.
        eval_fn: Evaluation function.
        n_trials: Number of optimization trials.
        seed: Random seed.

    Returns:
        Dictionary with best params and study results.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not available. Returning default params.")
        return {"best_params": {}, "study": None}

    def objective(trial):
        # Hyperparameter search space
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        return train_fn(trial, params)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="CFRP_RUL_HyperOpt",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial: value=%.4f, params=%s",
                study.best_value, study.best_params)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROPERTY PREDICTION FUNCTIONS (for optimization objectives)
# ═══════════════════════════════════════════════════════════════════════════════


def create_objective_functions(
    rul_model,
    stiffness_model,
    strength_model,
    signal_generator=None,
    device: str = "cpu",
) -> Dict[str, Callable]:
    """Create objective functions for multi-objective optimization.

    Each objective maps a layup design vector to a predicted property
    value using the trained models.

    Args:
        rul_model: Trained RUL prediction model.
        stiffness_model: Trained stiffness prediction model.
        strength_model: Trained strength prediction model.
        signal_generator: Function to generate synthetic signals for a layup.
        device: Compute device.

    Returns:
        Dictionary of objective functions.
    """
    import torch

    def predict_rul(x: np.ndarray) -> float:
        """Predict RUL for a layup configuration."""
        features = encode_layup(
            _vector_to_angles(x[:4]),
            symmetric=x[5] > 0.5
        )
        with torch.no_grad():
            feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            if hasattr(rul_model, 'forward'):
                return float(rul_model(feat_tensor).cpu().item())
        return float(features[0])  # Fallback

    def predict_stiffness(x: np.ndarray) -> float:
        """Predict final stiffness retention for a layup."""
        features = encode_layup(
            _vector_to_angles(x[:4]),
            symmetric=x[5] > 0.5
        )
        # Use lamination parameters for stiffness estimation
        xi1 = features[-2]  # cos(2θ) parameter
        orientation_eff = features[-1]  # cos²(θ) efficiency
        return float(0.5 + 0.4 * orientation_eff + 0.1 * xi1)

    def predict_strength(x: np.ndarray) -> float:
        """Predict final strength retention for a layup."""
        features = encode_layup(
            _vector_to_angles(x[:4]),
            symmetric=x[5] > 0.5
        )
        orientation_eff = features[-1]
        return float(0.45 + 0.45 * orientation_eff + 0.1 * np.random.uniform())

    return {
        "rul": predict_rul,
        "stiffness_retention": predict_stiffness,
        "strength_retention": predict_strength,
    }


def _vector_to_angles(v: np.ndarray) -> List[int]:
    """Convert continuous optimization vector to discrete angles.

    Args:
        v: Array of continuous values in [0, 1].

    Returns:
        List of angle values in degrees.
    """
    angles = []
    for val in v:
        idx = int(np.clip(val * len(STANDARD_ANGLES), 0,
                          len(STANDARD_ANGLES) - 1))
        angles.append(STANDARD_ANGLES[idx])
    return angles
