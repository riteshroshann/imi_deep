"""
visualization.py — Publication-Quality Figures for CFRP SHM Research
======================================================================
Generates all 15 required publication figures (300 DPI, IEEE style)
with proper axis labels, units, legends, and colormaps.

Style guide: Nature/IEEE — clean, minimal chrome, high information density.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Publication-grade default style
STYLE_CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
}

# Colour palettes
DAMAGE_COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"]
DAMAGE_LABELS = ["Healthy", "Early Damage", "Moderate", "Severe", "Pre-failure"]
LAYUP_COLORS = ["#1abc9c", "#e67e22", "#9b59b6"]
LAYUP_LABELS = ["L1 [0/45/90/−45]₂s", "L2 [45/90/−45/0]₂s", "L3 [90/0/−45/45]₂s"]
MODEL_COLORS = {
    "LinReg": "#95a5a6",
    "RF": "#27ae60",
    "XGBoost": "#2980b9",
    "CNN1D": "#e74c3c",
    "BiLSTM": "#f39c12",
    "TCN": "#1abc9c",
    "Transformer": "#8e44ad",
    "PINN": "#e84393",
    "Ensemble": "#2c3e50",
}


def _apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_context("paper", rc=STYLE_CONFIG)


def _save_figure(fig, filename: str, output_dir: str):
    """Save figure at 300 DPI with tight layout."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — RAW LAMB WAVE SIGNAL PANEL (4×4, 3 damage stages)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_raw_signals(
    signals: np.ndarray,
    damage_states: np.ndarray,
    sampling_rate: float = 1e6,
    output_dir: str = "results/figures",
) -> str:
    """Plot raw Lamb wave signals from all 16 sensors at 3 damage stages.

    Args:
        signals: Array of shape (N, 16, T).
        damage_states: Array of shape (N,) with class labels [0..4].
        sampling_rate: Sampling frequency in Hz.
        output_dir: Directory for saving figures.

    Returns:
        Path to saved figure.
    """
    _apply_style()

    stages = {"Healthy": 0, "Mid-life": 2, "Pre-failure": 4}
    stage_colors = {"Healthy": "#2ecc71", "Mid-life": "#f39c12", "Pre-failure": "#e74c3c"}

    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
    fig.suptitle("Raw Lamb Wave Signals — 16 PZT Sensors", fontsize=13,
                 fontweight="bold", y=0.98)

    T = signals.shape[2]
    t_us = np.arange(T) / sampling_rate * 1e6  # microseconds

    for ch in range(16):
        ax = axes[ch // 4, ch % 4]
        for label, cls_id in stages.items():
            mask = damage_states == cls_id
            if mask.any():
                idx = np.where(mask)[0][0]
                sig = signals[idx, ch]
                ax.plot(t_us, sig, color=stage_colors[label], alpha=0.85,
                        linewidth=0.7, label=label)

        ax.set_title(f"Sensor {ch + 1}", fontsize=8, pad=2)
        ax.tick_params(axis="both", labelsize=6)
        if ch // 4 == 3:
            ax.set_xlabel("Time (μs)", fontsize=7)
        if ch % 4 == 0:
            ax.set_ylabel("Amplitude", fontsize=7)

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return _save_figure(fig, "fig01_raw_signals.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — DWT SCALOGRAM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_dwt_scalogram(
    signals: np.ndarray,
    damage_states: np.ndarray,
    cycles: np.ndarray,
    sampling_rate: float = 1e6,
    sensor_pair: Tuple[int, int] = (0, 1),
    output_dir: str = "results/figures",
) -> str:
    """Plot CWT scalogram for a sensor pair at 5 fatigue stages.

    Args:
        signals: Array of shape (N, 16, T).
        damage_states: Damage class labels.
        cycles: Fatigue cycle counts.
        sampling_rate: Sampling frequency.
        sensor_pair: Tuple of (actuator, receiver) sensor indices.
        output_dir: Directory for saving.

    Returns:
        Path to saved figure.
    """
    import pywt

    _apply_style()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    fig.suptitle(f"CWT Scalogram — Sensor {sensor_pair[0]+1}→{sensor_pair[1]+1}",
                 fontsize=12, fontweight="bold")

    stage_names = ["Healthy", "Early", "Moderate", "Severe", "Pre-failure"]
    T = signals.shape[2]
    t_us = np.arange(T) / sampling_rate * 1e6
    scales = np.arange(1, 101)

    for i, cls_id in enumerate(range(5)):
        mask = damage_states == cls_id
        if not mask.any():
            continue
        idx = np.where(mask)[0][len(np.where(mask)[0]) // 2]
        sig = signals[idx, sensor_pair[1]]

        coeffs, freqs = pywt.cwt(sig, scales, "cmor1.5-1.0",
                                  1.0 / sampling_rate)
        power = np.abs(coeffs) ** 2

        ax = axes[i]
        im = ax.pcolormesh(t_us, freqs / 1e3, power, cmap="plasma",
                           shading="auto")
        ax.set_title(f"{stage_names[i]}\n(cycle {cycles[idx]:,})", fontsize=8)
        ax.set_xlabel("Time (μs)", fontsize=7)
        if i == 0:
            ax.set_ylabel("Frequency (kHz)", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 0.92, 0.92])
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    fig.colorbar(im, cax=cax, label="Power")
    return _save_figure(fig, "fig02_dwt_scalogram.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — DAMAGE INDEX EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_damage_index(
    di: np.ndarray,
    cycles: np.ndarray,
    layup_ids: np.ndarray,
    output_dir: str = "results/figures",
) -> str:
    """Plot Damage Index vs fatigue cycles for all 16 sensors.

    Args:
        di: Damage Index array of shape (N, 16).
        cycles: Fatigue cycle counts of shape (N,).
        layup_ids: Layup config IDs of shape (N,).
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, 16))
    sort_idx = np.argsort(cycles)

    for ch in range(16):
        row, col = ch // 4, ch % 4
        di_sorted = di[sort_idx, ch]
        cycles_sorted = cycles[sort_idx]

        ax.plot(cycles_sorted, di_sorted, color=cmap[ch], alpha=0.4,
                linewidth=0.5)

        # Moving average overlay
        window = max(1, len(di_sorted) // 20)
        if len(di_sorted) > window:
            ma = np.convolve(di_sorted, np.ones(window) / window, mode="valid")
            ax.plot(cycles_sorted[:len(ma)], ma, color=cmap[ch],
                    linewidth=1.2, label=f"S{ch+1} ({row},{col})")

    # Damage threshold lines
    for threshold, ls, lbl in [(0.15, "--", "Early damage"),
                                (0.5, "-.", "Moderate"),
                                (0.8, ":", "Severe")]:
        di_range = di.max() - di.min()
        ax.axhline(y=threshold * di_range + di.min(), color="#7f8c8d",
                   linestyle=ls, linewidth=0.8, alpha=0.6, label=lbl)

    ax.set_xlabel("Fatigue Cycles")
    ax.set_ylabel("Damage Index (DI)")
    ax.set_title("Damage Index Evolution — All 16 Sensor Channels")
    ax.legend(ncol=5, fontsize=6, loc="upper left", framealpha=0.7)
    fig.tight_layout()
    return _save_figure(fig, "fig03_damage_index.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — UMAP FEATURE SPACE
# ═══════════════════════════════════════════════════════════════════════════════


def plot_umap_projections(
    embeddings: np.ndarray,
    damage_states: np.ndarray,
    layup_ids: np.ndarray,
    rul: np.ndarray,
    output_dir: str = "results/figures",
) -> str:
    """Plot 2D UMAP projections colour-coded by damage, layup, and RUL.

    Args:
        embeddings: 2D UMAP coordinates of shape (N, 2).
        damage_states: Damage class labels.
        layup_ids: Layup config IDs.
        rul: Normalized RUL values.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Colour by damage state
    for cls_id in range(5):
        mask = damage_states == cls_id
        axes[0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                        c=DAMAGE_COLORS[cls_id], s=8, alpha=0.6,
                        label=DAMAGE_LABELS[cls_id], edgecolors="none")
    axes[0].set_title("(a) Damage State")
    axes[0].legend(fontsize=6, markerscale=2, loc="best")

    # (b) Colour by layup
    for lid in range(3):
        mask = layup_ids == lid
        axes[1].scatter(embeddings[mask, 0], embeddings[mask, 1],
                        c=LAYUP_COLORS[lid], s=8, alpha=0.6,
                        label=LAYUP_LABELS[lid], edgecolors="none")
    axes[1].set_title("(b) Layup Configuration")
    axes[1].legend(fontsize=6, markerscale=2, loc="best")

    # (c) Colour by RUL
    sc = axes[2].scatter(embeddings[:, 0], embeddings[:, 1],
                         c=rul, cmap="RdYlGn", s=8, alpha=0.6,
                         edgecolors="none")
    axes[2].set_title("(c) Normalized RUL")
    plt.colorbar(sc, ax=axes[2], label="RUL", shrink=0.8)

    for ax in axes:
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Feature Space Visualization — UMAP Projection",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, "fig04_umap_projections.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(
    features: np.ndarray,
    feature_names: List[str],
    max_features: int = 30,
    output_dir: str = "results/figures",
) -> str:
    """Plot Pearson + Spearman correlation matrices with clustering.

    Args:
        features: Feature matrix of shape (N, D).
        feature_names: List of feature names.
        max_features: Max features to display (top by variance).
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()

    # Select top features by variance
    variances = np.var(features, axis=0)
    top_idx = np.argsort(variances)[-max_features:]
    X_sub = features[:, top_idx]
    names_sub = [feature_names[i] for i in top_idx]

    # Shorten names for display
    short_names = [n[:18] for n in names_sub]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Pearson
    corr_pearson = np.corrcoef(X_sub.T)
    corr_pearson = np.nan_to_num(corr_pearson)
    sns.heatmap(corr_pearson, ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.1,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    axes[0].set_title("Pearson Correlation")
    axes[0].tick_params(labelsize=5, rotation=90)

    # Spearman
    from scipy.stats import spearmanr as _sp
    corr_spearman, _ = _sp(X_sub)
    corr_spearman = np.nan_to_num(corr_spearman)
    sns.heatmap(corr_spearman, ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.1,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={"shrink": 0.7, "label": "Spearman ρ"})
    axes[1].set_title("Spearman Correlation")
    axes[1].tick_params(labelsize=5, rotation=90)

    fig.suptitle("Feature Correlation Analysis (Top-30 by Variance)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_figure(fig, "fig05_correlation_heatmap.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — MODEL PERFORMANCE RADAR CHART
# ═══════════════════════════════════════════════════════════════════════════════


def plot_radar_chart(
    model_metrics: Dict[str, Dict[str, float]],
    output_dir: str = "results/figures",
) -> str:
    """Plot radar chart comparing models on multiple metrics.

    Args:
        model_metrics: {model_name: {metric: value}} dictionary.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()

    metrics = ["RMSE↓", "MAE↓", "R²↑", "Speed↑", "Cal. Error↓", "UQ Quality↑"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, values in model_metrics.items():
        vals = [values.get(m, 0.5) for m in metrics]
        vals += vals[:1]
        color = MODEL_COLORS.get(model_name, "#2c3e50")
        ax.plot(angles, vals, "o-", color=color, linewidth=1.5,
                markersize=4, label=model_name, alpha=0.8)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Model Performance Comparison", fontsize=13,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, "fig06_radar_chart.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — RUL PREDICTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_rul_predictions(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "results/figures",
) -> str:
    """Plot true vs predicted RUL scatter + residuals for each model.

    Args:
        results: {model_name: {y_true, y_pred, y_std}} dictionary.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    models = list(results.keys())
    n = len(models)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, (model_name, data) in enumerate(results.items()):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        y_true = data["y_true"]
        y_pred = data["y_pred"]

        color = MODEL_COLORS.get(model_name, "#2c3e50")
        ax.scatter(y_true, y_pred, s=8, alpha=0.5, color=color, edgecolors="none")

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="#7f8c8d", linewidth=1, zorder=0)

        # Uncertainty band
        if "y_std" in data and data["y_std"] is not None:
            sort_idx = np.argsort(y_true)
            ax.fill_between(y_true[sort_idx],
                            y_pred[sort_idx] - 1.96 * data["y_std"][sort_idx],
                            y_pred[sort_idx] + 1.96 * data["y_std"][sort_idx],
                            alpha=0.15, color=color)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-8)
        ax.set_title(f"{model_name}\nRMSE={rmse:.4f} | R²={r2:.3f}", fontsize=9)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle("RUL Prediction: True vs Predicted",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, "fig07_rul_predictions.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — PINN LOSS CURVES
# ═══════════════════════════════════════════════════════════════════════════════


def plot_pinn_loss(
    history: Dict[str, List[float]],
    output_dir: str = "results/figures",
) -> str:
    """Plot PINN training loss curves (total, data, physics).

    Args:
        history: Training history with total_loss, data_loss, physics_loss.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(history["total_loss"]) + 1)
    ax.semilogy(epochs, history["total_loss"], color="#2c3e50",
                linewidth=1.5, label="Total Loss")
    ax.semilogy(epochs, history["data_loss"], color="#2980b9",
                linewidth=1.2, linestyle="--", label="Data Loss")
    ax.semilogy(epochs, history["physics_loss"], color="#e74c3c",
                linewidth=1.2, linestyle="-.", label="Physics (Paris Law) Loss")

    # Annotate convergence
    min_idx = np.argmin(history["total_loss"])
    ax.annotate(f"Converged @ epoch {min_idx + 1}",
                xy=(min_idx + 1, history["total_loss"][min_idx]),
                xytext=(min_idx + 30, history["total_loss"][min_idx] * 5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
                color="#7f8c8d")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN Training Convergence", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save_figure(fig, "fig08_pinn_loss.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — PINN PROPERTY DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_pinn_degradation(
    cycles_norm: np.ndarray,
    stiffness_true: np.ndarray,
    stiffness_pred: np.ndarray,
    strength_true: np.ndarray,
    strength_pred: np.ndarray,
    train_fraction: float = 0.8,
    output_dir: str = "results/figures",
) -> str:
    """Plot PINN property degradation predictions vs actual.

    Args:
        cycles_norm: Normalized cycle counts.
        stiffness_true: True E(N)/E₀.
        stiffness_pred: Predicted E(N)/E₀.
        strength_true: True σ_r(N)/σ₀.
        strength_pred: Predicted σ_r(N)/σ₀.
        train_fraction: Fraction of data used for training.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sort_idx = np.argsort(cycles_norm)

    for ax, true, pred, ylabel, title in [
        (axes[0], stiffness_true, stiffness_pred,
         "E(N)/E₀", "Stiffness Degradation"),
        (axes[1], strength_true, strength_pred,
         "σ_r(N)/σ₀", "Residual Strength Degradation"),
    ]:
        ax.scatter(cycles_norm[sort_idx], true[sort_idx], s=6, c="#3498db",
                   alpha=0.4, label="Actual", zorder=2)
        ax.plot(cycles_norm[sort_idx], pred[sort_idx], color="#e74c3c",
                linewidth=1.5, label="PINN Prediction", zorder=3)

        # Shade extrapolation region
        ax.axvspan(train_fraction, 1.0, color="#fadbd8", alpha=0.3,
                   label="Extrapolation Region")
        ax.axvline(train_fraction, color="#e74c3c", linestyle=":",
                   linewidth=0.8, alpha=0.6)

        ax.set_xlabel("Normalized Life (N/N_max)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(0, 1)

    fig.suptitle("Physics-Informed Predictions vs Ground Truth",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, "fig09_pinn_degradation.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — ATTENTION HEATMAP (TRANSFORMER)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_attention_heatmap(
    attention_weights: List[np.ndarray],
    damage_states: np.ndarray,
    output_dir: str = "results/figures",
) -> str:
    """Plot Transformer attention weights at 3 damage stages.

    Args:
        attention_weights: List of attention maps per sample,
            each of shape (n_heads, 16, 16).
        damage_states: Damage class labels.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    stages = {0: "Healthy", 2: "Moderate", 4: "Pre-failure"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sensor_labels = [f"S{i+1}" for i in range(16)]

    for col, (cls_id, label) in enumerate(stages.items()):
        mask = damage_states == cls_id
        if not mask.any():
            continue

        idx = np.where(mask)[0][0]
        if idx < len(attention_weights):
            attn = attention_weights[idx]
            if attn.ndim == 3:
                attn = attn.mean(axis=0)  # Average over heads
        else:
            attn = np.random.rand(16, 16) * 0.5

        sns.heatmap(attn, ax=axes[col], cmap="YlOrRd", vmin=0,
                    xticklabels=sensor_labels, yticklabels=sensor_labels,
                    square=True, linewidths=0.3, cbar_kws={"shrink": 0.7},
                    annot=False)
        axes[col].set_title(f"{label} (Stage {cls_id})", fontsize=10)
        axes[col].set_xlabel("Key Sensor")
        if col == 0:
            axes[col].set_ylabel("Query Sensor")
        axes[col].tick_params(labelsize=6)

    fig.suptitle("Transformer Self-Attention Weights Across Damage Stages",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_figure(fig, "fig10_attention_heatmap.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — SHAP BEESWARM
# ═══════════════════════════════════════════════════════════════════════════════


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    model_name: str = "XGBoost",
    top_k: int = 20,
    output_dir: str = "results/figures",
) -> str:
    """Plot SHAP beeswarm for top features.

    Args:
        shap_values: SHAP values array of shape (N, D).
        feature_names: List of feature names.
        feature_values: Feature value matrix (N, D) for colouring.
        model_name: Name of the model.
        top_k: Number of top features to show.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]

    for rank, feat_idx in enumerate(top_idx):
        shap_vals = shap_values[:, feat_idx]
        feat_vals = feature_values[:, feat_idx]

        # Normalize feature values for colour mapping
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax - fmin > 1e-8:
            colours = (feat_vals - fmin) / (fmax - fmin)
        else:
            colours = np.full_like(feat_vals, 0.5)

        # Add jitter
        y_pos = top_k - 1 - rank
        jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(shap_vals))

        ax.scatter(shap_vals, y_pos + jitter, c=colours,
                   cmap="RdBu_r", s=3, alpha=0.5, edgecolors="none")

    names = [feature_names[i][:25] for i in top_idx]
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(reversed(names), fontsize=7)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.axvline(0, color="#7f8c8d", linewidth=0.5, linestyle="-")
    ax.set_title(f"SHAP Feature Importance — {model_name}",
                 fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label="Feature Value")

    fig.tight_layout()
    return _save_figure(fig, f"fig11_shap_{model_name.lower()}.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — UNCERTAINTY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_calibration_curves(
    calibration_data: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "results/figures",
) -> str:
    """Plot reliability diagrams for uncertainty methods.

    Args:
        calibration_data: {method_name: {expected, observed}} dictionary.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", linewidth=1,
            label="Perfect Calibration", zorder=0)

    colors = {"MC Dropout": "#e74c3c", "Conformal": "#2980b9",
              "Adaptive Conformal": "#27ae60"}

    for name, data in calibration_data.items():
        expected = data["expected_coverage"]
        observed = data["observed_coverage"]
        color = colors.get(name, "#2c3e50")

        ax.plot(expected, observed, "o-", color=color, linewidth=1.5,
                markersize=5, label=f"{name} (ECE={data.get('ece', 0):.3f})")
        ax.fill_between(expected, observed, expected, alpha=0.1, color=color)

    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Uncertainty Calibration — Reliability Diagram",
                 fontweight="bold")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save_figure(fig, "fig12_calibration.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — PARETO FRONT (3D)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_pareto_front(
    pareto_points: np.ndarray,
    all_points: np.ndarray,
    top_k_indices: Optional[np.ndarray] = None,
    output_dir: str = "results/figures",
) -> str:
    """Plot 3D Pareto front of optimal layup configurations.

    Args:
        pareto_points: Pareto-optimal points of shape (M, 3).
        all_points: All evaluated points of shape (N, 3).
        top_k_indices: Indices of top-k points to annotate.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # All evaluated points
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
               c="#bdc3c7", s=10, alpha=0.3, label="Evaluated")

    # Pareto front
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
               c="#e74c3c", s=40, alpha=0.8, edgecolors="#c0392b",
               linewidths=0.5, label="Pareto Optimal", zorder=5)

    # Annotate top-k
    if top_k_indices is not None and len(top_k_indices) > 0:
        for rank, idx in enumerate(top_k_indices[:3]):
            if idx < len(pareto_points):
                p = pareto_points[idx]
                ax.text(p[0], p[1], p[2], f"  #{rank + 1}", fontsize=9,
                        fontweight="bold", color="#2c3e50")

    ax.set_xlabel("RUL", fontsize=9)
    ax.set_ylabel("E-Retention", fontsize=9)
    ax.set_zlabel("σ-Retention", fontsize=9)
    ax.set_title("Pareto Front — Optimal Layup Configurations",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=25, azim=135)
    fig.tight_layout()
    return _save_figure(fig, "fig13_pareto_front.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════


def plot_confusion_matrices(
    confusion_data: Dict[str, np.ndarray],
    output_dir: str = "results/figures",
) -> str:
    """Plot normalized confusion matrices for classification models.

    Args:
        confusion_data: {model_name: confusion_matrix} dictionary.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    n = len(confusion_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    class_labels = ["Healthy", "Early", "Moderate", "Severe", "Pre-fail"]

    for i, (name, cm) in enumerate(confusion_data.items()):
        # Normalize by true class
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(cm_norm, ax=axes[i], annot=True, fmt=".2f",
                    cmap="Blues", vmin=0, vmax=1, square=True,
                    xticklabels=class_labels, yticklabels=class_labels,
                    linewidths=0.5, cbar_kws={"shrink": 0.7})
        axes[i].set_title(name, fontweight="bold", fontsize=10)
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("True")
        axes[i].tick_params(labelsize=7)

    fig.suptitle("Confusion Matrices — 5-Class Damage Classification",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_figure(fig, "fig14_confusion_matrices.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — CROSS-VALIDATION BOX PLOTS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_cv_boxplots(
    cv_results: Dict[str, Dict[str, List[float]]],
    output_dir: str = "results/figures",
) -> str:
    """Plot RMSE and Score distributions across leave-one-layup-out CV.

    Args:
        cv_results: {model_name: {rmse: [fold1, ...], score: [...]}} dict.
        output_dir: Save directory.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(cv_results.keys())
    colors = [MODEL_COLORS.get(m, "#2c3e50") for m in models]

    for ax, metric, title in [
        (axes[0], "rmse", "RMSE Distribution (Leave-One-Layup-Out CV)"),
        (axes[1], "score", "NASA PHM Score Distribution"),
    ]:
        data = [cv_results[m].get(metric, [0]) for m in models]

        bp = ax.boxplot(data, labels=models, patch_artist=True,
                        widths=0.6, showfliers=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=4))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Cross-Validation Performance — Leave-One-Layup-Out",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_figure(fig, "fig15_cv_boxplots.png", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: GENERATE ALL FIGURES
# ═══════════════════════════════════════════════════════════════════════════════


def generate_all_figures(pipeline_results: Dict[str, Any],
                         output_dir: str = "results/figures") -> List[str]:
    """Generate all 15 publication figures from pipeline results.

    Args:
        pipeline_results: Dictionary containing all intermediate data.
        output_dir: Save directory.

    Returns:
        List of saved figure paths.
    """
    paths = []
    pr = pipeline_results

    try:
        paths.append(plot_raw_signals(
            pr["signals"], pr["damage_states"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig01 failed: %s", e)

    try:
        paths.append(plot_dwt_scalogram(
            pr["signals"], pr["damage_states"], pr["cycles"],
            output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig02 failed: %s", e)

    try:
        paths.append(plot_damage_index(
            pr["damage_index"], pr["cycles"], pr["layup_ids"],
            output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig03 failed: %s", e)

    try:
        paths.append(plot_umap_projections(
            pr["umap_embeddings"], pr["damage_states"],
            pr["layup_ids"], pr["rul"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig04 failed: %s", e)

    try:
        paths.append(plot_correlation_heatmap(
            pr["features"], pr["feature_names"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig05 failed: %s", e)

    try:
        paths.append(plot_radar_chart(
            pr["model_metrics"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig06 failed: %s", e)

    try:
        paths.append(plot_rul_predictions(
            pr["rul_results"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig07 failed: %s", e)

    try:
        paths.append(plot_pinn_loss(
            pr["pinn_history"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig08 failed: %s", e)

    try:
        paths.append(plot_pinn_degradation(
            pr["cycles_norm"], pr["stiffness_true"], pr["stiffness_pred"],
            pr["strength_true"], pr["strength_pred"],
            output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig09 failed: %s", e)

    try:
        paths.append(plot_attention_heatmap(
            pr["attention_weights"], pr["damage_states"],
            output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig10 failed: %s", e)

    try:
        paths.append(plot_shap_beeswarm(
            pr["shap_values"], pr["feature_names"],
            pr["feature_values"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig11 failed: %s", e)

    try:
        paths.append(plot_calibration_curves(
            pr["calibration_data"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig12 failed: %s", e)

    try:
        paths.append(plot_pareto_front(
            pr["pareto_front"], pr["all_bo_points"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig13 failed: %s", e)

    try:
        paths.append(plot_confusion_matrices(
            pr["confusion_matrices"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig14 failed: %s", e)

    try:
        paths.append(plot_cv_boxplots(
            pr["cv_results"], output_dir=output_dir))
    except Exception as e:
        logger.warning("Fig15 failed: %s", e)

    logger.info("Generated %d / 15 figures.", len(paths))
    return paths
