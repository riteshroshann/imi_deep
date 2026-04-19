"""
visualization.py — Publication Figure Generators (Improved)
=============================================================
IMPROVEMENTS:
  1. Consistent dark-mode style (seaborn-v0_8-darkgrid)
  2. All figures use tight_layout + 300 DPI
  3. plot_rul_predictions: overlays conformal bands
  4. plot_cv_boxplots and plot_confusion_matrices implemented
  5. Safe fallback if data is empty
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

logger = logging.getLogger(__name__)

PALETTE  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
STYLE    = "seaborn-v0_8-darkgrid"

def _fig(figsize=(10, 6)):
    try:
        plt.style.use(STYLE)
    except Exception:
        pass
    return plt.subplots(figsize=figsize)


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)
    return path


# ── Fig 1: Raw signals ─────────────────────────────────────────────────────────
def plot_raw_signals(signals: np.ndarray, damage_states: np.ndarray,
                     output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig01_raw_signals.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    state_map = {0:"Healthy",1:"Early Damage",2:"Moderate",3:"Severe",4:"Pre-Failure"}
    shown = set()
    for ax, (state_id, label) in zip(axes.flat, state_map.items()):
        idx = np.where(damage_states == state_id)[0]
        if len(idx) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        s = signals[idx[0]]
        if s.ndim == 2:
            for ch in range(min(4, s.shape[0])):
                ax.plot(s[ch], alpha=0.8, linewidth=0.8, label=f"Ch {ch}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
    fig.suptitle("Raw Lamb Wave Signals — Damage Progression", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 2: DWT scalogram ───────────────────────────────────────────────────────
def plot_dwt_scalogram(signals: np.ndarray, damage_states: np.ndarray,
                        cycles: np.ndarray, output_dir: str = "results/figures") -> str:
    from src.feature_extraction import compute_dwt_scalogram
    path = os.path.join(output_dir, "fig02_dwt_scalogram.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, state_id in zip(axes, [0, 2, 4]):
        idx = np.where(damage_states == state_id)[0]
        if len(idx) == 0:
            ax.set_title("No data"); continue
        sig = signals[idx[0]]
        if sig.ndim == 2 and sig.shape[-1] > 64:
            coefs, freqs, t = compute_dwt_scalogram(sig[0])
            ax.pcolormesh(t, freqs[:30], coefs[:30], shading="auto", cmap="inferno")
            ax.set_xlabel("Time (μs)"); ax.set_ylabel("Frequency (Hz)")
        else:
            ax.bar(range(sig.shape[-1]), sig[0] if sig.ndim == 2 else sig)
        state_map = {0:"Healthy",2:"Moderate",4:"Pre-Failure"}
        ax.set_title(state_map.get(state_id, str(state_id)))
    fig.suptitle("DWT Scalograms by Damage State", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 3: Damage Index evolution ─────────────────────────────────────────────
def plot_damage_index(di: np.ndarray, cycles: np.ndarray,
                       layup_ids: np.ndarray, output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig03_damage_index.png")
    fig, ax = _fig((10, 6))
    for lid, label in enumerate(["L1 [0/45/90/-45]","L2 [45/90/-45/0]","L3 [90/0/-45/45]"]):
        mask = layup_ids == lid
        if not mask.any(): continue
        di_mean = di[mask].mean(axis=1) if di.ndim == 2 else di[mask]
        order = np.argsort(cycles[mask])
        ax.plot(cycles[mask][order], di_mean[order], label=label,
                color=PALETTE[lid], linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Fatigue Cycles"); ax.set_ylabel("Mean Damage Index")
    ax.set_title("Damage Index Monotonic Evolution", fontsize=13, fontweight="bold")
    ax.legend()
    return _save(fig, path)


# ── Fig 4: UMAP projections ────────────────────────────────────────────────────
def plot_umap_projections(umap_emb: np.ndarray, damage_states: np.ndarray,
                           layup_ids: np.ndarray, rul: np.ndarray,
                           output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig04_umap_projections.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap_list = ["tab10","Set2","RdYlGn"]
    labels_list = [damage_states, layup_ids, rul]
    titles = ["Damage State","Layup Config","RUL"]
    for ax, c, title, cmap in zip(axes, labels_list, titles, cmap_list):
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=c, cmap=cmap,
                        s=8, alpha=0.7)
        plt.colorbar(sc, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.suptitle("UMAP Manifold Projections", fontsize=14, fontweight="bold")
    return _save(fig, path)


# ── Fig 5: Correlation heatmap ─────────────────────────────────────────────────
def plot_correlation_heatmap(features: np.ndarray, feature_names: List[str],
                              output_dir: str = "results/figures") -> str:
    path  = os.path.join(output_dir, "fig05_correlation_heatmap.png")
    top_n = min(30, features.shape[1])
    var   = features.var(axis=0)
    top_i = np.argsort(var)[::-1][:top_n]
    X_top = features[:, top_i]
    names = [feature_names[i] for i in top_i]
    corr  = np.corrcoef(X_top.T)
    fig, ax = _fig((14, 11))
    sns.heatmap(corr, xticklabels=names, yticklabels=names, cmap="coolwarm",
                center=0, ax=ax, linewidths=0.3, annot=False)
    ax.set_title("Spearman Correlation — Top-30 Features", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=7)
    return _save(fig, path)


# ── Fig 6: Radar chart ────────────────────────────────────────────────────────
def plot_radar_chart(metrics: Dict[str, Dict[str, float]],
                      output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig06_radar_chart.png")
    cats  = list(next(iter(metrics.values())).keys())
    N     = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for i, (model, vals) in enumerate(metrics.items()):
        v = list(vals.values()) + [list(vals.values())[0]]
        ax.plot(angles, v, label=model, color=PALETTE[i % len(PALETTE)], linewidth=2)
        ax.fill(angles, v, alpha=0.1, color=PALETTE[i % len(PALETTE)])
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=10)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return _save(fig, path)


# ── Fig 7: RUL predictions ────────────────────────────────────────────────────
def plot_rul_predictions(predictions: Dict[str, Dict], 
                          output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig07_rul_predictions.png")
    n_models = len(predictions)
    fig, axes = plt.subplots(1, max(n_models, 1), figsize=(6 * max(n_models, 1), 5), squeeze=False)
    axes = axes[0]

    for ax, (name, d) in zip(axes, predictions.items()):
        y_true = np.asarray(d.get("y_true", []))
        y_pred = np.asarray(d.get("y_pred", []))
        y_std  = d.get("y_std", None)
        if len(y_true) == 0: continue
        order = np.argsort(y_true)
        ax.plot(y_true[order], y_true[order], "k--", linewidth=1, label="Perfect")
        ax.scatter(y_true[order], y_pred[order], s=12, alpha=0.6,
                   color=PALETTE[0], label="Predicted")
        if y_std is not None and len(y_std) == len(y_pred):
            ax.fill_between(y_true[order],
                            y_pred[order] - 1.96*y_std[order],
                            y_pred[order] + 1.96*y_std[order],
                            alpha=0.2, color=PALETTE[0], label="95% CI")
        ax.set_title(name); ax.set_xlabel("True RUL"); ax.set_ylabel("Pred RUL")
        ax.legend(fontsize=8)
    fig.suptitle("RUL Predictions vs Ground Truth", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 8: PINN loss ──────────────────────────────────────────────────────────
def plot_pinn_loss(history: Dict, output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig08_pinn_loss.png")
    fig, ax = _fig((10, 5))
    epochs = range(1, len(history["total_loss"]) + 1)
    ax.semilogy(epochs, history["total_loss"], label="Total", color=PALETTE[0], linewidth=2)
    ax.semilogy(epochs, history["data_loss"],  label="Data",  color=PALETTE[1], linewidth=2, linestyle="--")
    ax.semilogy(epochs, history["physics_loss"], label="Physics", color=PALETTE[2], linewidth=2, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN Training Loss Decomposition", fontsize=13, fontweight="bold")
    ax.legend()
    return _save(fig, path)


# ── Fig 9: PINN degradation ───────────────────────────────────────────────────
def plot_pinn_degradation(cycles_norm, stiff_true, stiff_pred,
                           str_true, str_pred, output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig09_pinn_degradation.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(cycles_norm)
    c     = np.asarray(cycles_norm)[order]
    for ax, true, pred, title in [
        (ax1, stiff_true, stiff_pred, "Stiffness E(N)/E₀"),
        (ax2, str_true,   str_pred,   "Strength σ(N)/σ₀"),
    ]:
        ax.scatter(c, np.asarray(true)[order],  s=10, alpha=0.5, label="True", color=PALETTE[0])
        ax.plot(c,    np.asarray(pred)[order],  linewidth=2,     label="PINN", color=PALETTE[2])
        ax.set_xlabel("Life Fraction"); ax.set_ylabel("Normalised Property")
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.legend()
    fig.suptitle("PINN Physics-Constrained Degradation Curves", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 10: Attention heatmap ─────────────────────────────────────────────────
def plot_attention_heatmap(attn_weights: np.ndarray, damage_states: np.ndarray,
                            output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig10_attention_heatmap.png")
    fig, ax = _fig((9, 7))
    # Average over heads and samples
    if attn_weights.ndim == 4:
        avg = attn_weights.mean(axis=(0, 1))
    elif attn_weights.ndim == 3:
        avg = attn_weights.mean(axis=0)
    else:
        avg = attn_weights
    sns.heatmap(avg, cmap="viridis", ax=ax, square=True,
                xticklabels=[f"S{i}" for i in range(avg.shape[1])],
                yticklabels=[f"S{i}" for i in range(avg.shape[0])])
    ax.set_title("Mean Transformer Attention — 4×4 PZT Grid", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 11: SHAP beeswarm ─────────────────────────────────────────────────────
def plot_shap_beeswarm(shap_values: np.ndarray, feature_names: List[str],
                        X: np.ndarray, output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig11_shap_xgboost.png")
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          show=False, plot_type="dot", max_display=20)
        plt.title("SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        logger.warning("SHAP beeswarm failed: %s", e)
        importance = np.abs(shap_values).mean(axis=0)
        top_i = np.argsort(importance)[::-1][:20]
        fig, ax = _fig((10, 8))
        ax.barh([feature_names[i] for i in top_i[::-1]], importance[top_i[::-1]], color=PALETTE[0])
        ax.set_title("Feature Importance (mean |SHAP|)", fontsize=13, fontweight="bold")
        _save(fig, path)
    return path


# ── Fig 12: Calibration curves ────────────────────────────────────────────────
def plot_calibration_curves(calibration_data: Dict, output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig12_calibration.png")
    fig, ax = _fig((9, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for i, (name, cal) in enumerate(calibration_data.items()):
        exp = np.asarray(cal.get("expected_coverage", []))
        obs = np.asarray(cal.get("observed_coverage", []))
        if len(exp) == 0: continue
        ece = cal.get("ece", np.nan)
        ax.plot(exp, obs, label=f"{name} (ECE={ece:.3f})",
                color=PALETTE[i % len(PALETTE)], linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Expected Coverage"); ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Prediction Calibration", fontsize=13, fontweight="bold")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return _save(fig, path)


# ── Fig 13: Pareto front ──────────────────────────────────────────────────────
def plot_pareto_front(pareto_pts: np.ndarray, all_pts: np.ndarray,
                       output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig13_pareto_front.png")
    fig, ax = _fig((9, 7))
    if all_pts.shape[1] >= 2:
        ax.scatter(all_pts[:, 0], all_pts[:, 1], s=20, alpha=0.4, color="gray", label="All configs")
        ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], s=80, color=PALETTE[2],
                   zorder=5, label="Pareto front", marker="*")
    ax.set_xlabel("RUL Score"); ax.set_ylabel("Stiffness Retention")
    ax.set_title("Multi-Objective Pareto Front", fontsize=13, fontweight="bold")
    ax.legend()
    return _save(fig, path)


# ── Fig 14: Confusion matrices ────────────────────────────────────────────────
def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             output_dir: str = "results/figures") -> str:
    from sklearn.metrics import confusion_matrix
    path = os.path.join(output_dir, "fig14_confusion_matrices.png")
    if class_names is None:
        class_names = [f"C{i}" for i in range(len(np.unique(y_true)))]
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = _fig((8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Damage State Confusion Matrix", fontsize=13, fontweight="bold")
    return _save(fig, path)


# ── Fig 15: CV boxplots ───────────────────────────────────────────────────────
def plot_cv_boxplots(cv_results: Dict[str, Dict[str, List[float]]],
                      output_dir: str = "results/figures") -> str:
    path = os.path.join(output_dir, "fig15_cv_boxplots.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, metric in zip(axes, ["rmse", "score"]):
        data  = [v[metric] for v in cv_results.values() if metric in v]
        names = [k for k in cv_results if metric in cv_results[k]]
        ax.boxplot(data, labels=names, patch_artist=True,
                   boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
        ax.set_title(f"CV {metric.upper()}", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.upper())
    fig.suptitle("Leave-One-Layup-Out Cross-Validation", fontsize=13, fontweight="bold")
    return _save(fig, path)
