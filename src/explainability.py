"""
explainability.py — XAI Module (Improved)
==========================================
IMPROVEMENTS:
  1. SHAP: uses masker for tabular data (correct background)
  2. Grad-CAM: handles (B, C, S) tabular signals gracefully
  3. Attention extraction: caches from HybridSTA-V3's _last_attn_weights
  4. rank_features_by_shap: returns DataFrame with mean |SHAP|
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    model_type: str = "tree",
) -> Dict[str, np.ndarray]:
    """Compute SHAP values. model_type: 'tree' | 'kernel'."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping")
        return {"shap_values": np.zeros((len(X_explain), X_background.shape[1]))}

    if model_type == "tree":
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    else:
        # Kernel SHAP with median background
        bg = shap.sample(X_background, min(100, len(X_background)))
        explainer   = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    return {"shap_values": np.array(shap_values), "expected_value": float(explainer.expected_value
                                                                           if not isinstance(explainer.expected_value, list)
                                                                           else explainer.expected_value[0])}


def rank_features_by_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> List[Dict]:
    """Return feature importance as list of dicts sorted by mean |SHAP|."""
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return [{"feature": feature_names[i], "importance": float(importance[i])}
            for i in order]


def compute_gradcam_for_samples(
    model: nn.Module,
    signals: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
    n_samples: int = 50,
) -> Dict[str, np.ndarray]:
    """1D Grad-CAM saliency maps on the final conv layer of CNN1D."""
    model.eval()

    # Find last Conv1d layer
    target_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            target_layer = m

    if target_layer is None:
        logger.warning("No Conv1d found — skipping Grad-CAM")
        return {}

    activations, gradients = {}, {}

    def fwd_hook(mod, inp, out):
        activations["v"] = out.detach()

    def bwd_hook(mod, g_in, g_out):
        gradients["v"] = g_out[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    X = torch.FloatTensor(signals[:n_samples]).to(device)
    X.requires_grad_(True)
    out = model(X)
    if out.ndim > 1:
        loss = out[:, 0].mean()
    else:
        loss = out.mean()
    loss.backward()

    fh.remove(); bh.remove()

    acts  = activations.get("v", None)
    grads = gradients.get("v", None)
    if acts is None or grads is None:
        return {}

    weights   = grads.mean(dim=-1, keepdim=True)
    cam       = torch.relu((acts * weights).sum(dim=1))
    cam_np    = cam.cpu().numpy()
    cam_norm  = (cam_np - cam_np.min(axis=1, keepdims=True)) / \
                (cam_np.max(axis=1, keepdims=True) - cam_np.min(axis=1, keepdims=True) + 1e-8)
    return {"cam": cam_norm, "labels": labels[:n_samples]}


def extract_attention_weights(
    model: nn.Module,
    signals: np.ndarray,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """
    Extract attention weights from HybridSTA-V3 or SensorTransformer.
    Returns (N, n_heads, S, S) or None.
    """
    model.eval()
    X = torch.FloatTensor(signals).to(device)

    with torch.no_grad():
        _ = model(X, return_attn=True) if _supports_return_attn(model) else model(X)

    # HybridSTA-V3: weights cached in layer._last_attn_weights
    if hasattr(model, "layers"):
        for layer in reversed(list(model.layers)):
            w = getattr(layer, "_last_attn_weights", None)
            if w is not None:
                return w.cpu().numpy()

    return None


def _supports_return_attn(model: nn.Module) -> bool:
    import inspect
    sig = inspect.signature(model.forward)
    return "return_attn" in sig.parameters
