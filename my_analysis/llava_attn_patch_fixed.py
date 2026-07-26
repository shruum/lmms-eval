"""
Fixed LLaVA attention patch using ClearSight's approach (multiplicative scaling on attention weights).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,  # Visual enhancement factor
    "sup_para": 1.0,  # System suppression factor
    "img_start": None,
    "img_end": None,
    "sys_end": None,
    "current_layer": -1,
    "layer_start": 9,
    "layer_end": 14,
    "salience_mask": None,  # Optional saliency for per-token scaling
}

_ORIGINAL_SOFTMAX = None
_HOOKS: list = []

def _patched_softmax(input: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """Patched softmax that applies multiplicative scaling to attention weights."""
    # Debug logging
    if _STATE["enabled"] and _STATE["method"] != "baseline":
        print(f"[PATCH] Called! dim={input.dim()}, in_lm={_STATE.get('in_language_model', False)}, layer={_STATE.get('current_layer', -1)}")

    if (
        _STATE["enabled"]
        and _STATE["method"] != "baseline"
        and input.dim() == 4
        and _STATE.get("in_language_model", False)
    ):
        layer_idx = _STATE["current_layer"]
        layer_start = _STATE["layer_start"]
        layer_end = _STATE["layer_end"]

        # Only apply to target layers
        if layer_start <= layer_idx <= layer_end:
            img_start = _STATE["img_start"]
            img_end = _STATE["img_end"]
            sys_end = _STATE["sys_end"]

            if img_start is not None and img_end is not None:
                # First compute softmax normally
                attn_weights = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

                # Apply multiplicative scaling (ClearSight's approach)
                enh_para = _STATE["enh_para"]
                sup_para = _STATE["sup_para"]
                sal = _STATE.get("salience_mask")

                # System suppression: multiply system token attention weights
                if sys_end is not None and sys_end >= 0 and sup_para != 1.0:
                    attn_weights[:, :, :, : sys_end + 1] *= sup_para

                # Image enhancement: multiply image token attention weights
                if enh_para != 1.0:
                    if sal is not None:
                        # Per-token enhancement based on saliency
                        sal_dev = sal.to(device=input.device, dtype=input.dtype)
                        # Create scaling factors: 1.0 + (enh_para - 1.0) * saliency
                        scaling = 1.0 + (enh_para - 1.0) * sal_dev
                        attn_weights[:, :, :, img_start : img_end + 1] *= scaling.unsqueeze(0).unsqueeze(0)
                    else:
                        # Uniform enhancement
                        attn_weights[:, :, :, img_start : img_end + 1] *= enh_para

                # Renormalize to maintain probability distribution
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

                return attn_weights

    # Default: pass through to original softmax
    return _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0, sup_para: float = 1.0) -> None:
    """Activate the attention intervention using F.softmax patching."""
    global _ORIGINAL_SOFTMAX

    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}")

    # Force eager attention
    model.config._attn_implementation = "eager"

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["enh_para"] = enh_para
    _STATE["sup_para"] = sup_para

    if _ORIGINAL_SOFTMAX is None:
        _ORIGINAL_SOFTMAX = torch.nn.functional.softmax
        torch.nn.functional.softmax = _patched_softmax

    # Register hooks to track layer index and language model context
    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        raise AttributeError(f"Cannot find layers")

    if not _HOOKS:
        def _lm_pre(module, args):
            _STATE["in_language_model"] = True

        def _lm_post(module, args, output):
            _STATE["in_language_model"] = False
            _STATE["current_layer"] = -1

        _HOOKS.append(lm.register_forward_pre_hook(_lm_pre))
        _HOOKS.append(lm.register_forward_hook(_lm_post))

        for layer_idx, layer in enumerate(layers):
            def _make_layer_pre(idx: int):
                def _layer_pre(module, args):
                    _STATE["current_layer"] = idx
                return _layer_pre
            _HOOKS.append(layer.self_attn.register_forward_pre_hook(_make_layer_pre(layer_idx)))


def unpatch_model(model: Any) -> None:
    """Restore original softmax and remove hooks."""
    global _ORIGINAL_SOFTMAX
    if _ORIGINAL_SOFTMAX is not None:
        torch.nn.functional.softmax = _ORIGINAL_SOFTMAX
        _ORIGINAL_SOFTMAX = None
    for h in _HOOKS:
        h.remove()
    _HOOKS.clear()
    _STATE["enabled"] = False
    _STATE["in_language_model"] = False


def update_sample(img_start: int, img_end: int) -> None:
    """Call once per sample before generate()."""
    _STATE["img_start"] = img_start
    _STATE["img_end"] = img_end
    _STATE["sys_end"] = max(0, img_start - 1)


def get_image_token_range(inputs: Any, model: Any) -> Tuple[int, int]:
    """Return (img_start, img_end) for LLaVA."""
    image_token_id = model.config.image_token_index
    ids_cpu = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]
    img_start = int(positions[0].item())
    vis_cfg = model.config.vision_config
    n_img_tokens = (vis_cfg.image_size // vis_cfg.patch_size) ** 2
    return img_start, img_start + n_img_tokens - 1
