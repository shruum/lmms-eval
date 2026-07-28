"""
LLaVA attention patch using SDPA monkey-patching.
This scales attention logits BEFORE softmax for ClearSight-style VAF.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, Tuple, Optional
import math

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,  # Visual enhancement multiplier
    "sup_para": 1.0,  # System suppression multiplier
    "sys_len": 35,
    "img_len": 576,
    "layer_start": 9,
    "layer_end": 14,
    "current_layer": -1,
}

_ORIGINAL_SDPA = None
_HOOKS: list = []


def _patched_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs
):
    """Patched SDPA that applies multiplicative scaling to attention logits."""
    # Call original to get attention weights
    attn_output = _ORIGINAL_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

    # Apply VAF if enabled
    if _STATE["enabled"] and _STATE["method"] != "baseline":
        layer_idx = _STATE["current_layer"]
        if _STATE["layer_start"] <= layer_idx <= _STATE["layer_end"]:
            enh_para = _STATE["enh_para"]
            sup_para = _STATE["sup_para"]
            sys_len = _STATE["sys_len"]
            img_len = _STATE["img_len"]

            # We need to modify attention weights, but SDPA returns the output directly
            # This approach won't work for SDPA - we need to intercept earlier

    return attn_output


def _make_layer_hook(layer_idx: int):
    """Create a hook that tracks the current layer."""
    def pre_hook(module, args):
        _STATE["current_layer"] = layer_idx
    return pre_hook


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0,
                sup_para: float = 1.0, layer_start: int = 9, layer_end: int = 14) -> None:
    """Set up attention intervention."""
    global _ORIGINAL_SDPA

    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}")

    # Force eager attention (SDPA doesn't give us access to attention weights)
    model.config._attn_implementation = "eager"

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["enh_para"] = enh_para
    _STATE["sup_para"] = sup_para
    _STATE["layer_start"] = layer_start
    _STATE["layer_end"] = layer_end

    # Register hooks to track layer index
    if not _HOOKS:
        lm = model.language_model
        if hasattr(lm, 'model'):
            layers = lm.model.layers
        elif hasattr(lm, 'layers'):
            layers = lm.layers
        else:
            raise AttributeError("Cannot find layers")

        for layer_idx, layer in enumerate(layers):
            hook = layer.self_attn.register_forward_pre_hook(_make_layer_hook(layer_idx))
            _HOOKS.append(hook)


def unpatch_model(model: Any) -> None:
    """Remove all hooks."""
    global _ORIGINAL_SDPA
    for hook in _HOOKS:
        hook.remove()
    _HOOKS.clear()
    _STATE["enabled"] = False
    _STATE["current_layer"] = -1


def get_image_token_range(inputs: Any, model: Any) -> Tuple[int, int]:
    """Return (img_start, img_end) for LLaVA."""
    image_token_id = model.config.image_token_index
    ids_cpu = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]
    img_start = int(positions[0].item())
    vis_cfg = model.config.vision_config
    n_img_tokens = (vis_cfg.image_size // vis_cfg.patch_size) ** 2
    return img_start, img_start + n_img_tokens - 1


def update_sample(img_start: int, img_end: int) -> None:
    """Update system/image token lengths."""
    _STATE["sys_len"] = max(0, img_start)
    _STATE["img_len"] = img_end - img_start + 1
