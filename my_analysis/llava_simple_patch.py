"""
Simple LLaVA attention patch using F.softmax interception (same as qwen_attn_patch).
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
    "value": 1.0,
    "img_start": None,
    "img_end": None,
    "sys_end": None,
    "head_mask": None,
    "vaf_beta": 0.1,
    "vaf_layer_start": 8,
    "vaf_layer_end": 15,
    "current_layer": -1,
    "salience_mask": None,
    "srf_background_eps": 0.1,
    "srf_bias_mode": "additive_logit",
    "srf_apply_phase": "both",
}

_ORIGINAL_SOFTMAX = None
_HOOKS: list = []

def _patched_softmax(input: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """Patched softmax that adds bias to attention logits before softmax."""
    if (
        _STATE["enabled"]
        and _STATE["method"] != "baseline"
        and input.dim() == 4
        and _STATE.get("in_language_model", False)
    ):
        layer_idx = _STATE["current_layer"]
        layer_start = _STATE["vaf_layer_start"]
        layer_end = _STATE["vaf_layer_end"]

        if layer_start <= layer_idx <= layer_end:
            img_start = _STATE["img_start"]
            img_end = _STATE["img_end"]

            if img_start is not None and img_end is not None:
                input = input.clone()
                sal = _STATE["salience_mask"]
                beta = _STATE["vaf_beta"]
                sys_end = _STATE["sys_end"]
                value = _STATE["value"]
                bias_mode = _STATE.get("srf_bias_mode", "additive_logit")

                # System suppression
                if sys_end is not None and sys_end > 0 and beta > 0:
                    input[:, :, :, : sys_end + 1] -= beta

                # Image boost (additive_logit only for now)
                if bias_mode == "additive_logit" and sal is not None:
                    n_img = img_end - img_start + 1
                    sal_dev = sal.to(device=input.device, dtype=input.dtype)
                    alpha_val = float(value)
                    bias_row = alpha_val * sal_dev
                    input[:, :, :, img_start : img_end + 1] += bias_row

    return _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

def patch_model(model: Any, method: str = "baseline", value: float = 1.0) -> None:
    """Activate the attention intervention using F.softmax patching."""
    global _ORIGINAL_SOFTMAX

    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}")

    # Force eager attention
    model.config._attn_implementation = "eager"

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["value"] = value

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
