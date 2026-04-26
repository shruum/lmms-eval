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
    # Calibration state for identify_visual_heads
    "_calibrate_heads": False,
    "_calib_head_acc": None,
    "_calib_head_count": 0,
    "head_mask": None,
}

_ORIGINAL_SOFTMAX = None
_HOOKS: list = []

def _patched_softmax(input: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """Patched softmax that applies multiplicative scaling to attention weights."""

    # Calibration: accumulate per-head attention to image tokens
    if _STATE.get("_calibrate_heads", False) and input.dim() == 4 and _STATE.get("in_language_model", False):
        # First compute softmax normally
        attn_weights = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

        # Accumulate mean attention to image region for each head
        img_start = _STATE.get("img_start", 0)
        img_end = _STATE.get("img_end", 0)
        if img_end > img_start:
            # Extract attention to image tokens and average per head
            img_attn = attn_weights[:, :, :, img_start : img_end + 1].mean(dim=-1)  # (batch, heads, seq)
            img_attn = img_attn.mean(dim=(0, 2))  # Average over batch and seq → (n_heads,)

            # Accumulate
            if _STATE["_calib_head_acc"] is None:
                _STATE["_calib_head_acc"] = img_attn
            else:
                _STATE["_calib_head_acc"] += img_attn
            _STATE["_calib_head_count"] += 1

        return attn_weights

    # Normal SRF intervention
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
        transformer = lm.model  # LLaVA: LlamaForCausalLM.model -> LlamaModel
    elif hasattr(lm, 'layers'):
        transformer = lm
    else:
        raise AttributeError(f"Cannot find transformer in language_model")

    if hasattr(transformer, 'layers'):
        layers = transformer.layers
    elif hasattr(transformer, 'h'):
        layers = transformer.h
    else:
        raise AttributeError(f"Cannot find layers in transformer")

    if not _HOOKS:
        def _transformer_pre(module, args):
            _STATE["in_language_model"] = True

        def _transformer_post(module, args, output):
            _STATE["in_language_model"] = False
            _STATE["current_layer"] = -1

        _HOOKS.append(transformer.register_forward_pre_hook(_transformer_pre))
        _HOOKS.append(transformer.register_forward_hook(_transformer_post))

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


def identify_visual_heads(
    model: Any,
    calibration_inputs: List[Any],
    img_ranges: List[Tuple[int, int]],
    top_k_pct: float = 0.20,
) -> torch.Tensor:
    """
    Compute per-head mean attention to image tokens across calibration samples
    and return a bool mask selecting the top top_k_pct vision-aware heads.

    The result is also stored in _STATE["head_mask"] so vhr_boost picks it up
    automatically.

    Args:
        model               : patched model (eager attention)
        calibration_inputs  : list of preprocessed input dicts (on model device)
        img_ranges          : list of (img_start, img_end) per sample
        top_k_pct           : fraction of heads to select (default 0.20 = top 20%)

    Returns:
        head_mask : bool tensor of shape (n_heads,)
    """
    assert len(calibration_inputs) == len(img_ranges), \
        "calibration_inputs and img_ranges must have equal length"
    assert 0 < top_k_pct <= 1.0, "top_k_pct must be in (0, 1]"

    # Run in baseline mode so no intervention distorts attention
    patch_model(model, "baseline", 1.0)

    # Debug: check if hooks are registered
    print(f"  [CALIB] Registered {_HOOKS} hooks")

    # Reset calibration state
    _STATE["_calibrate_heads"]  = True
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    with torch.no_grad():
        for inputs, (img_start, img_end) in zip(calibration_inputs, img_ranges):
            update_sample(img_start, img_end)
            model(**inputs)

    _STATE["_calibrate_heads"] = False

    count = _STATE["_calib_head_count"]
    print(f"  [CALIB] Captured {count} softmax calls")
    assert count > 0, (
        "VHR calibration captured 0 decoder softmax calls — "
        "check that the model uses eager attention and hooks are active"
    )

    scores   = _STATE["_calib_head_acc"] / count   # (n_heads,)
    n_heads  = len(scores)
    k        = max(1, round(n_heads * top_k_pct))
    # Use topk to find the threshold — avoids floating-point tie issues
    topk_vals = scores.topk(k).values
    threshold = topk_vals[-1]
    head_mask = scores >= threshold

    _STATE["head_mask"]         = head_mask
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    return head_mask
