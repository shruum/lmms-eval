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
    "srf_text_beta": 0.0,  # Post-image text suppression (language prior reduction)
    "srf_text_layer_start": 20,  # Deep layers where language priors form
    "srf_text_layer_end": 27,
    "srf_background_eps": 0.1,  # Suppress non-salient image tokens by this amount
    "suppress_visual_on_absent": False,  # Suppress ALL visual tokens when object absent
    "absence_detected": False,  # CLIP absence detection result
    "img_start": None,
    "img_end": None,
    "sys_end": None,
    "current_layer": -1,
    "layer_start": None,      # FIXED: No hardcoded values - set by srf.py via args
    "layer_end": None,        # FIXED: No hardcoded values - set by srf.py via args
    "salience_mask": None,     # Optional saliency for per-token scaling
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

        # CRITICAL VALIDATION: Ensure layer ranges are set before use
        if layer_start is None or layer_end is None:
            raise ValueError(
                f"layer_start and layer_end must be set by srf.py before using SRF! "
                f"Got: layer_start={layer_start}, layer_end={layer_end}. "
                f"Call srf.setup() first."
            )

        # Only apply to target layers
        if layer_start <= layer_idx <= layer_end:
            img_start = _STATE["img_start"]
            img_end = _STATE["img_end"]
            sys_end = _STATE["sys_end"]

            if layer_idx == layer_start:  # Debug: print once per run
                print(f"    [RANGE DEBUG] layer={layer_idx}, img=[{img_start}, {img_end}], sys_end={sys_end}")

            if img_start is not None and img_end is not None:
                # First compute softmax normally
                attn_weights = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

                # Apply multiplicative scaling (ClearSight's approach)
                base_enh_para = _STATE["enh_para"]
                sup_para = _STATE["sup_para"]
                sal = _STATE.get("salience_mask")
                head_mask = _STATE.get("head_mask")  # BUG #4 FIX: Use head mask

                # Layer-wise alpha strategy (Strategy 1)
                if _STATE.get("use_layerwise", False):
                    if layer_idx < layer_start:
                        # Early layers: weak boost
                        enh_para = base_enh_para * _STATE.get("early_alpha_mult", 0.3)
                    elif layer_start <= layer_idx <= layer_end:
                        # Middle layers: strong boost
                        enh_para = base_enh_para * _STATE.get("mid_alpha_mult", 1.5)
                    else:
                        # Late layers: minimal boost
                        enh_para = base_enh_para * _STATE.get("late_alpha_mult", 0.1)
                else:
                    enh_para = base_enh_para

                # System suppression: multiply system token attention weights
                if sys_end is not None and sys_end >= 0 and sup_para != 1.0:
                    if layer_idx == layer_start:  # Debug: print only once per layer range
                        print(f"    [SUPPRESS DEBUG] layer={layer_idx}, sys_end={sys_end}, sup_para={sup_para:.3f}")
                    attn_weights[:, :, :, : sys_end + 1] *= sup_para

                # Post-image text suppression (language prior reduction)
                # Suppresses text tokens AFTER image tokens to reduce parametric bias
                # Applied in deep layers where language priors form (default 20-27)
                text_beta = _STATE.get("srf_text_beta", 0.0)
                text_l_start = _STATE.get("srf_text_layer_start", 20)
                text_l_end = _STATE.get("srf_text_layer_end", 27)
                if text_beta > 0.0 and text_l_start <= layer_idx <= text_l_end:
                    n_kv = attn_weights.shape[-1]
                    if img_end + 1 < n_kv:
                        # Subtract text_beta from post-image text token logits
                        # Applied to ALL heads (language prior is not head-specific)
                        attn_weights[:, :, :, img_end + 1 :] *= (1.0 - text_beta)

                # Image enhancement: multiply image token attention weights
                if enh_para != 1.0:
                    if sal is not None and sal.numel() == (img_end - img_start + 1):
                        # Per-token enhancement based on saliency (only if dims match)
                        sal_dev = sal.to(device=input.device, dtype=input.dtype)
                        # Create scaling factors: 1.0 + (enh_para - 1.0) * saliency
                        scaling = 1.0 + (enh_para - 1.0) * sal_dev

                        # BUG #4 FIX: Apply only to vision-aware heads if head_mask exists
                        if head_mask is not None:
                            # Create head-specific scaling: vision heads get scaling, others get 1.0
                            n_heads = attn_weights.shape[1]
                            # head_scaling = torch.ones(n_heads, device=input.device, dtype=input.dtype)
                            # head_scaling[head_mask] = scaling.flatten()  # Only vision-aware heads [SHAPE MISMATCH BUG - COMMENTED OUT]
                            # Apply per-head scaling
                            for h in range(n_heads):
                                if head_mask[h]:
                                    attn_weights[:, :, h, img_start : img_end + 1] *= scaling.unsqueeze(0)
                        else:
                            # No head mask: apply to all heads (fallback)
                            attn_weights[:, :, :, img_start : img_end + 1] *= scaling.unsqueeze(0).unsqueeze(0)
                    else:
                        # Uniform enhancement (saliency disabled or dimension mismatch)
                        if head_mask is not None:
                            # BUG #4 FIX: Apply uniform enhancement only to vision-aware heads
                            n_heads = attn_weights.shape[1]
                            for h in range(n_heads):
                                if head_mask[h]:
                                    attn_weights[:, :, h, img_start : img_end + 1] *= enh_para
                        else:
                            # No head mask: apply to all heads (fallback)
                            attn_weights[:, :, :, img_start : img_end + 1] *= enh_para

                # Non-salient visual token suppression
                # Suppress non-salient image tokens by background_eps to reduce distraction
                background_eps = _STATE.get("srf_background_eps", 0.0)
                if background_eps > 0.0 and sal is not None and sal.numel() == (img_end - img_start + 1):
                    sal_dev = sal.to(device=input.device, dtype=input.dtype)
                    # Create suppression mask: 1.0 - eps for non-salient, 1.0 for salient
                    # Invert saliency for suppression (high saliency = low suppression)
                    suppress_mask = 1.0 - (1.0 - sal_dev) * background_eps
                    attn_weights[:, :, :, img_start : img_end + 1] *= suppress_mask.unsqueeze(0).unsqueeze(0)

                # Visual token suppression on absence detection
                # When object is absent, suppress ALL visual tokens to force reliance on language priors
                if _STATE.get("suppress_visual_on_absent", False) and _STATE.get("absence_detected", False):
                    attn_weights[:, :, :, img_start : img_end + 1] *= 0.1  # Strong suppression

                # Renormalize to maintain probability distribution
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

                return attn_weights

    # Default: pass through to original softmax
    return _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0, sup_para: float = 1.0, text_beta: float = 0.0) -> None:
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
    _STATE["srf_text_beta"] = text_beta

    if _ORIGINAL_SOFTMAX is None:
        _ORIGINAL_SOFTMAX = torch.nn.functional.softmax
        torch.nn.functional.softmax = _patched_softmax

    # Register hooks to track layer index and language model context
    # Handle different LLaVA architectures
    if hasattr(model, 'language_model'):
        lm = model.language_model
        transformer = lm
    elif hasattr(model, 'model'):
        lm = model.model  # LLaVA-1.5-7B: model.model is LlavaModel
        transformer = lm  # LlavaModel is the transformer
    elif hasattr(model, 'layers'):
        lm = model
        transformer = lm
    else:
        lm = model
        transformer = lm

    # Find the actual transformer layers
    if hasattr(lm, 'model') and not hasattr(transformer, 'layers'):
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
    """Return (img_start, img_end) for LLaVA.

    CRITICAL FIX: LLaVA internally expands a single <image> placeholder token into
    576 actual image tokens (24x24 patches from CLIP ViT-L/14 vision encoder).

    The attention mechanism operates on these EXPANDED tokens, not the placeholder.
    So we need to calculate the actual range that attention sees.
    """
    image_token_id = model.config.image_token_index
    ids_cpu = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]

    if len(positions) == 0:
        raise ValueError("No image tokens found in input")

    placeholder_pos = int(positions[0].item())

    # For LLaVA-1.5-7B: CLIP ViT-L/14 produces 24x24 = 576 image tokens
    # The single placeholder gets replaced with 576 actual tokens during forward pass
    # Attention operates on position [placeholder_pos, placeholder_pos + 575]
    grid_h, grid_w = 24, 24  # CLIP ViT-L/14 patch grid
    num_image_tokens = grid_h * grid_w  # 576

    img_start = placeholder_pos
    img_end = placeholder_pos + num_image_tokens - 1  # [X, X+575]

    print(f"    [IMG TOKENS FIXED] Placeholder at {placeholder_pos} → Actual image tokens: [{img_start}, {img_end}] ({num_image_tokens} tokens)")
    return img_start, img_end


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
