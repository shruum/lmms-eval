"""
LLaVA attention patch — post-softmax multiplicative scaling (ClearSight approach).

Key differences from qwen_attn_patch:
  - Post-softmax multiplicative (not pre-softmax additive logit).
  - Uses enh_para/sup_para (not value/srf_background_eps).
  - Layer range read from vaf_layer_start/vaf_layer_end (_sync_patch_state writes these).
  - boost alpha read from _STATE["value"] first, fallback to _STATE["enh_para"].
  - Context gate uses in_language_model (hooks on lm.model / LlamaModel).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,            # Visual enhancement factor (multiplicative)
    "sup_para": 1.0,            # System suppression factor (multiplicative)
    "img_start": None,
    "img_end": None,
    "sys_end": None,
    "current_layer": -1,
    # Layer range — _sync_patch_state writes vaf_layer_start/end; fall back to layer_start/end.
    "layer_start": 9,
    "layer_end": 14,
    "vaf_layer_start": 9,
    "vaf_layer_end": 14,
    "vaf_beta": 0.0,
    "salience_mask": None,
    # Boost mode — controls how image token attention is modified:
    #   "multiplicative" : post-softmax multiply + renorm (original, VAF-compatible)
    #   "additive"       : pre-softmax logit addition (matches qwen_attn_patch, SRF-native)
    "boost_mode": "multiplicative",
    # New srf.py writes these keys via _sync_patch_state / prepare_sample — kept as no-ops.
    "value": None,              # boost alpha from srf.py; None = use enh_para
    "srf_background_eps": 0.0,
    "srf_bias_mode": "multiplicative",
    "srf_interp_lambda": 1.0,
    "srf_prob_floor": 0.005,
    "srf_img_scale": 1.5,
    "srf_apply_phase": "both",
    "srf_text_beta": 0.0,
    "srf_text_layer_start": 20,
    "srf_text_layer_end": 27,
    "srf_layer_alphas": None,
    "srf_vr_target": 0.0,
    "srf_vr_k": 3.0,
    # Calibration state
    "_calibrate_heads": False,
    "_calib_head_acc": None,
    "_calib_head_count": 0,
    "head_mask": None,
    "in_language_model": False,
}

_ORIGINAL_SOFTMAX = None
_HOOKS: list = []


def _patched_softmax(input: torch.Tensor, dim: int = -1,
                     dtype: Optional[torch.dtype] = None, **kwargs) -> torch.Tensor:
    """Post-softmax multiplicative scaling of image token attention weights."""

    # Calibration: accumulate per-head attention to image tokens
    if _STATE.get("_calibrate_heads", False) and input.dim() == 4 and _STATE.get("in_language_model", False):
        attn_weights = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)
        img_start = _STATE.get("img_start", 0)
        img_end   = _STATE.get("img_end",   0)
        if img_end > img_start:
            img_attn = attn_weights[:, :, :, img_start:img_end + 1].mean(dim=-1)
            img_attn = img_attn.mean(dim=(0, 2))   # (n_heads,)
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
        # Layer range: prefer vaf_layer_start/end (written by _sync_patch_state)
        layer_start = _STATE.get("vaf_layer_start", _STATE.get("layer_start", 9))
        layer_end   = _STATE.get("vaf_layer_end",   _STATE.get("layer_end",   14))
        layer_idx   = _STATE["current_layer"]

        if layer_start <= layer_idx <= layer_end:
            img_start = _STATE["img_start"]
            img_end   = _STATE["img_end"]
            sys_end   = _STATE["sys_end"]

            if img_start is not None and img_end is not None:
                attn_weights = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

                # Boost alpha: prefer _STATE["value"] (set by new srf.py prepare_sample)
                # Fall back to enh_para (set by patch_model / old calling convention).
                value    = _STATE.get("value")
                enh_para = float(value) if value is not None else float(_STATE["enh_para"])
                sup_para = float(_STATE["sup_para"])
                sal      = _STATE.get("salience_mask")

                # System-token suppression (multiplicative)
                if sys_end is not None and sys_end >= 0 and sup_para != 1.0:
                    attn_weights[:, :, :, :sys_end + 1] *= sup_para

                boost_mode = _STATE.get("boost_mode", "multiplicative")

                if boost_mode == "additive":
                    # Pre-softmax additive logit — same mechanism as qwen_attn_patch.
                    # Works cleanly for both positive (boost) and negative (suppress).
                    if enh_para != 0.0:
                        n_img = img_end - img_start + 1
                        modified = input.clone()
                        if sal is not None and sal.numel() == n_img:
                            sal_dev = sal.to(device=input.device, dtype=input.dtype)
                            modified[:, :, :, img_start:img_end + 1] += enh_para * sal_dev.unsqueeze(0).unsqueeze(0)
                        else:
                            modified[:, :, :, img_start:img_end + 1] += enh_para
                        return _ORIGINAL_SOFTMAX(modified, dim=dim, dtype=dtype, **kwargs)
                else:
                    # Post-softmax multiplicative (original, VAF-compatible)
                    if enh_para != 1.0:
                        n_img = img_end - img_start + 1
                        if sal is not None and sal.numel() == n_img:
                            sal_dev  = sal.to(device=input.device, dtype=input.dtype)
                            scaling  = 1.0 + (enh_para - 1.0) * sal_dev
                            attn_weights[:, :, :, img_start:img_end + 1] *= scaling.unsqueeze(0).unsqueeze(0)
                        else:
                            attn_weights[:, :, :, img_start:img_end + 1] *= enh_para
                        # Renormalise to maintain probability distribution
                        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                return attn_weights

    return _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)


def patch_model(model: Any, method: str = "baseline",
                enh_para: float = 1.0, sup_para: float = 1.0) -> None:
    """Activate the attention intervention using F.softmax patching."""
    global _ORIGINAL_SOFTMAX

    if method not in ("baseline", "srf", "vaf"):
        raise ValueError(f"Unknown method: {method!r}")

    model.config._attn_implementation = "eager"
    _STATE["enabled"]  = True
    _STATE["method"]   = method
    _STATE["enh_para"] = enh_para
    _STATE["sup_para"] = sup_para

    if _ORIGINAL_SOFTMAX is None:
        _ORIGINAL_SOFTMAX = torch.nn.functional.softmax
        torch.nn.functional.softmax = _patched_softmax

    # Find the LlamaModel (or equivalent) inside language_model.
    # Transformers 5.x: LlavaForConditionalGeneration.model.language_model
    # Transformers 4.x: LlavaForConditionalGeneration.language_model
    if hasattr(model, "language_model"):
        lm = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    else:
        raise AttributeError(f"Cannot find language_model in {type(model)}")

    if hasattr(lm, "model"):
        transformer = lm.model          # LlavaForCausalLM → LlamaForCausalLM → LlamaModel
    elif hasattr(lm, "layers"):
        transformer = lm
    else:
        raise AttributeError(f"Cannot find transformer in language_model: {type(lm)}")

    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "h"):
        layers = transformer.h
    else:
        raise AttributeError(f"Cannot find layers in transformer: {type(transformer)}")

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
    _STATE["img_end"]   = img_end
    _STATE["sys_end"]   = max(0, img_start - 1)


def get_image_token_range(inputs: Any, model: Any) -> Tuple[int, int]:
    """Return (img_start, img_end) for LLaVA-1.5 (single image, square patches)."""
    image_token_id = model.config.image_token_index
    ids_cpu   = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]
    img_start = int(positions[0].item())
    vis_cfg   = model.config.vision_config
    n_tokens  = (vis_cfg.image_size // vis_cfg.patch_size) ** 2
    return img_start, img_start + n_tokens - 1


def identify_visual_heads(
    model: Any,
    calibration_inputs: List[Any],
    img_ranges: List[Tuple[int, int]],
    top_k_pct: float = 0.20,
) -> torch.Tensor:
    """
    Identify the top top_k_pct vision-aware heads by mean attention to image tokens.
    Result stored in _STATE["head_mask"] (bool tensor of shape (n_heads,)).
    """
    assert len(calibration_inputs) == len(img_ranges)
    assert 0 < top_k_pct <= 1.0

    patch_model(model, "baseline", 1.0)

    _STATE["_calibrate_heads"]  = True
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    with torch.no_grad():
        for inputs, (img_start, img_end) in zip(calibration_inputs, img_ranges):
            update_sample(img_start, img_end)
            model(**inputs)

    _STATE["_calibrate_heads"] = False

    count = _STATE["_calib_head_count"]
    assert count > 0, (
        "LLaVA calibration captured 0 softmax calls — "
        "check that eager attention is set and hooks are active"
    )

    scores   = _STATE["_calib_head_acc"] / count
    n_heads  = len(scores)
    k        = max(1, round(n_heads * top_k_pct))
    topk_vals = scores.topk(k).values
    threshold = topk_vals[-1]
    head_mask = scores >= threshold

    _STATE["head_mask"]         = head_mask
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0
    return head_mask
