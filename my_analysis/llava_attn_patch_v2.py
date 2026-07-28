"""
LLaVA attention patch using hooks (simplest approach, works across transformers versions).
Based on ClearSight's multiplicative scaling approach.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Tuple, Optional

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,  # Visual enhancement multiplier
    "sup_para": 1.0,  # System suppression multiplier
    "sys_len": 35,  # System prompt token count
    "img_len": 576,  # Image token count
    "layer_start": 9,
    "layer_end": 14,
}

_HOOKS: list = []


class AttnWeightHook:
    """Hook that modifies attention weights after softmax."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, module: nn.Module, input: tuple, output: tuple) -> tuple:
        """Modify attention weights in the output tuple."""
        if not _STATE["enabled"] or _STATE["method"] == "baseline":
            return output

        # Check if we're in the target layer range
        if not (_STATE["layer_start"] <= self.layer_idx <= _STATE["layer_end"]):
            return output

        # Extract attention weights if present
        # output format for SDPA: (attn_output, attn_weights, past_key_values)
        # or just (attn_output,) if output_attentions=False
        if len(output) < 2 or output[1] is None:
            return output

        attn_weights = output[1]

        # Apply ClearSight VAF: multiplicative scaling
        enh_para = _STATE["enh_para"]
        sup_para = _STATE["sup_para"]
        sys_len = _STATE["sys_len"]
        img_len = _STATE["img_len"]

        # attn_weights shape: [batch, num_heads, seq_len, kv_seq_len]
        # During generation, seq_len is the number of tokens being generated (usually 1)
        # kv_seq_len is the total sequence length including cached tokens

        batch, num_heads, seq_len, kv_seq_len = attn_weights.shape

        # Check if we have image tokens in the key/value sequence
        if sys_len + img_len <= kv_seq_len:
            # Prefill or generation phase
            # Multiply attention to image tokens by enh_para
            attn_weights[:, :, :, sys_len:sys_len + img_len] *= enh_para

            # Multiply attention to system tokens by sup_para
            if sys_len > 0:
                attn_weights[:, :, :, :sys_len] *= sup_para

            # Renormalize to maintain probability distribution
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

            # Return modified output
            if len(output) == 2:
                return (output[0], attn_weights)
            else:
                return (output[0], attn_weights, output[2])

        return output


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0,
                sup_para: float = 1.0, layer_start: int = 9, layer_end: int = 14) -> None:
    """Register forward hooks on attention layers."""
    global _HOOKS

    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}")

    # Force eager attention to get attention weights
    model.config._attn_implementation = "eager"

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["enh_para"] = enh_para
    _STATE["sup_para"] = sup_para
    _STATE["layer_start"] = layer_start
    _STATE["layer_end"] = layer_end

    # Register hooks if not already done
    if not _HOOKS:
        lm = model.language_model
        if hasattr(lm, 'model'):
            layers = lm.model.layers
        elif hasattr(lm, 'layers'):
            layers = lm.layers
        else:
            raise AttributeError("Cannot find layers")

        for layer_idx, layer in enumerate(layers):
            # Register forward hook on the attention module
            # The hook receives (module, input, output)
            hook = layer.self_attn.register_forward_hook(AttnWeightHook(layer_idx))
            _HOOKS.append(hook)


def unpatch_model(model: Any) -> None:
    """Remove all hooks."""
    global _HOOKS
    for hook in _HOOKS:
        hook.remove()
    _HOOKS.clear()
    _STATE["enabled"] = False


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
