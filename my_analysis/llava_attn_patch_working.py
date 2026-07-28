"""
Working LLaVA attention patch for transformers 4.57+ using ClearSight's approach.
Replaces attention modules with adapters that scale attention logits before softmax.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,
    "sup_para": 1.0,
    "sys_len": 35,
    "img_len": 576,
}

_ORIGINAL_MODULES: dict = {}


class AttnAdapter(nn.Module):
    """Adapter that wraps LlamaAttention and applies ClearSight-style VAF.

    This adapter recomputes attention with multiplicative scaling on logits before softmax.
    Compatible with transformers 4.57+ Cache API.
    """

    def __init__(self, original_attn: nn.Module, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.original_attn = original_attn

        # Copy config and projections (share weights)
        self.config = original_attn.config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.hidden_size = self.config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        # rotary_emb is passed as position_embeddings argument in new API

        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(original_attn, 'attention_dropout', 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with ClearSight VAF."""
        # Debug logging (only on first call)
        if not hasattr(self, '_logged'):
            print(f"[AttnAdapter layer {self.layer_idx}] Called! enabled={_STATE['enabled']}, method={_STATE['method']}")
            self._logged = True

        # If not enabled, delegate to original
        if not _STATE["enabled"] or _STATE["method"] == "baseline":
            return self.original_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )

        bsz, q_len, _ = hidden_states.shape

        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        cos, sin = position_embeddings if position_embeddings is not None else (None, None)
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle cache
        if past_key_values is not None:
            # New Cache API: call update() method
            cache_kwargs = {"sin": sin, "cos": cos} if sin is not None else {}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat k/v heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores (logits)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Apply ClearSight VAF: multiplicative scaling on attention logits (before softmax!)
        enh_para = _STATE["enh_para"]
        sup_para = _STATE["sup_para"]
        sys_len = _STATE["sys_len"]
        img_len = _STATE["img_len"]

        kv_seq_len = key_states.shape[-2]

        # ClearSight's approach: scale attention from NEW tokens TO image/system tokens
        # This boosts/suppresses attention to visual/contextual info when generating new tokens
        if q_len > sys_len + img_len:
            # Generation phase: queries are after sys+img, keys are the full sequence
            # attn_weights[:, :, query_tokens, key_tokens]
            # Scale attention from generated queries TO image keys
            attn_weights[:, :, sys_len + img_len:, sys_len:sys_len + img_len] *= enh_para
            # Scale attention from generated queries TO system keys
            if sys_len > 0:
                attn_weights[:, :, sys_len + img_len:, :sys_len] *= sup_para
        else:
            # Prefill phase: queries include everything
            # Scale all attention TO image keys
            if sys_len + img_len <= kv_seq_len:
                attn_weights[:, :, :, sys_len:sys_len + img_len] *= enh_para
                # Scale all attention TO system keys
                if sys_len > 0:
                    attn_weights[:, :, :, :sys_len] *= sup_para

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast to fp32 for softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # Return with placeholder for attention weights (not used in generation)
        return attn_output, None


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0,
                sup_para: float = 1.0, layer_start: int = 9, layer_end: int = 14) -> None:
    """Replace attention modules with adapters."""
    global _ORIGINAL_MODULES

    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}")

    # Force eager attention
    model.config._attn_implementation = "eager"

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["enh_para"] = enh_para
    _STATE["sup_para"] = sup_para

    # Get language model layers
    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        raise AttributeError("Cannot find layers")

    # Replace attention modules in target layers
    if not _ORIGINAL_MODULES:
        for layer_idx in range(layer_start, layer_end + 1):
            original_attn = layers[layer_idx].self_attn
            _ORIGINAL_MODULES[layer_idx] = original_attn
            adapter = AttnAdapter(original_attn, layer_idx)
            layers[layer_idx].self_attn = adapter


def unpatch_model(model: Any) -> None:
    """Restore original attention modules."""
    global _ORIGINAL_MODULES

    if not _ORIGINAL_MODULES:
        return

    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        return

    for layer_idx, original_attn in _ORIGINAL_MODULES.items():
        layers[layer_idx].self_attn = original_attn

    _ORIGINAL_MODULES.clear()
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