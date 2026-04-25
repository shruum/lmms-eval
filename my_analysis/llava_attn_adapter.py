"""
LLaVA attention adapter using ClearSight's approach (replaces attention modules).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple

# Lazy import of LlamaAttention to avoid early import issues
_LlamaAttention = None
_repeat_kv = None
_apply_rotary_pos_emb = None

_STATE: dict = {
    "enabled": False,
    "method": "baseline",
    "enh_para": 1.0,
    "sup_para": 1.0,
    "sys_len": 35,
    "img_len": 576,
}

_ORIGINAL_ATTN_MODULES: dict = {}  # Store original attention modules


class AttnAdapter(nn.Module):
    """Adapter that wraps LlamaAttention and applies ClearSight-style VAF."""

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.original_attn = original_attn

        # Copy all attributes from original attention
        self.config = original_attn.config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.hidden_size = self.config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Copy projections (share the same weights)
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Handle rotary embeddings - may not be a direct attribute
        if hasattr(original_attn, 'rotary_emb'):
            self.rotary_emb = original_attn.rotary_emb
        else:
            # Try to get it from the parent layer or model
            # Will access it through original_attn when needed
            self._rotary_emb_source = original_attn

        # Pretraining tensor parallelism (if any)
        self.pretraining_tp = getattr(original_attn, 'pretraining_tp', 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs  # Accept additional kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        # If not enabled or baseline, delegate to original
        if not _STATE["enabled"] or _STATE["method"] == "baseline":
            return self.original_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )

        # VAF enabled: compute attention with multiplicative scaling
        bsz, q_len, _ = hidden_states.size()

        # Compute Q, K, V
        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            # past_key_values is a tuple of (key, value) for this layer
            past_key, past_value = past_key_values
            kv_seq_len += past_key.shape[-2]

        # Rotary embeddings
        rotary_emb = getattr(self, 'rotary_emb', None)
        if rotary_emb is None:
            # Access through original attention module
            rotary_emb = self.original_attn.rotary_emb
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Cache
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        past_key_values = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        from transformers.models.llama.modeling_llama import repeat_kv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply ClearSight VAF: multiplicative scaling on attention weights (before softmax!)
        enh_para = _STATE["enh_para"]
        sup_para = _STATE["sup_para"]
        sys_len = _STATE["sys_len"]
        img_len = _STATE["img_len"]

        if q_len > sys_len + img_len:
            # Generation phase
            attn_weights[:, :, sys_len + img_len:, sys_len:sys_len + img_len] = enh_para * attn_weights[:, :, sys_len + img_len:, sys_len:sys_len + img_len]
            attn_weights[:, :, sys_len + img_len:, :sys_len] = sup_para * attn_weights[:, :, sys_len + img_len:, :sys_len]
        else:
            # Prefill phase
            attn_weights[:, :, :, sys_len:sys_len + img_len] = enh_para * attn_weights[:, :, :, sys_len:sys_len + img_len]
            attn_weights[:, :, :, :sys_len] = sup_para * attn_weights[:, :, :, :sys_len]

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_values, attn_weights


def patch_model(model: Any, method: str = "baseline", enh_para: float = 1.0, sup_para: float = 1.0,
                layer_start: int = 9, layer_end: int = 14) -> None:
    """Replace attention modules with AttnAdapter."""
    global _ORIGINAL_ATTN_MODULES

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
    if not _ORIGINAL_ATTN_MODULES:
        for layer_idx, layer in enumerate(layers):
            if layer_start <= layer_idx <= layer_end:
                # Store original and replace with adapter
                original_attn = layer.self_attn
                _ORIGINAL_ATTN_MODULES[layer_idx] = original_attn
                adapter = AttnAdapter(original_attn)
                layer.self_attn = adapter


def unpatch_model(model: Any) -> None:
    """Restore original attention modules."""
    global _ORIGINAL_ATTN_MODULES

    if not _ORIGINAL_ATTN_MODULES:
        return

    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        return

    for layer_idx, original_attn in _ORIGINAL_ATTN_MODULES.items():
        layers[layer_idx].self_attn = original_attn

    _ORIGINAL_ATTN_MODULES.clear()
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
