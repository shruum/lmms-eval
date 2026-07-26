#!/usr/bin/env python3
"""
VAF (Visual Amplification Fusion) - ClearSight method implementation.

Simple attention amplification that boosts visual tokens and suppresses system tokens.
Based on ClearSight paper (CVPR 2025).

Parameters (from ClearSight paper):
  - alpha: 0.15 (visual enhancement strength)
  - beta: 0.1 (system suppression strength)
  - layers: 10-15 (middle fusion layers)
  - enh_para = 1.0 + alpha = 1.15
  - sup_para = 1.0 - beta = 0.9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import sys


class VAFPatcher:
    """VAF (Visual Amplification Fusion) patcher for LLaVA models."""

    def __init__(self):
        self.enabled = False
        self.patched_layers = {}
        self.original_attn_modules = {}

        # VAF parameters (ClearSight defaults)
        self.enh_para = 1.15    # Enhancement: 1.0 + 0.15
        self.sup_para = 0.9     # Suppression: 1.0 - 0.1
        self.layer_start = 10   # First layer to patch
        self.layer_end = 15     # Last layer to patch

        # Token length tracking
        self.sys_len = 35        # System prompt length
        self.img_len = 576       # Image tokens (24x24 patches for LLaVA-1.5-7B)

    def patch_model(self, model: Any) -> None:
        """Patch model with VAF attention adapters."""
        if not self.enabled:
            return

        print(f"[VAF] Patching layers {self.layer_start}-{self.layer_end}")
        print(f"[VAF] enh_para={self.enh_para}, sup_para={self.sup_para}")

        # Get language model layers
        lm = model.language_model
        if hasattr(lm, 'model'):
            layers = lm.model.layers
        elif hasattr(lm, 'layers'):
            layers = lm.layers
        else:
            raise AttributeError("Cannot find model layers")

        # Patch each target layer
        for layer_idx in range(self.layer_start, self.layer_end + 1):
            if layer_idx >= len(layers):
                print(f"[VAF] Warning: layer {layer_idx} >= num_layers {len(layers)}")
                continue

            original_attn = layers[layer_idx].self_attn
            self.original_attn_modules[layer_idx] = original_attn

            # Create VAF adapter
            vaf_adapter = VAFAttentionAdapter(
                original_attn, layer_idx,
                self.enh_para, self.sup_para,
                self.sys_len, self.img_len
            )

            # Replace with adapter
            layers[layer_idx].self_attn = vaf_adapter
            self.patched_layers[layer_idx] = vaf_adapter

        print(f"[VAF] Patched {len(self.patched_layers)} layers")

    def unpatch_model(self, model: Any) -> None:
        """Restore original attention modules."""
        lm = model.language_model
        if hasattr(lm, 'model'):
            layers = lm.model.layers
        elif hasattr(lm, 'layers'):
            layers = lm.layers
        else:
            return

        for layer_idx, original_attn in self.original_attn_modules.items():
            if layer_idx < len(layers):
                layers[layer_idx].self_attn = original_attn

        self.patched_layers.clear()
        self.original_attn_modules.clear()
        print("[VAF] Unpatched all layers")

    def update_token_lengths(self, img_start: int, img_end: int) -> None:
        """Update system and image token lengths."""
        self.sys_len = max(0, img_start)
        self.img_len = img_end - img_start + 1

        # Update all patched adapters
        for adapter in self.patched_layers.values():
            adapter.sys_len = self.sys_len
            adapter.img_len = self.img_len


class VAFAttentionAdapter(nn.Module):
    """VAF Attention Adapter for LLaVA models.

    Applies multiplicative scaling to attention logits before softmax:
    - Boost attention TO image tokens
    - Suppress attention TO system tokens
    """

    def __init__(self, original_attn, layer_idx, enh_para, sup_para, sys_len, img_len):
        super().__init__()
        self.layer_idx = layer_idx
        self.original_attn = original_attn

        # Copy configuration
        self.config = original_attn.config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Share projections with original
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        self.scaling = self.head_dim ** -0.5

        # VAF parameters (mutable for token length updates)
        self.enh_para = enh_para
        self.sup_para = sup_para
        self.sys_len = sys_len
        self.img_len = img_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with VAF scaling."""

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
            cache_kwargs = {"sin": sin, "cos": cos} if sin is not None else {}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat k/v heads for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores (logits)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Apply VAF: multiplicative scaling on attention logits (before softmax)
        kv_seq_len = key_states.shape[-2]

        # Scale attention TO image keys (boost)
        if self.sys_len + self.img_len <= kv_seq_len:
            attn_weights[:, :, :, self.sys_len:self.sys_len + self.img_len] *= self.enh_para

            # Scale attention TO system keys (suppress)
            if self.sys_len > 0:
                attn_weights[:, :, :, :self.sys_len] *= self.sup_para

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# Global VAF patcher instance
_vaf_patcher = VAFPatcher()


def setup_vaf(
    model: Any,
    alpha: float = 0.15,
    beta: float = 0.1,
    layer_start: int = 10,
    layer_end: int = 15
) -> VAFPatcher:
    """Setup VAF for the given model.

    Args:
        model: The LLaVA model to patch
        alpha: Visual enhancement parameter (default: 0.15)
        beta: System suppression parameter (default: 0.1)
        layer_start: First layer to patch (default: 10)
        layer_end: Last layer to patch (default: 15)

    Returns:
        VAFPatcher instance
    """
    global _vaf_patcher

    # Reset patcher
    _vaf_patcher = VAFPatcher()
    _vaf_patcher.enabled = True
    _vaf_patcher.enh_para = 1.0 + alpha
    _vaf_patcher.sup_para = 1.0 - beta
    _vaf_patcher.layer_start = layer_start
    _vaf_patcher.layer_end = layer_end

    # Patch model
    _vaf_patcher.patch_model(model)

    return _vaf_patcher


def cleanup_vaf(model: Any) -> None:
    """Cleanup VAF patches."""
    global _vaf_patcher
    _vaf_patcher.unpatch_model(model)
    _vaf_patcher.enabled = False


def update_token_lengths(img_start: int, img_end: int) -> None:
    """Update token lengths for VAF."""
    _vaf_patcher.update_token_lengths(img_start, img_end)


if __name__ == "__main__":
    print("VAF (Visual Amplification Fusion) Module")
    print("ClearSight parameters: alpha=0.15, beta=0.1, layers=10-15")
    print("Import this module and call setup_vaf(model) to use")
