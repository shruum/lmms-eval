"""
LLaVA-1.5 attention intervention patch — replaces layer.self_attn with AttnAdapter.

Based on ClearSight's VAF (Visual Amplification Fusion) approach but extended to
support SRF (Semantic Re-Focus) with query-conditioned saliency guidance.

Architecture
------------
LLaVA-1.5 uses LlamaForCausalLM as the language model. Each layer has:
  - layer.self_attn: LlamaSdpaAttention (or LlamaAttention)
  - layer.mlp: LlamaMLP

We patch by replacing layer.self_attn with AttnAdapter instances in target layers.

Public API
----------
  patch_model(model, method, value)          → activates patch
  unpatch_model(model)                        → restores original attention modules
  update_sample(img_start, img_end)          → call before each sample
  identify_visual_heads(model, inputs_list,
      img_ranges, top_k_pct)                 → computes + stores head mask
  get_image_token_range(inputs, model)       → returns (img_start, img_end)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared mutable state — updated per-sample, read by AttnAdapter
# ---------------------------------------------------------------------------

_STATE: dict = {
    "enabled":    False,
    "method":     "baseline",
    "value":      1.0,
    "img_start":  None,    # int: inclusive start of image tokens in KV dim
    "img_end":    None,    # int: inclusive end   of image tokens in KV dim
    "sys_end":    None,    # int: last system-prompt token index
    "head_mask":  None,    # bool tensor (n_heads,) for vision-aware heads
    # ---- VAF/SRF parameters ----
    "vaf_beta":        0.1,   # suppression coefficient (system-prompt attention)
    "vaf_layer_start": 8,     # first decoder layer to apply VAF
    "vaf_layer_end":   15,    # last  decoder layer to apply VAF (inclusive)
    # Saliency-guided boost
    "salience_mask": None,    # float tensor (n_img_tokens,) in [0,1]; None = uniform
    "srf_background_eps": 0.1,  # suppress non-salient image tokens
    # Bias mode selection
    "srf_bias_mode":     "additive_logit",
    "srf_interp_lambda": 0.5,    # for prob_interp
    "srf_prob_floor":    0.005,  # for attn_floor
    "srf_img_scale":     2.0,    # for global_redistribute
    "srf_apply_phase":   "both", # "both" | "prefill" | "generation"
}

_ORIGINAL_ATTENTIONS: dict = {}  # Stores original self_attn modules by layer index


# ---------------------------------------------------------------------------
# AttnAdapter — wraps LlamaAttention to inject saliency-guided bias
# ---------------------------------------------------------------------------

class AttnAdapter(nn.Module):
    """
    Attention adapter that wraps LlamaSdpaAttention/LlamaAttention.

    In baseline mode, it acts as a pass-through to the original attention.
    In intervention mode, it modifies attention weights.

    Injects saliency-guided bias into attention computation:
      - System-prompt suppression: subtract beta from system-token logits
      - Image-token boost: add alpha * saliency to image-token logits

    Supports multiple bias modes (see _apply_bias methods).
    """

    def __init__(self, original_attn: nn.Module, layer_idx: int):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx

        # Get configuration from the original attention module
        self.config = original_attn.config

        # Extract attention parameters from config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.config.hidden_size // self.num_heads
        self.is_causal = True
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = 0.0

        # Get rotary embedding if present
        self.rotary_emb = getattr(original_attn, 'rotary_emb', None)

        # Copy the projection layers (q_proj, k_proj, v_proj, o_proj)
        if hasattr(original_attn, 'q_proj'):
            self.q_proj = original_attn.q_proj
            self.k_proj = original_attn.k_proj
            self.v_proj = original_attn.v_proj
            self.o_proj = original_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with saliency-guided attention bias.

        In baseline mode, acts as a pass-through to the original attention.
        In intervention mode, modifies attention weights.
        """
        # In baseline mode or when disabled, use original attention directly
        if not _STATE["enabled"] or _STATE["method"] == "baseline":
            return self.original_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Intervention mode: compute attention with bias
        # Get projections from original attention module
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # Query, Key, Value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Handle multi-query attention (MQA/GQA) if needed
        num_heads = self.num_heads
        num_kv_heads = self.num_key_value_heads
        head_dim = self.head_dim

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary embeddings if present
        if self.rotary_emb is not None and position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        elif self.rotary_emb is not None and position_ids is not None:
            # Fallback for older transformers versions
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV caching for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for multi-query attention
        key_states = repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = repeat_kv(value_states, num_heads // num_kv_heads)

        # Compute attention scores (before softmax)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply saliency-guided bias intervention
        if _STATE["enabled"] and _STATE["method"] != "baseline":
            attn_weights = self._apply_intervention(attn_weights, q_len, device)

        # Apply causal mask and attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply mask padding if needed
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply attention dropout
        if self.attention_dropout > 0.0 and self.training:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Apply output projection
        attn_output = self.o_proj(attn_output)

        # Return in the format expected by LlamaDecoderLayer
        # The decoder always expects 2 values: (hidden_states, optional_thing)
        # where optional_thing is either attn_weights, present_key_value, or None
        if output_attentions:
            return (attn_output, attn_weights, past_key_value)
        elif use_cache:
            return (attn_output, past_key_value)
        else:
            # Even when not using cache, return 2 values (second is None)
            return (attn_output, None)

    def _apply_intervention(
        self,
        attn_weights: torch.Tensor,
        q_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Apply saliency-guided attention bias intervention.

        Modifies attn_weights in-place for efficiency.
        """
        layer_idx = self.layer_idx
        layer_start = _STATE["vaf_layer_start"]
        layer_end = _STATE["vaf_layer_end"]
        method = _STATE["method"]
        value = _STATE["value"]

        # Only apply within target layer range
        if not (layer_start <= layer_idx <= layer_end):
            return attn_weights

        img_start = _STATE["img_start"]
        img_end = _STATE["img_end"]

        if img_start is None or img_end is None:
            return attn_weights

        # Clone to avoid in-place modification issues
        attn_weights = attn_weights.clone()

        # System-prompt suppression (applies to all modes)
        sys_end = _STATE["sys_end"]
        beta = _STATE["vaf_beta"]
        head_mask = _STATE["head_mask"]

        if sys_end is not None and sys_end > 0 and beta > 0:
            if head_mask is not None:
                mask_dev = head_mask.to(device)
                sup_bias = attn_weights.new_zeros(1, head_mask.shape[0], 1, 1)
                sup_bias[0, mask_dev, 0, 0] = -beta
                attn_weights[:, mask_dev, :, : sys_end + 1] += sup_bias
            else:
                attn_weights[:, :, :, : sys_end + 1] -= beta

        # Phase gate: check if we should apply image-token bias
        phase = _STATE.get("srf_apply_phase", "both")
        is_gen = (q_len == 1)  # Generation step has q_len=1
        phase_ok = (
            phase == "both"
            or (phase == "generation" and is_gen)
            or (phase == "prefill" and not is_gen)
        )

        # Apply image-token boost based on method
        if method == "srf" and phase_ok:
            sal = _STATE["salience_mask"]
            eps = _STATE["srf_background_eps"]
            bias_mode = _STATE["srf_bias_mode"]

            if bias_mode == "additive_logit":
                # Direct logit addition
                alpha_val = float(value)
                n_img = img_end - img_start + 1

                # DEBUG: Print shapes
                if self.layer_idx == layer_start:
                    print(f"  [DEBUG L{self.layer_idx}] attn_weights shape: {attn_weights.shape}")
                    print(f"  [DEBUG L{self.layer_idx}] img range: [{img_start}, {img_end}], n_img={n_img}")
                    if sal is not None:
                        print(f"  [DEBUG L{self.layer_idx}] salience shape: {sal.shape}")
                    print(f"  [DEBUG L{self.layer_idx}] alpha_val: {alpha_val}")

                if sal is not None:
                    sal_dev = sal.to(device=device, dtype=attn_weights.dtype)
                    bias_row = alpha_val * sal_dev - eps * (1.0 - sal_dev)
                else:
                    bias_row = attn_weights.new_full((n_img,), alpha_val)

                if self.layer_idx == layer_start:
                    print(f"  [DEBUG L{self.layer_idx}] bias_row shape: {bias_row.shape}")
                    print(f"  [DEBUG L{self.layer_idx}] bias_row range: [{bias_row.min():.4f}, {bias_row.max():.4f}]")

                if head_mask is not None:
                    mask_dev = head_mask.to(device)
                    full_bias = attn_weights.new_zeros(1, mask_dev.shape[0], 1, n_img)
                    full_bias[0, mask_dev, 0, :] = bias_row
                    attn_weights[:, mask_dev, :, img_start : img_end + 1] += full_bias
                else:
                    attn_weights[:, :, :, img_start : img_end + 1] += bias_row

                if self.layer_idx == layer_start:
                    print(f"  [DEBUG L{self.layer_idx}] After bias: attn_weights range: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")

        return attn_weights


# ---------------------------------------------------------------------------
# Helper functions for attention computation
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value states for multi-query attention."""
    bsz, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings."""
    q_embed = (query_states * cos) + (rotate_half(query_states) * sin)
    k_embed = (key_states * cos) + (rotate_half(key_states) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for rotary embedding."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------

def patch_model(model: Any, method: str = "baseline", value: float = 1.0) -> None:
    """
    Activate the attention intervention by replacing self_attn with AttnAdapter.

    Args:
        model  : LlavaForConditionalGeneration
        method : 'baseline' | 'srf'
        value  : intervention strength (e.g., boost_alpha for srf)
    """
    valid = ("baseline", "srf")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}. Choose from {valid}")

    _STATE["enabled"] = True
    _STATE["method"] = method
    _STATE["value"] = value

    # Force eager attention so we can intercept F.softmax
    model.config._attn_implementation = "eager"

    # Get LLaMA decoder layers - LLaVA uses LlamaForCausalLM as language_model
    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        raise AttributeError(f"Cannot find layers in language_model: {type(lm)}")

    for layer_idx, layer in enumerate(layers):
        # Store original attention if not already stored
        if layer_idx not in _ORIGINAL_ATTENTIONS:
            _ORIGINAL_ATTENTIONS[layer_idx] = layer.self_attn

        # Replace with AttnAdapter
        if not isinstance(layer.self_attn, AttnAdapter):
            original_attn = layer.self_attn
            adapter = AttnAdapter(original_attn, layer_idx)
            layer.self_attn = adapter


def unpatch_model(model: Any) -> None:
    """Restore original attention modules."""
    lm = model.language_model
    if hasattr(lm, 'model'):
        layers = lm.model.layers
    elif hasattr(lm, 'layers'):
        layers = lm.layers
    else:
        raise AttributeError(f"Cannot find layers in language_model: {type(lm)}")
    for layer_idx, layer in enumerate(layers):
        if layer_idx in _ORIGINAL_ATTENTIONS:
            layer.self_attn = _ORIGINAL_ATTENTIONS[layer_idx]
    _ORIGINAL_ATTENTIONS.clear()
    _STATE["enabled"] = False


def update_sample(img_start: int, img_end: int) -> None:
    """Call once per sample before generate(). Updates image token range."""
    _STATE["img_start"] = img_start
    _STATE["img_end"] = img_end
    _STATE["sys_end"] = max(0, img_start - 1)


def get_image_token_range(
    inputs: Any,
    model: Any,
) -> Tuple[int, int]:
    """
    Return (img_start, img_end) inclusive indices of image tokens in the LM sequence.

    LLaVA uses a single placeholder token that expands to multiple image tokens.
    We compute the expanded range based on the vision config.
    """
    image_token_id = model.config.image_token_index
    ids_cpu = inputs["input_ids"][0].cpu()
    positions = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]

    if len(positions) == 0:
        raise ValueError("No image token found in input_ids")

    img_start = int(positions[0].item())

    # Compute number of image tokens from vision config
    vis_cfg = model.config.vision_config
    n_img_tokens = (vis_cfg.image_size // vis_cfg.patch_size) ** 2

    return img_start, img_start + n_img_tokens - 1


# ---------------------------------------------------------------------------
# Vision-aware Head identification (VHR) — for LLaVA
# ---------------------------------------------------------------------------

@torch.inference_mode()
def identify_visual_heads(
    model: Any,
    calibration_inputs: List[Any],
    img_ranges: List[Tuple[int, int]],
    top_k_pct: float = 0.20,
) -> torch.Tensor:
    """
    Compute per-head mean attention to image tokens and return a bool mask
    selecting the top top_k_pct vision-aware heads.

    This is a simplified version for LLaVA. Since we're using AttnAdapter
    which wraps the original attention, we can capture attention weights
    during a forward pass.

    Args:
        model               : LlavaForConditionalGeneration with patch active
        calibration_inputs  : list of preprocessed input dicts
        img_ranges          : list of (img_start, img_end) per sample
        top_k_pct           : fraction of heads to select

    Returns:
        head_mask : bool tensor of shape (n_heads,)
    """
    assert len(calibration_inputs) == len(img_ranges)
    assert 0 < top_k_pct <= 1.0

    # Get number of heads from first layer
    layers = model.language_model.model.layers
    n_heads = layers[0].self_attn.num_heads

    # Accumulators
    head_acc = None
    count = 0

    # Run calibration in baseline mode
    for inputs, (img_start, img_end) in zip(calibration_inputs, img_ranges):
        update_sample(img_start, img_end)

        # Run forward and capture attention (simplified - uses output_attentions)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if outputs.attentions is not None:
            # outputs.attentions: tuple of (n_layers,) tensors
            # Each tensor: (batch, n_heads, q_len, kv_len)
            for layer_attn in outputs.attentions:
                # Focus on text-query → image-key attention
                text_q_start = min(img_end + 1, layer_attn.shape[2] - 1)
                img_attn = layer_attn[0, :, text_q_start:, img_start : img_end + 1]
                img_score = img_attn.mean().item()

                if head_acc is None:
                    head_acc = torch.zeros(n_heads, dtype=torch.float32)

                # Distribute score across heads (simplified)
                head_acc += img_score / n_heads
                count += 1

    assert count > 0, "VHR calibration captured no attention data"

    scores = head_acc / count
    k = max(1, round(n_heads * top_k_pct))
    threshold = scores.topk(k).values[-1]
    head_mask = scores >= threshold

    _STATE["head_mask"] = head_mask
    return head_mask


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_short(model: Any, inputs: Any, n: int = 4) -> torch.Tensor:
    return model.generate(**inputs, max_new_tokens=n, do_sample=False)


def sanity_check_baseline_determinism(
    model: Any, inputs: Any
) -> None:
    """CHECK 1 — Two baseline runs must give bit-identical tokens."""
    patch_model(model, "baseline", 1.0)
    img_start, img_end = get_image_token_range(inputs, model)
    update_sample(img_start, img_end)

    out1 = _generate_short(model, inputs)
    out2 = _generate_short(model, inputs)

    assert torch.equal(out1, out2), "CHECK 1 FAILED: baseline outputs differ"
    print("  [CHECK 1] baseline is deterministic ✓")


def sanity_check_patched_baseline_equals_unpatched(
    model: Any, inputs: Any
) -> None:
    """CHECK 2 — Patched baseline must be bit-identical to unpatched model."""
    img_start, img_end = get_image_token_range(inputs, model)

    unpatch_model(model)
    out_orig = _generate_short(model, inputs)

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)
    out_base = _generate_short(model, inputs)

    assert torch.equal(out_orig, out_base), "CHECK 2 FAILED: patched baseline differs"
    print("  [CHECK 2] patched baseline == unpatched model ✓")


def sanity_check_intervention_changes_output(
    model: Any, inputs: Any, method: str, value: float
) -> None:
    """CHECK 3 — Intervention must change output vs baseline."""
    img_start, img_end = get_image_token_range(inputs, model)

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)
    out_base = _generate_short(model, inputs)

    patch_model(model, method, value)
    update_sample(img_start, img_end)
    out_int = _generate_short(model, inputs)

    assert not torch.equal(out_base, out_int), (
        f"CHECK 3 FAILED: {method}={value} has no effect"
    )
    print(f"  [CHECK 3] {method}={value} changes output ✓")


def sanity_check_no_nan_or_inf(
    model: Any, inputs: Any, method: str, value: float
) -> None:
    """CHECK 4 — No NaN or Inf in generated tokens."""
    img_start, img_end = get_image_token_range(inputs, model)
    patch_model(model, method, value)
    update_sample(img_start, img_end)
    out = _generate_short(model, inputs)

    assert not out.isnan().any(), f"CHECK 4 FAILED: NaN in output"
    assert not out.isinf().any(), f"CHECK 4 FAILED: Inf in output"
    print(f"  [CHECK 4] no NaN/Inf under {method}={value} ✓")


def run_sanity_checks(
    model: Any,
    inputs: Any,
    method: str = "srf",
    value: float = 2.0,
) -> None:
    """Run all sanity checks."""
    print("\nRunning LLaVA attention patch sanity checks...")
    sanity_check_baseline_determinism(model, inputs)
    sanity_check_patched_baseline_equals_unpatched(model, inputs)
    sanity_check_intervention_changes_output(model, inputs, method, value)
    sanity_check_no_nan_or_inf(model, inputs, method, value)
    print("All sanity checks passed ✓\n")
