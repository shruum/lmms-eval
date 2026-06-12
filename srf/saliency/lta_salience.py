"""
Last-Token Attention (LTA) saliency for image-token boosting.

Key design decisions:
  1. Tap the last decoder layer (or any specified layer) attention weights.
     The last text token's attention distribution over image tokens reflects
     which visual regions the model relies on when about to generate — directly
     causal, unlike HSSA (mid-layer concept alignment) or CLIP (external signal).

  2. Baseline forward: run under patch "baseline" mode to avoid circular
     dependency (SRF modifies attention, which would corrupt the LTA signal).

  3. Head filtering: average over vision-aware heads identified by SRF
     calibration, or over all heads when no head_mask is provided.

  4. output_attentions=True forces eager attention for the forward pass.
     This is accurate but stores all-layer attention matrices. For production
     efficiency, use the hook-based variant (compute_lta_salience_hook).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LTASalienceResult:
    saliency: torch.Tensor      # (n_img_tokens,) float32 in [0, 1]
    mask: torch.Tensor          # (n_img_tokens,) float32 binary top-k
    max_attn: float             # max raw attention weight (before normalisation)
    n_heads_used: int           # number of heads averaged


def compute_lta_salience(
    model,
    inputs,
    img_start: int,
    img_end: int,
    layer_idx: int = -1,
    head_mask: torch.Tensor | None = None,
    top_k_pct: float = 0.30,
) -> LTASalienceResult:
    """
    Compute LTA saliency: attention from the last input token → image tokens.

    Runs a single no_grad forward pass with output_attentions=True to capture
    exact attention weights (including RoPE). Only the target layer's weights
    are used; all others are discarded.

    Args:
        model:      Qwen2.5-VL (or compatible) loaded with attn_implementation="eager".
        inputs:     Tokenised inputs dict on the model's device.
        img_start:  Index of first image token in input_ids.
        img_end:    Index of last  image token in input_ids (inclusive).
        layer_idx:  Decoder layer to read attention from. -1 = last layer.
        head_mask:  Boolean/float tensor (n_heads,) of vision-aware heads.
                    None = average over all heads uniformly.
        top_k_pct:  Fraction of image tokens selected for the binary mask.

    Returns:
        LTASalienceResult with saliency map in [0, 1].
    """
    n_img_tokens = img_end - img_start + 1
    seq_len      = inputs["input_ids"].shape[1]
    last_pos     = seq_len - 1   # last token position (before generation)

    # Resolve decoder layer list (Qwen2.5-VL nests layers under language_model)
    if hasattr(model.model, "layers"):
        _layers = model.model.layers
    elif hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        _layers = model.model.language_model.layers
    else:
        raise AttributeError(
            f"Cannot find decoder layers in model.model. "
            f"Checked: model.model.layers, model.model.language_model.layers. "
            f"Submodules: {[n for n, _ in model.model.named_children()]}"
        )

    n_layers     = len(_layers)
    actual_layer = layer_idx if layer_idx >= 0 else n_layers + layer_idx

    # ── Capture attention from one layer only via a forward hook ──────────────
    # Pre-hook forces output_attentions=True for just the target layer,
    # avoiding the memory cost of storing all-layer attention matrices.
    _attn: list = [None]

    def _pre_hook(module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs

    def _post_hook(module, input, output):
        # Qwen2_5_VLAttention eager mode returns (attn_out, attn_weights, past_kv)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            _attn[0] = output[1].detach().cpu()  # (batch, n_heads, seq_len, seq_len)

    layer = _layers[actual_layer]
    try:
        h_pre  = layer.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        h_post = layer.self_attn.register_forward_hook(_post_hook)
    except TypeError:
        # PyTorch < 2.0: with_kwargs not supported — fall back to full output_attentions
        h_pre  = None
        h_post = layer.self_attn.register_forward_hook(_post_hook)

    try:
        with torch.no_grad():
            if h_pre is not None:
                # Efficient: only one layer computes/returns attention weights
                model(**inputs, output_attentions=False)
            else:
                # Fallback: all layers compute attention; grab what we need
                out     = model(**inputs, output_attentions=True)
                _attn[0] = out.attentions[actual_layer].detach().cpu()
    finally:
        if h_pre is not None:
            h_pre.remove()
        h_post.remove()

    torch.cuda.empty_cache()

    attn_all = _attn[0]   # (1, n_heads, seq_len, seq_len) or None

    if attn_all is None:
        # Hook failed (e.g. flash attention); uniform fallback
        n_heads = model.config.num_attention_heads
        saliency = torch.full((n_img_tokens,), 1.0 / n_img_tokens)
        mask     = torch.ones(n_img_tokens, dtype=torch.float32)
        return LTASalienceResult(saliency=saliency, mask=mask,
                                 max_attn=float(saliency.max()), n_heads_used=0)

    # ── Extract last-position → image-token attention ─────────────────────────
    # attn_all: (1, n_heads, seq_len, seq_len)
    # attn_last_to_img: (n_heads, n_img_tokens)
    attn_last = attn_all[0, :, last_pos, img_start:img_end + 1].float()

    # ── Head selection ────────────────────────────────────────────────────────
    if head_mask is not None:
        sel = head_mask.bool().cpu()
        if sel.shape[0] == attn_last.shape[0] and sel.any():
            attn_last    = attn_last[sel]
            n_heads_used = int(sel.sum().item())
        else:
            n_heads_used = attn_last.shape[0]
    else:
        n_heads_used = attn_last.shape[0]

    # ── Aggregate + normalise ─────────────────────────────────────────────────
    saliency_raw = attn_last.mean(dim=0)     # (n_img_tokens,)
    max_attn     = float(saliency_raw.max())

    s_min, s_max = saliency_raw.min(), saliency_raw.max()
    saliency = (saliency_raw - s_min) / (s_max - s_min + 1e-8)

    # ── Binary top-k mask ─────────────────────────────────────────────────────
    k_val    = max(1, round(n_img_tokens * top_k_pct))
    topk_idx = saliency.topk(k_val).indices
    mask     = torch.zeros(n_img_tokens, dtype=torch.float32)
    mask[topk_idx] = 1.0

    return LTASalienceResult(saliency=saliency, mask=mask,
                             max_attn=max_attn, n_heads_used=n_heads_used)
