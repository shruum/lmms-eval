"""
ILVAD — Inter-Layer Visual Attention Discrepancy for Qwen2.5-VL.

Faithful adaptation of ILVAD (arXiv:2605.20965)
https://github.com/ytx-ML/ILVAD

Algorithm (per sample):
  Phase 1 — build enhanced_map (one forward pass, output_attentions=True):
    1. For each layer, select top-50% heads by mean attention to image tokens
    2. Over first T=10 text-query positions, accumulate attention to image tokens
    3. Binarize per image-token: count positions where mean attention > attn_thresh
    4. Binary count ∈ [0, T]; persistent if count ≥ τ (=5 out of 10)
    5. Inter-layer positive discrepancy: pos_diff[l] = max(0, bin[l+1] − bin[l])
    6. Normalize globally; enhanced_map[l] = exp(α * norm_diff[l]) per image token

  Phase 2 — inference (hooks active):
    For each attention softmax call at layer l:
         attn_weights[..., img_tokens] *= enhanced_map[l]
         attn_weights[..., text_tokens] *= (1 + β)   [text grounding, β=1.0]
         renormalize rows to sum to 1

Design: ILVAD does NOT modify qwen_attn_patch.py (core SRF code is untouched).
Instead, it installs a CHAINING softmax wrapper that runs AFTER qwen_attn_patch's
wrapper, reading layer/decoder context from patch._STATE (read-only).

Hyperparameters (from ILVAD repo/paper):
  T           = 10    (prefill positions to aggregate)
  attn_thresh = 0.01  (attention binarization threshold)
  τ (tau)     = 5     (persistence count threshold out of T)
  α (alpha)   = 5.0   (exponential sharpening)
  β (beta)    = 1.0   (text grounding scale)

Interface matches srf.py / vaf.py so eval.py can dispatch with --method ilvad.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import qwen_attn_patch as patch

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

_T:           int   = 10
_TAU:         float = 5.0
_ALPHA:       float = 5.0
_BETA:        float = 1.0
_ATTN_THRESH: float = 0.01

# ---------------------------------------------------------------------------
# ILVAD-local state (separate from patch._STATE — never modify SRF internals)
# ---------------------------------------------------------------------------

_ILVAD: dict = {
    "active":        False,         # True during inference phase
    "enhanced_map":  None,          # dict {layer_idx: (n_img,) CPU float32} or None
    "beta":          _BETA,
    "T":             _T,
    "tau":           _TAU,
    "alpha":         _ALPHA,
    "attn_thresh":   _ATTN_THRESH,
}

# Chaining: we wrap whatever is currently in F.softmax (may be patch's wrapper)
_ILVAD_ORIG_SOFTMAX = None   # set once in setup()


# ---------------------------------------------------------------------------
# Chaining softmax wrapper
# ---------------------------------------------------------------------------

def _ilvad_softmax(input, dim=-1, dtype=None, **kwargs):
    """
    Runs the underlying softmax (qwen_attn_patch's patched version or original),
    then applies ILVAD post-softmax attention scaling if active.
    """
    # Step 1: delegate to the underlying softmax (preserves SRF / VAF modifications)
    result = _ILVAD_ORIG_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

    # Step 2: apply ILVAD enhanced_map if this is a 4-D attention tensor inside decoder
    if (
        _ILVAD["active"]
        and patch._STATE.get("in_decoder", False)
        and result.dim() == 4
    ):
        s2   = patch._STATE.get("img_start")
        e2   = patch._STATE.get("img_end")
        cl   = patch._STATE.get("current_layer", -1)
        emap = _ILVAD["enhanced_map"]

        if emap is not None and s2 is not None and e2 is not None and cl >= 0:
            layer_map = emap.get(cl)
            if layer_map is not None:
                result   = result.clone()
                lmap_dev = layer_map.to(result.device, dtype=result.dtype)

                # Scale image-token attention by enhanced_map
                result[..., s2 : e2 + 1] = result[..., s2 : e2 + 1] * lmap_dev

                # Text grounding: scale post-image text attention by (1 + β)
                beta = float(_ILVAD["beta"])
                if beta > 0.0 and e2 + 1 < result.shape[-1]:
                    result[..., e2 + 1 :] = result[..., e2 + 1 :] * (1.0 + beta)

                # Renormalize rows to sum to 1
                total  = result.sum(-1, keepdim=True).clamp(min=1e-8)
                result = result / total

    return result


# ---------------------------------------------------------------------------
# Architecture helper
# ---------------------------------------------------------------------------

def _get_decoder_layers(model):
    lm = model.language_model
    if hasattr(lm, "layers"):
        return lm.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(lm).__name__}")


# ---------------------------------------------------------------------------
# Phase 1 — build enhanced_map
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _build_enhanced_map(model, inp: dict, img_start: int, img_end: int) -> dict:
    """
    One forward pass with output_attentions=True (SRF in baseline mode),
    then compute per-layer enhanced_map from inter-layer attention discrepancy.

    Returns:
        enhanced_map: {layer_idx: (n_img_tokens,) float32 tensor on CPU}
    """
    # Temporarily deactivate ILVAD scaling during Phase 1 collection
    _ILVAD["active"] = False

    # Run baseline forward with attention weights returned
    orig_method = patch._STATE["method"]
    patch._STATE["method"] = "baseline"
    out = model(**inp, output_attentions=True)
    patch._STATE["method"] = orig_method

    attentions = out.attentions   # tuple of (bsz, n_heads, q_len, kv_len) per layer
    if attentions is None:
        return {}

    n_layers    = len(attentions)
    n_img       = img_end - img_start + 1
    T           = _ILVAD["T"]
    alpha       = _ILVAD["alpha"]
    attn_thresh = _ILVAD["attn_thresh"]

    # For each layer: compute binary count of persistent image attention
    text_q_start = img_end + 1
    bin_counts: list = []

    for layer_idx, attn in enumerate(attentions):
        # attn: (1, n_heads, q_len, kv_len)
        attn_cpu = attn[0].float().cpu()   # (n_heads, q_len, kv_len)

        # Select top-50% heads by mean attention to image tokens
        img_attn_all = attn_cpu[:, :, img_start : img_end + 1]  # (n_heads, q_len, n_img)
        head_scores  = img_attn_all.mean(dim=(1, 2))             # (n_heads,)
        top_heads    = (head_scores >= head_scores.median()).nonzero().flatten()

        # Take the first T text-query positions attending to image tokens
        q_len = attn_cpu.shape[1]
        if text_q_start < q_len:
            end_q = min(text_q_start + T, q_len)
            text_attn = attn_cpu[top_heads][:, text_q_start:end_q, img_start : img_end + 1]
            # text_attn: (n_top_heads, T_actual, n_img)
        else:
            # Fallback: all positions
            end_q = min(T, q_len)
            text_attn = attn_cpu[top_heads][:, :end_q, img_start : img_end + 1]

        if text_attn.numel() == 0:
            bin_counts.append(torch.zeros(n_img))
            continue

        # Mean over selected heads → (T_actual, n_img)
        mean_attn = text_attn.mean(0)

        # Binary count: how many of the T positions exceed attn_thresh per image token
        bin_count = (mean_attn > attn_thresh).float().sum(0)   # (n_img,)
        bin_counts.append(bin_count)

    if n_layers < 2:
        return {}

    # Inter-layer positive discrepancy + normalization + exponentiation
    pos_diffs: list = []
    for l in range(n_layers):
        if l + 1 < n_layers:
            diff = torch.clamp(bin_counts[l + 1] - bin_counts[l], min=0.0)
        else:
            diff = torch.zeros(n_img)
        pos_diffs.append(diff)

    # Global max for normalization
    all_diffs  = torch.stack(pos_diffs)   # (n_layers, n_img)
    global_max = all_diffs.max().item()
    eps        = 1e-8

    enhanced_map: dict = {}
    for l in range(n_layers):
        norm_diff        = pos_diffs[l] / (global_max + eps)
        enhanced_map[l]  = torch.exp(torch.as_tensor(alpha) * norm_diff)

    return enhanced_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup(model, processor, calib_dataset: str = "pope") -> None:
    """
    Install qwen_attn_patch baseline patch and chain ILVAD's softmax wrapper on top.
    The chain is: F.softmax → _ilvad_softmax → (qwen_attn_patch's softmax) → original.
    """
    global _ILVAD_ORIG_SOFTMAX

    # First ensure qwen_attn_patch is active (provides in_decoder / current_layer tracking)
    patch.patch_model(model, method="baseline", value=1.0)

    # Chain our wrapper AFTER qwen_attn_patch has replaced F.softmax
    if _ILVAD_ORIG_SOFTMAX is None:
        _ILVAD_ORIG_SOFTMAX = F.softmax
        torch.nn.functional.softmax = _ilvad_softmax

    _ILVAD["active"] = False
    print(f"  [ILVAD] chaining softmax hook installed")
    print(f"  [ILVAD] T={_ILVAD['T']}  tau={_ILVAD['tau']}  "
          f"alpha={_ILVAD['alpha']}  beta={_ILVAD['beta']}")


def reset_for_dataset(
    dataset: str = "pope",
    *,
    ilvad_T:     int   | None = None,
    ilvad_tau:   float | None = None,
    ilvad_alpha: float | None = None,
    ilvad_beta:  float | None = None,
    **kwargs,
) -> None:
    """Reset per-dataset state (ILVAD has no dataset-specific tuning needed)."""
    if ilvad_T     is not None: _ILVAD["T"]     = ilvad_T
    if ilvad_tau   is not None: _ILVAD["tau"]   = ilvad_tau
    if ilvad_alpha is not None: _ILVAD["alpha"] = ilvad_alpha
    if ilvad_beta  is not None: _ILVAD["beta"]  = ilvad_beta

    _ILVAD["active"]        = False
    _ILVAD["enhanced_map"]  = None
    patch._STATE["method"]  = "baseline"


def prepare_sample(
    inp,
    img_start: int,
    img_end: int,
    image,
    question: str,
    model,
    processor,
    **kwargs,
) -> None:
    """
    Phase 1: build enhanced_map from inter-layer attention discrepancy.
    Stores result in _ILVAD["enhanced_map"] and arms the softmax hook.
    """
    # Build enhanced_map with ILVAD hook temporarily inactive
    emap = _build_enhanced_map(model, inp, img_start, img_end)

    # Store image range in patch._STATE so the hook can read it.
    # Convention: img_end is the INCLUSIVE last image-token index.
    # The hook uses [img_start : img_end + 1] to select all image tokens.
    patch._STATE["img_start"] = img_start
    patch._STATE["img_end"]   = img_end        # inclusive end (no -1)
    patch._STATE["method"]    = "baseline"     # SRF stays off; ILVAD hook handles it

    # Arm ILVAD for Phase 2
    _ILVAD["enhanced_map"] = emap
    _ILVAD["active"]       = True


def cleanup() -> None:
    """Disarm ILVAD hook and clear per-sample state."""
    _ILVAD["active"]       = False
    _ILVAD["enhanced_map"] = None
    patch._STATE["method"] = "baseline"


# ---------------------------------------------------------------------------
# Inference helpers — called by eval.py via method_get_logits / method_generate
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, **kwargs) -> torch.Tensor:
    """Single forward pass with ILVAD chaining hook active."""
    # patch method stays "baseline" — ILVAD hook fires independently
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def generate_contrastive(
    model,
    inp: dict,
    processor,
    max_new_tokens: int = 20,
    content_offset: int = 0,
    **kwargs,
) -> list[int]:
    """Standard greedy generation with ILVAD chaining hook active."""
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return out_ids[0, inp["input_ids"].shape[1]:].tolist()
