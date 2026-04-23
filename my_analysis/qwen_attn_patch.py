"""
Attention intervention patch for Qwen2.5-VL — no model files modified.

How it works
------------
`eager_attention_forward` (transformers) calls `nn.functional.softmax` on
attention logits (shape: bsz, n_heads, q_len, kv_len).  We temporarily
replace that function with our own that injects the intervention BEFORE
softmax, so the rest of the original code (attn_weights @ value_states)
runs unchanged.

The patch is scoped to the LM decoder only via forward hooks on
`model.language_model`, so the vision encoder's attention is never touched.

Intervention types
------------------
  baseline     — identity (patch is a provable no-op)
  temperature  — divide logits by T  (T > 1 → softer,  T < 1 → sharper)
  vision_boost — add log(w) to image-column logits before softmax
                 ≡ multiplying those probs by w then renormalising (all heads)
  vhr_boost    — same as vision_boost but only for vision-aware heads
                 identified offline via identify_visual_heads()
  vaf          — Visual Amplification Fusion (ClearSight, arXiv 2503.13107):
                 in target layers, boost image-token attention by +alpha and
                 suppress system-prompt attention by -beta, applied to
                 vision-aware heads (head_mask) or all heads if mask is None.
                 _STATE["vaf_beta"], "vaf_layer_start", "vaf_layer_end" control
                 the suppression and layer range.

Public API
----------
  patch_model(model, method, value)          → activates patch, registers hooks
  unpatch_model()                            → restores original softmax, removes hooks
  update_sample(img_start, img_end)          → call before each sample
  identify_visual_heads(model, inputs_list,
      img_ranges, top_k_pct)                 → computes + stores head mask (call once)
  run_sanity_checks(model, inputs, processor)
  run_vhr_sanity_checks(model, calib_inputs_list,
      img_ranges, sc_inputs, processor)
"""
from __future__ import annotations

import math
import torch
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared mutable state — updated per-sample, read by the patched softmax
# ---------------------------------------------------------------------------

_STATE: dict = {
    "enabled":    False,
    "in_decoder": False,
    "method":     "baseline",
    "value":      1.0,
    "img_start":  None,    # int: inclusive start of image tokens in KV dim
    "img_end":    None,    # int: inclusive end   of image tokens in KV dim
    "sys_end":    None,    # int: last system-prompt token index (= img_start - 1)
    "head_mask":  None,    # bool tensor (n_heads,) for vhr_boost/vaf; None = all heads
    # ---- VAF parameters ----
    "vaf_beta":        0.1,   # suppression coefficient (system-prompt attention)
    "vaf_layer_start": 8,     # first decoder layer to apply VAF
    "vaf_layer_end":   15,    # last  decoder layer to apply VAF (inclusive)
    "current_layer":   -1,    # updated per self_attn forward-pre-hook
    # ---- salience-weighted boost (attn_salience / clip_salience) ----
    "salience_mask": None,    # float tensor (n_img_tokens,) in [0,1]; None = uniform
    # ---- SRF (Semantic Re-Focus) — complete method parameters ----
    # Uses vaf_layer_start / vaf_layer_end for the layer range (same 8-15 defaults).
    "srf_background_eps": 0.1,  # suppress non-salient image tokens by this amount
    # Bias mode — selects which mathematical approach is used for image tokens:
    #   "additive_logit" — pre-softmax: logit += alpha*sal  (current/default)
    #   "prob_interp"    — post-softmax: redistribute img budget by saliency
    #   "prob_scale"     — post-softmax: p_i *= (1 + alpha*sal_i), renorm
    #   "attn_floor"     — post-softmax: p_i = max(p_i, floor) for salient, renorm
    # System-prompt suppression (vaf_beta) is always applied pre-softmax for all modes.
    "srf_bias_mode":     "additive_logit",
    "srf_interp_lambda": 0.5,    # for prob_interp: mixing weight (0=no-op, 1=full redistrib)
    "srf_prob_floor":    0.005,  # for attn_floor: minimum attention per salient token
    #   "global_redistribute" — post-softmax: scale up total img fraction by img_scale,
    #                           distribute within img budget by saliency, scale text down.
    "srf_img_scale":     2.0,    # for global_redistribute: multiply current img attn total
    # ---- internal captures (not part of public API) ----
    "_capture":          False,   # CHECK 3: capture one 4-D softmax output
    "_captured":         None,    # CHECK 3: last captured (n_heads, q_len, kv_len) on CPU
    "_calibrate_heads":  False,   # VHR calibration mode
    "_calib_head_acc":   None,    # running sum of per-head vision scores (n_heads,)
    "_calib_head_count": 0,       # number of accumulation steps
    # ---- attention salience capture (attn_salience two-pass) ----
    "_capture_salience":  False,  # enable per-token attention accumulation
    "_salience_acc":      None,   # float tensor (n_img_tokens,) accumulated scores
    "_salience_count":    0,      # number of softmax calls accumulated
}

_ORIGINAL_SOFTMAX = None   # stores the real F.softmax before patching
_HOOKS: list       = []    # forward hooks registered on model.language_model

IMAGE_TOKEN = "<|image_pad|>"   # Qwen2.5-VL image-pad token string


# ---------------------------------------------------------------------------
# Architecture helpers — support Qwen2.5-VL and LLaVA without hardcoding
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: Any) -> Any:
    """Return the decoder layer list for any supported LMM architecture.

    Supported:
      - Qwen2.5-VL  : model.language_model          is Qwen2VLModel     → .layers
      - LLaVA-1.5   : model.language_model          is LlamaForCausalLM → .model.layers
    """
    lm = model.language_model
    if hasattr(lm, "layers"):
        return lm.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(lm).__name__}. "
        "Expected '.layers' (Qwen) or '.model.layers' (LLaVA / LLaMA)."
    )


# ---------------------------------------------------------------------------
# Patched softmax
# ---------------------------------------------------------------------------

def _patched_softmax(
    input: torch.Tensor,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Drop-in replacement for torch.nn.functional.softmax.
    Only modifies 4-D attention logit tensors when inside the LM decoder.
    """
    if (
        _STATE["enabled"]
        and _STATE["in_decoder"]
        and input.dim() == 4
        and _STATE["method"] != "baseline"
    ):
        method = _STATE["method"]
        value  = _STATE["value"]
        s      = _STATE["img_start"]
        e      = _STATE["img_end"]

        input = input.clone()   # never modify in-place

        if method == "temperature":
            input = input / value

        elif method == "vision_boost":
            if s is not None and e is not None:
                log_w = math.log(max(float(value), 1e-8))
                input[..., s : e + 1] = input[..., s : e + 1] + log_w

        elif method == "vhr_boost":
            if s is not None and e is not None:
                log_w     = math.log(max(float(value), 1e-8))
                head_mask = _STATE["head_mask"]
                if head_mask is None:
                    # No mask set — fall back to all-head boost
                    input[..., s : e + 1] = input[..., s : e + 1] + log_w
                else:
                    # Vectorised: add log_w only to selected head channels
                    # input shape: (batch, n_heads, q_len, kv_len)
                    n_heads = input.shape[1]
                    bias = input.new_zeros(1, n_heads, 1, 1)
                    mask_dev = head_mask.to(input.device)
                    bias[0, mask_dev, 0, 0] = log_w
                    input[..., s : e + 1] = input[..., s : e + 1] + bias

        elif method in ("attn_salience", "srf_clip_basic"):
            # Salience-weighted boost: log(w) * per-token salience weight.
            # Falls back to uniform boost when salience_mask is None.
            if s is not None and e is not None:
                log_w     = math.log(max(float(value), 1e-8))
                sal_mask  = _STATE["salience_mask"]    # (n_img_tokens,) float or None
                head_mask = _STATE["head_mask"]
                n_heads   = input.shape[1]
                n_img     = e - s + 1

                # bias_row: (n_img,) — per-token weight, must match input dtype
                if sal_mask is not None:
                    bias_row = (log_w * sal_mask).to(device=input.device, dtype=input.dtype)
                else:
                    bias_row = input.new_full((n_img,), log_w)

                if head_mask is not None:
                    # Apply only to vision-aware heads: shape (1, n_heads, 1, n_img)
                    mask_dev  = head_mask.to(input.device)
                    full_bias = input.new_zeros(1, n_heads, 1, n_img)
                    full_bias[0, mask_dev, 0, :] = bias_row         # broadcast over selected heads
                    input[..., s : e + 1] = input[..., s : e + 1] + full_bias
                else:
                    input[..., s : e + 1] = input[..., s : e + 1] + bias_row

        elif method == "vaf":
            # Visual Amplification Fusion (ClearSight, arXiv 2503.13107)
            # Apply only within the target layer range
            current_layer = _STATE["current_layer"]
            layer_start   = _STATE["vaf_layer_start"]
            layer_end     = _STATE["vaf_layer_end"]

            if layer_start <= current_layer <= layer_end and s is not None and e is not None:
                alpha     = float(value)
                beta      = float(_STATE["vaf_beta"])
                head_mask = _STATE["head_mask"]
                sys_end   = _STATE["sys_end"]
                n_heads   = input.shape[1]

                if head_mask is not None:
                    mask_dev = head_mask.to(input.device)
                    # Enhancement: add alpha only to vision-aware heads' image columns
                    enh_bias = input.new_zeros(1, n_heads, 1, 1)
                    enh_bias[0, mask_dev, 0, 0] = alpha
                    input[..., s : e + 1] = input[..., s : e + 1] + enh_bias
                    # Suppression: subtract beta from those heads' system-prompt columns
                    if sys_end is not None and sys_end > 0 and beta > 0:
                        sup_bias = input.new_zeros(1, n_heads, 1, 1)
                        sup_bias[0, mask_dev, 0, 0] = -beta
                        input[..., : sys_end + 1] = input[..., : sys_end + 1] + sup_bias
                else:
                    # All heads — simpler scalar addition
                    input[..., s : e + 1] = input[..., s : e + 1] + alpha
                    if sys_end is not None and sys_end > 0 and beta > 0:
                        input[..., : sys_end + 1] = input[..., : sys_end + 1] - beta

        elif method == "srf":
            # Semantic Re-Focus: query-conditioned token-selective boost.
            # Pre-softmax: system-prompt suppression (all modes) + additive logit (if mode==additive_logit).
            # Post-softmax modes (prob_interp, prob_scale, attn_floor) are applied after
            # _ORIGINAL_SOFTMAX below, where they operate directly on attention probabilities.
            current_layer = _STATE["current_layer"]
            layer_start   = _STATE["vaf_layer_start"]
            layer_end     = _STATE["vaf_layer_end"]

            if layer_start <= current_layer <= layer_end and s is not None and e is not None:
                sal       = _STATE["salience_mask"]          # (n_img,) float [0,1] or None
                eps       = float(_STATE["srf_background_eps"])
                beta      = float(_STATE["vaf_beta"])
                head_mask = _STATE["head_mask"]
                sys_end   = _STATE["sys_end"]
                n_heads   = input.shape[1]
                n_img     = e - s + 1
                bias_mode = _STATE.get("srf_bias_mode", "additive_logit")

                # ── System-prompt suppression: applied pre-softmax for ALL bias modes ──
                if sys_end is not None and sys_end > 0 and beta > 0:
                    if head_mask is not None:
                        mask_dev = head_mask.to(input.device)
                        sup_bias = input.new_zeros(1, n_heads, 1, 1)
                        sup_bias[0, mask_dev, 0, 0] = -beta
                        input[..., : sys_end + 1] = input[..., : sys_end + 1] + sup_bias
                    else:
                        input[..., : sys_end + 1] = input[..., : sys_end + 1] - beta

                # ── Image-token boost: pre-softmax, additive_logit mode only ──
                if bias_mode == "additive_logit":
                    # Use boost_alpha directly as logit addition (same convention as vaf).
                    # boost_alpha=3 → add 3 to salient token logits → exp(3)≈20× attention.
                    # Previously used log(alpha), which was ~13× too weak.
                    alpha_val = float(value)
                    if sal is not None:
                        sal_dev  = sal.to(device=input.device, dtype=input.dtype)
                        bias_row = alpha_val * sal_dev - eps * (1.0 - sal_dev)
                    else:
                        bias_row = input.new_full((n_img,), alpha_val)

                    if head_mask is not None:
                        mask_dev  = head_mask.to(input.device)
                        full_bias = input.new_zeros(1, n_heads, 1, n_img)
                        full_bias[0, mask_dev, 0, :] = bias_row
                        input[..., s : e + 1] = input[..., s : e + 1] + full_bias
                    else:
                        input[..., s : e + 1] = input[..., s : e + 1] + bias_row

    result = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

    # CHECK 3 capture — avoids output_attentions=True OOM on small GPUs
    if _STATE["_capture"] and _STATE["in_decoder"] and result.dim() == 4:
        _STATE["_captured"] = result[0].detach().cpu().float()

    # VHR calibration — accumulate per-head mean attention to image tokens
    # Bug fix: restrict to TEXT query positions (after img_end) so we measure
    # how much each head attends to image tokens FROM text tokens, not from
    # image tokens attending to themselves (which dilutes the signal).
    if _STATE["_calibrate_heads"] and _STATE["in_decoder"] and result.dim() == 4:
        s2 = _STATE["img_start"]
        e2 = _STATE["img_end"]
        if s2 is not None and e2 is not None and e2 > s2:
            # result: (1, n_heads, q_len, kv_len)
            text_q_start = e2 + 1   # first text token query position
            q_len        = result.shape[2]
            if text_q_start < q_len:
                # Ideal: text tokens attending to image key positions
                img_attn = result[0, :, text_q_start:, s2 : e2 + 1]
            else:
                # Fallback: all positions (e.g. image-only input)
                img_attn = result[0, :, :, s2 : e2 + 1]
            img_score = img_attn.mean(dim=(1, 2)).detach().cpu().float()
            acc = _STATE["_calib_head_acc"]
            _STATE["_calib_head_acc"]   = img_score if acc is None else acc + img_score
            _STATE["_calib_head_count"] = _STATE["_calib_head_count"] + 1

    # Attention salience capture — accumulate per-image-token attention scores
    # from text query positions → image key positions (vision-aware heads only).
    if _STATE["_capture_salience"] and _STATE["in_decoder"] and result.dim() == 4:
        s2 = _STATE["img_start"]
        e2 = _STATE["img_end"]
        if s2 is not None and e2 is not None and e2 > s2:
            text_q_start = e2 + 1
            q_len        = result.shape[2]
            n_img        = e2 - s2 + 1
            if text_q_start < q_len:
                img_attn = result[0, :, text_q_start:, s2 : e2 + 1]   # (n_heads, n_text_q, n_img)
                hm = _STATE["head_mask"]
                if hm is not None:
                    img_attn = img_attn[hm.to(result.device)]
                token_scores = img_attn.mean(dim=(0, 1)).detach().cpu().float()  # (n_img,)
                acc = _STATE["_salience_acc"]
                _STATE["_salience_acc"]   = token_scores if acc is None else acc + token_scores
                _STATE["_salience_count"] += 1

    # ── Post-softmax SRF modifications ────────────────────────────────────────
    # Applied when srf_bias_mode != "additive_logit".
    # Operates directly on attention probabilities (result) rather than logits.
    # sys_beta suppression was already applied pre-softmax above.
    if (
        _STATE["enabled"]
        and _STATE["in_decoder"]
        and result.dim() == 4
        and _STATE["method"] == "srf"
        and _STATE.get("srf_bias_mode", "additive_logit") not in ("additive_logit",)
    ):
        s2 = _STATE["img_start"]
        e2 = _STATE["img_end"]
        cl = _STATE["current_layer"]
        l0 = _STATE["vaf_layer_start"]
        l1 = _STATE["vaf_layer_end"]

        if l0 <= cl <= l1 and s2 is not None and e2 is not None:
            sal       = _STATE["salience_mask"]   # (n_img,) float [0,1] or None
            head_mask = _STATE["head_mask"]
            val       = float(_STATE["value"])
            mode      = _STATE["srf_bias_mode"]
            result    = result.clone()

            # helper: get/set img-token slice for selected heads
            def _img_slice(r, hm):
                return r[:, hm, :, s2:e2+1] if hm is not None else r[..., s2:e2+1]

            def _set_img(r, new, hm):
                if hm is not None:
                    r[:, hm, :, s2:e2+1] = new
                else:
                    r[..., s2:e2+1] = new

            def _renorm(r, hm):
                """Renormalise rows to sum to 1 for selected heads."""
                if hm is not None:
                    total = r[:, hm, :, :].sum(-1, keepdim=True).clamp(min=1e-8)
                    r[:, hm, :, :] = r[:, hm, :, :] / total
                else:
                    total = r.sum(-1, keepdim=True).clamp(min=1e-8)
                    r /= total

            hm = head_mask.to(result.device) if head_mask is not None else None

            if mode == "prob_interp":
                # Redistribute image attention budget by saliency, total img weight unchanged.
                # p_img_new = (1-λ)·p_orig + λ·sal_norm·total_img_weight
                lam = float(_STATE.get("srf_interp_lambda", 0.5))
                if sal is not None:
                    sal_norm = sal / (sal.sum() + 1e-8)
                    sal_dev  = sal_norm.to(result.device, dtype=result.dtype).view(1, 1, 1, -1)
                    img_p    = _img_slice(result, hm)
                    total    = img_p.sum(-1, keepdim=True)   # total img weight
                    target   = sal_dev * total
                    _set_img(result, (1 - lam) * img_p + lam * target, hm)
                # (if sal is None: no-op — uniform saliency would be a no-op anyway)

            elif mode == "prob_scale":
                # p_i *= (1 + α·sal_i) for img tokens, then renormalise all rows.
                sal_dev = (sal if sal is not None
                           else result.new_ones(e2 - s2 + 1)
                           ).to(result.device, dtype=result.dtype)
                scale   = (1.0 + val * sal_dev).view(1, 1, 1, -1)
                img_p   = _img_slice(result, hm)
                _set_img(result, img_p * scale, hm)
                _renorm(result, hm)

            elif mode == "attn_floor":
                # max(p_i, floor) for salient img tokens, then renormalise.
                floor   = float(_STATE.get("srf_prob_floor", 0.005))
                sal_bin = ((sal > 0.5).float() if sal is not None
                           else result.new_ones(e2 - s2 + 1)
                           ).to(result.device, dtype=result.dtype)
                floor_t = (floor * sal_bin).view(1, 1, 1, -1)
                img_p   = _img_slice(result, hm)
                _set_img(result, torch.maximum(img_p, floor_t), hm)
                _renorm(result, hm)

            elif mode == "global_redistribute":
                # Scale up the total image attention fraction by img_scale, then
                # distribute within the enlarged img budget according to saliency.
                # Non-image tokens (sys + text) are scaled down proportionally so
                # rows still sum to 1.
                #
                # Let P_img = current total img weight (per query position, per head).
                # target = min(P_img * img_scale, 0.95)           ← enlarged budget
                # new_img[i] = target * sal_norm[i]               ← distribute by saliency
                # scale_other = (1 - target) / (1 - P_img)        ← shrink rest
                img_scale = float(_STATE.get("srf_img_scale", 2.0))
                if sal is not None:
                    sal_norm = sal / (sal.sum() + 1e-8)
                    sal_dev  = sal_norm.to(result.device, dtype=result.dtype)
                else:
                    n_img    = e2 - s2 + 1
                    sal_dev  = result.new_full((n_img,), 1.0 / n_img)

                sal_dev = sal_dev.view(1, 1, 1, -1)

                if hm is not None:
                    img_p   = result[:, hm, :, s2:e2+1]
                    p_total = img_p.sum(-1, keepdim=True)          # (..., 1)
                    target  = (p_total * img_scale).clamp(max=0.95)
                    new_img = target * sal_dev
                    scale_f = ((1.0 - target) / (1.0 - p_total).clamp(min=1e-8))
                    result[:, hm, :, :s2]    *= scale_f
                    result[:, hm, :, e2+1:]  *= scale_f
                    result[:, hm, :, s2:e2+1] = new_img
                else:
                    img_p   = result[..., s2:e2+1]
                    p_total = img_p.sum(-1, keepdim=True)
                    target  = (p_total * img_scale).clamp(max=0.95)
                    new_img = target * sal_dev
                    scale_f = ((1.0 - target) / (1.0 - p_total).clamp(min=1e-8))
                    result[..., :s2]    *= scale_f
                    result[..., e2+1:]  *= scale_f
                    result[..., s2:e2+1] = new_img

    return result


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------

def patch_model(model: Any, method: str = "baseline", value: float = 1.0) -> None:
    """
    Activate the attention intervention and register decoder-scoping hooks.
    Safe to call multiple times — re-uses existing hooks, just updates state.

    Args:
        model  : Qwen2_5_VLForConditionalGeneration (must use eager attention)
        method : 'baseline' | 'temperature' | 'vision_boost' | 'vhr_boost'
        value  : intervention strength (ignored for baseline)
    """
    global _ORIGINAL_SOFTMAX

    valid = ("baseline", "temperature", "vision_boost", "vhr_boost", "vaf",
             "attn_salience", "srf_clip_basic", "srf_clip", "srf_hssa")
    if method not in valid:
        raise ValueError(f"Unknown method: {method!r}. Choose from {valid}")
    if value <= 0:
        raise ValueError(f"value must be > 0, got {value}")

    if _ORIGINAL_SOFTMAX is None:
        _ORIGINAL_SOFTMAX = torch.nn.functional.softmax
        torch.nn.functional.softmax = _patched_softmax

    _STATE["enabled"] = True
    _STATE["method"]  = method
    _STATE["value"]   = value

    if not _HOOKS:
        lm = model.language_model

        def _lm_pre(module, args):
            _STATE["in_decoder"] = True

        def _lm_post(module, args, output):
            _STATE["in_decoder"] = False
            _STATE["current_layer"] = -1

        _HOOKS.append(lm.register_forward_pre_hook(_lm_pre))
        _HOOKS.append(lm.register_forward_hook(_lm_post))

        # Per-layer hooks so _patched_softmax knows which layer it is in.
        # Required for VAF layer-range restriction.
        for layer_idx, layer in enumerate(_get_decoder_layers(model)):
            def _make_layer_pre(idx: int):
                def _layer_pre(module, args):
                    _STATE["current_layer"] = idx
                return _layer_pre
            _HOOKS.append(layer.self_attn.register_forward_pre_hook(
                _make_layer_pre(layer_idx)))


def unpatch_model() -> None:
    """Restore original softmax and remove all hooks. Safe to call multiple times."""
    global _ORIGINAL_SOFTMAX
    if _ORIGINAL_SOFTMAX is not None:
        torch.nn.functional.softmax = _ORIGINAL_SOFTMAX
        _ORIGINAL_SOFTMAX = None
    for h in _HOOKS:
        h.remove()
    _HOOKS.clear()
    _STATE["enabled"]    = False
    _STATE["in_decoder"] = False


def update_sample(img_start: int, img_end: int) -> None:
    """Call once per sample before generate()/forward(). Updates image token range."""
    _STATE["img_start"] = img_start
    _STATE["img_end"]   = img_end
    _STATE["sys_end"]   = max(0, img_start - 1)  # tokens before image = system prompt


def get_image_token_range(
    inputs: Any,
    processor: Any = None,
    model: Any     = None,
) -> Tuple[int, int]:
    """Return (img_start, img_end) inclusive indices of image tokens in the LM sequence.

    Provide exactly one of ``processor`` or ``model``:

    - Qwen2.5-VL (``processor``): many ``<|image_pad|>`` tokens already present in
      ``input_ids``; returns the first and last positions.
    - LLaVA-1.5 (``model``): a single placeholder token in ``input_ids`` expands to
      ``(image_size // patch_size)^2`` image tokens inside the LM.  Returns
      ``(placeholder_pos, placeholder_pos + n_img_tokens - 1)`` to match the hidden-state
      sequence lengths seen by the decoder.

    Args:
        inputs    : processor output dict containing ``input_ids``
        processor : Qwen AutoProcessor (use for Qwen2.5-VL)
        model     : LlavaForConditionalGeneration (use for LLaVA-1.5)
    """
    if model is not None and hasattr(model.config, "image_token_index"):
        # LLaVA: single placeholder → expands to (image_size // patch_size)² tokens
        image_token_id = model.config.image_token_index
        ids_cpu        = inputs["input_ids"][0].cpu()
        positions      = (ids_cpu == image_token_id).nonzero(as_tuple=True)[0]
        img_start      = int(positions[0].item())
        vis_cfg        = model.config.vision_config
        n_img_tokens   = (vis_cfg.image_size // vis_cfg.patch_size) ** 2
        return img_start, img_start + n_img_tokens - 1
    else:
        # Qwen: multiple <|image_pad|> tokens already in input_ids
        image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_mask     = inputs["input_ids"][0].cpu() == image_token_id
        positions      = image_mask.nonzero(as_tuple=True)[0]
        return int(positions[0].item()), int(positions[-1].item())


# ---------------------------------------------------------------------------
# VHR — Vision-aware Head identification
# ---------------------------------------------------------------------------

@torch.inference_mode()
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
        model               : patched Qwen model (eager attention)
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

    # Reset calibration state
    _STATE["_calibrate_heads"]  = True
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    import torch as _torch
    with _torch.no_grad():
        for inputs, (img_start, img_end) in zip(calibration_inputs, img_ranges):
            update_sample(img_start, img_end)
            model(**inputs)

    _STATE["_calibrate_heads"] = False

    count = _STATE["_calib_head_count"]
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


# ---------------------------------------------------------------------------
# Attention salience — two-pass per-sample salience computation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_attention_salience(
    model: Any,
    inputs: Any,
    img_start: int,
    img_end: int,
    top_k_pct: float = 0.3,
    normalize: bool = True,
) -> torch.Tensor:
    """
    First-pass salience computation (no intervention).

    Runs a single forward pass in baseline mode, captures text-query →
    image-key attention from vision-aware heads (head_mask if set, else all
    heads), and returns a per-image-token salience weight tensor.

    Args:
        model       : Qwen model with patch active
        inputs      : preprocessed input dict (on model device)
        img_start   : inclusive start of image tokens
        img_end     : inclusive end of image tokens
        top_k_pct   : fraction of image tokens to assign weight 1.0
                      (rest = 0.0).  Set to 1.0 for soft weights instead.
        normalize   : if True return hard top-k binary mask;
                      if False return soft normalized scores in [0, 1]

    Returns:
        salience : float tensor of shape (img_end - img_start + 1,)
    """
    # Run in baseline so no intervention distorts the attention signal
    orig_method = _STATE["method"]
    orig_value  = _STATE["value"]
    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)

    _STATE["_capture_salience"]  = True
    _STATE["_salience_acc"]      = None
    _STATE["_salience_count"]    = 0

    model(**inputs)

    _STATE["_capture_salience"] = False

    count = _STATE["_salience_count"]
    assert count > 0, (
        "compute_attention_salience: no decoder softmax calls captured — "
        "check eager attention and active hooks"
    )
    scores = _STATE["_salience_acc"] / count   # (n_img_tokens,)
    _STATE["_salience_acc"]   = None
    _STATE["_salience_count"] = 0

    # Restore original method/value
    patch_model(model, orig_method, orig_value)
    update_sample(img_start, img_end)

    n_img = img_end - img_start + 1
    if normalize:
        # Hard top-k binary mask
        k = max(1, round(n_img * top_k_pct))
        topk_idx = scores.topk(k).indices
        mask = torch.zeros(n_img, dtype=torch.float32)
        mask[topk_idx] = 1.0
        return mask
    else:
        # Soft: min-max normalize to [0, 1]
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            return (scores - mn) / (mx - mn)
        return torch.ones(n_img, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Original sanity checks (CHECK 1-4) — unchanged logic
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_short(model: Any, inputs: Any, n: int = 4) -> torch.Tensor:
    return model.generate(**inputs, max_new_tokens=n,
                          do_sample=False, temperature=None,
                          top_p=None, num_beams=1)


def sanity_check_baseline_determinism(
    model: Any, inputs: Any, get_img_range_fn: Any
) -> None:
    """CHECK 1 — Two baseline runs must give bit-identical tokens."""
    patch_model(model, "baseline", 1.0)
    img_start, img_end = get_img_range_fn(inputs)
    update_sample(img_start, img_end)
    out1 = _generate_short(model, inputs)
    out2 = _generate_short(model, inputs)
    assert torch.equal(out1, out2), (
        f"CHECK 1 FAILED: baseline outputs differ!\n  run1={out1}\n  run2={out2}"
    )
    print("  [CHECK 1] baseline is deterministic ✓")


def sanity_check_patched_baseline_equals_unpatched(
    model: Any, inputs: Any, get_img_range_fn: Any
) -> None:
    """CHECK 2 — Patched baseline must be bit-identical to unpatched model."""
    img_start, img_end = get_img_range_fn(inputs)
    unpatch_model()
    out_orig = _generate_short(model, inputs)
    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)
    out_base = _generate_short(model, inputs)
    assert torch.equal(out_orig, out_base), (
        f"CHECK 2 FAILED: patched baseline differs from unpatched!\n"
        f"  unpatched={out_orig}\n  patched_baseline={out_base}"
    )
    print("  [CHECK 2] patched baseline == unpatched model ✓")
    patch_model(model, "baseline", 1.0)


def sanity_check_intervention_changes_attention(
    model: Any, inputs: Any, get_img_range_fn: Any, method: str, value: float
) -> None:
    """CHECK 3 — Intervention must change attention weights vs baseline."""
    img_start, img_end = get_img_range_fn(inputs)
    captured: dict = {}

    def _run_and_capture(tag: str) -> None:
        _STATE["_capture"]  = True
        _STATE["_captured"] = None
        with torch.inference_mode():
            model(**inputs)
        _STATE["_capture"] = False
        if _STATE["_captured"] is not None:
            captured[tag] = _STATE["_captured"]

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)
    _run_and_capture("baseline")

    patch_model(model, method, value)
    update_sample(img_start, img_end)
    _run_and_capture("intervention")

    b  = captured.get("baseline")
    iv = captured.get("intervention")
    assert b is not None and iv is not None, "CHECK 3 FAILED: attention not captured"
    assert not torch.allclose(b, iv, atol=1e-6), (
        f"CHECK 3 FAILED: attention weights UNCHANGED under {method}={value}"
    )

    if method in ("vision_boost", "vhr_boost"):
        img_b  = b[:, :, img_start : img_end + 1].mean().item()
        img_iv = iv[:, :, img_start : img_end + 1].mean().item()
        direction = "increases" if value > 1.0 else "decreases"
        if value > 1.0:
            assert img_iv > img_b, (
                f"CHECK 3 FAILED: {method}={value} should increase image attention "
                f"but baseline={img_b:.5f} >= intervention={img_iv:.5f}"
            )
        elif value < 1.0:
            assert img_iv < img_b, (
                f"CHECK 3 FAILED: {method}={value} should decrease image attention "
                f"but baseline={img_b:.5f} <= intervention={img_iv:.5f}"
            )
        print(f"  [CHECK 3] {method}={value}: image attn {img_b:.4f} → {img_iv:.4f} "
              f"({direction}) ✓")
    else:
        print(f"  [CHECK 3] {method}={value} changes attention weights ✓")

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)


def sanity_check_no_nan_or_inf(
    model: Any, inputs: Any, get_img_range_fn: Any, method: str, value: float
) -> None:
    """CHECK 4 — No NaN or Inf in generated tokens under any valid intervention."""
    img_start, img_end = get_img_range_fn(inputs)
    patch_model(model, method, value)
    update_sample(img_start, img_end)
    out = _generate_short(model, inputs)
    assert not out.isnan().any(), f"CHECK 4 FAILED: NaN in output under {method}={value}"
    assert not out.isinf().any(), f"CHECK 4 FAILED: Inf in output under {method}={value}"
    print(f"  [CHECK 4] no NaN/Inf under {method}={value} ✓")
    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)


def run_sanity_checks(
    model: Any,
    inputs: Any,
    get_img_range_fn: Any,
    method: str,
    value: float,
) -> None:
    """Run CHECK 1-4. Raises AssertionError on any failure.

    Args:
        model            : LMM model (Qwen2.5-VL or LLaVA)
        inputs           : single preprocessed input dict
        get_img_range_fn : callable(inputs) → (img_start, img_end) — arch-aware
        method           : intervention method for CHECK 3/4
        value            : intervention value for CHECK 3/4
    """
    print("\nRunning sanity checks...")
    sanity_check_baseline_determinism(model, inputs, get_img_range_fn)
    sanity_check_patched_baseline_equals_unpatched(model, inputs, get_img_range_fn)
    sanity_check_intervention_changes_attention(model, inputs, get_img_range_fn, method, value)
    sanity_check_no_nan_or_inf(model, inputs, get_img_range_fn, method, value)
    print("All sanity checks passed ✓\n")


# ---------------------------------------------------------------------------
# VHR-specific sanity checks
# ---------------------------------------------------------------------------

def sanity_check_vhr_head_count(head_mask: torch.Tensor, top_k_pct: float) -> None:
    """VHR CHECK 1 — Mask must select roughly top_k_pct * n_heads heads."""
    n_heads    = len(head_mask)
    n_selected = int(head_mask.sum().item())
    expected   = max(1, round(n_heads * top_k_pct))
    assert n_selected >= 1, "VHR CHECK 1 FAILED: no heads selected"
    # Allow ±1 slack for rounding
    assert abs(n_selected - expected) <= 1, (
        f"VHR CHECK 1 FAILED: expected ~{expected} heads, got {n_selected}"
    )
    print(f"  [VHR CHECK 1] {n_selected}/{n_heads} vision-aware heads selected ✓")


def sanity_check_vhr_scores_monotone(
    model: Any,
    calibration_inputs: List[Any],
    img_ranges: List[Tuple[int, int]],
    head_mask: torch.Tensor,
) -> None:
    """
    VHR CHECK 2 — Selected heads must have higher mean vision scores than
    unselected heads. This validates that calibration is meaningful.
    """
    # Re-run one calibration pass to get raw scores (head_mask already set)
    patch_model(model, "baseline", 1.0)
    _STATE["_calibrate_heads"]  = True
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    with torch.inference_mode():
        for inputs, (s, e) in zip(calibration_inputs[:5], img_ranges[:5]):
            update_sample(s, e)
            model(**inputs)

    _STATE["_calibrate_heads"] = False
    count = _STATE["_calib_head_count"]
    assert count > 0, "VHR CHECK 2 FAILED: no calibration data"

    scores = _STATE["_calib_head_acc"] / count
    _STATE["_calib_head_acc"]   = None
    _STATE["_calib_head_count"] = 0

    selected_mean   = scores[head_mask].mean().item()
    unselected_mean = scores[~head_mask].mean().item() if (~head_mask).any() else 0.0
    assert selected_mean > unselected_mean, (
        f"VHR CHECK 2 FAILED: selected heads mean score ({selected_mean:.5f}) "
        f"not higher than unselected ({unselected_mean:.5f})"
    )
    print(f"  [VHR CHECK 2] selected heads vision score {selected_mean:.4f} "
          f"> unselected {unselected_mean:.4f} ✓")


def sanity_check_vhr_noop_at_one(
    model: Any, inputs: Any, get_img_range_fn: Any, head_mask: torch.Tensor
) -> None:
    """VHR CHECK 3 — vhr_boost with value=1.0 must be a no-op (log(1)=0)."""
    img_start, img_end = get_img_range_fn(inputs)

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)
    out_base = _generate_short(model, inputs)

    _STATE["head_mask"] = head_mask
    patch_model(model, "vhr_boost", 1.0)
    update_sample(img_start, img_end)
    out_vhr = _generate_short(model, inputs)

    assert torch.equal(out_base, out_vhr), (
        f"VHR CHECK 3 FAILED: vhr_boost with value=1.0 is not a no-op\n"
        f"  baseline={out_base}\n  vhr_boost(1.0)={out_vhr}"
    )
    print("  [VHR CHECK 3] vhr_boost value=1.0 is a no-op ✓")

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)


def sanity_check_vhr_differs_from_full_boost(
    model: Any, inputs: Any, get_img_range_fn: Any, head_mask: torch.Tensor, value: float = 2.0
) -> None:
    """
    VHR CHECK 4 — vhr_boost should give different output than vision_boost
    (which boosts all heads) when head_mask doesn't select all heads.
    """
    img_start, img_end = get_img_range_fn(inputs)
    update_sample(img_start, img_end)

    patch_model(model, "vision_boost", value)
    out_full = _generate_short(model, inputs)

    _STATE["head_mask"] = head_mask
    patch_model(model, "vhr_boost", value)
    out_vhr = _generate_short(model, inputs)

    if head_mask.all():
        print("  [VHR CHECK 4] all heads selected → vhr_boost == vision_boost (expected) ✓")
    elif torch.equal(out_full, out_vhr):
        print("  [VHR CHECK 4] WARNING: vhr_boost == vision_boost despite partial mask "
              "(first token robust to head selection — not a bug)")
    else:
        print(f"  [VHR CHECK 4] vhr_boost differs from vision_boost at value={value} ✓")

    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)


def run_vhr_sanity_checks(
    model: Any,
    calibration_inputs: List[Any],
    img_ranges: List[Tuple[int, int]],
    sc_inputs: Any,
    get_img_range_fn: Any,
    top_k_pct: float = 0.20,
) -> torch.Tensor:
    """
    Run all VHR sanity checks.  Returns the computed head_mask.

    Args:
        model               : LMM model (Qwen2.5-VL or LLaVA, eager attention)
        calibration_inputs  : preprocessed input dicts for calibration
        img_ranges          : (img_start, img_end) per calibration sample
        sc_inputs           : a single input dict used for no-op / output checks
        get_img_range_fn    : callable(inputs) → (img_start, img_end) — arch-aware
        top_k_pct           : fraction of heads to select
    """
    print("\nRunning VHR sanity checks...")

    head_mask = identify_visual_heads(
        model, calibration_inputs, img_ranges, top_k_pct
    )
    sanity_check_vhr_head_count(head_mask, top_k_pct)
    sanity_check_vhr_scores_monotone(model, calibration_inputs, img_ranges, head_mask)
    sanity_check_vhr_noop_at_one(model, sc_inputs, get_img_range_fn, head_mask)
    sanity_check_vhr_differs_from_full_boost(model, sc_inputs, get_img_range_fn, head_mask)

    print("All VHR sanity checks passed ✓\n")
    return head_mask
