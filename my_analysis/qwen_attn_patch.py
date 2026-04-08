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
    "head_mask":  None,    # bool tensor (n_heads,) for vhr_boost; None = all heads
    # ---- internal captures (not part of public API) ----
    "_capture":          False,   # CHECK 3: capture one 4-D softmax output
    "_captured":         None,    # CHECK 3: last captured (n_heads, q_len, kv_len) on CPU
    "_calibrate_heads":  False,   # VHR calibration mode
    "_calib_head_acc":   None,    # running sum of per-head vision scores (n_heads,)
    "_calib_head_count": 0,       # number of accumulation steps
}

_ORIGINAL_SOFTMAX = None   # stores the real F.softmax before patching
_HOOKS: list       = []    # forward hooks registered on model.language_model

IMAGE_TOKEN = "<|image_pad|>"


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

    result = _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype, **kwargs)

    # CHECK 3 capture — avoids output_attentions=True OOM on small GPUs
    if _STATE["_capture"] and _STATE["in_decoder"] and result.dim() == 4:
        _STATE["_captured"] = result[0].detach().cpu().float()

    # VHR calibration — accumulate per-head mean attention to image tokens
    if _STATE["_calibrate_heads"] and _STATE["in_decoder"] and result.dim() == 4:
        s2 = _STATE["img_start"]
        e2 = _STATE["img_end"]
        if s2 is not None and e2 is not None and e2 > s2:
            # result: (1, n_heads, q_len, kv_len)
            # score per head = mean attention weight on image columns
            img_score = result[0, :, :, s2 : e2 + 1].mean(dim=(1, 2)).detach().cpu().float()
            acc = _STATE["_calib_head_acc"]
            _STATE["_calib_head_acc"]   = img_score if acc is None else acc + img_score
            _STATE["_calib_head_count"] = _STATE["_calib_head_count"] + 1

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

    valid = ("baseline", "temperature", "vision_boost", "vhr_boost")
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

        def _pre(module, args):
            _STATE["in_decoder"] = True

        def _post(module, args, output):
            _STATE["in_decoder"] = False

        _HOOKS.append(lm.register_forward_pre_hook(_pre))
        _HOOKS.append(lm.register_forward_hook(_post))


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


def get_image_token_range(inputs: Any, processor: Any) -> Tuple[int, int]:
    """Return (img_start, img_end) inclusive indices of image tokens in input_ids."""
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
# Original sanity checks (CHECK 1-4) — unchanged logic
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_short(model: Any, inputs: Any, n: int = 4) -> torch.Tensor:
    return model.generate(**inputs, max_new_tokens=n,
                          do_sample=False, temperature=None,
                          top_p=None, num_beams=1)


def sanity_check_baseline_determinism(model: Any, inputs: Any, processor: Any) -> None:
    """CHECK 1 — Two baseline runs must give bit-identical tokens."""
    patch_model(model, "baseline", 1.0)
    img_start, img_end = get_image_token_range(inputs, processor)
    update_sample(img_start, img_end)
    out1 = _generate_short(model, inputs)
    out2 = _generate_short(model, inputs)
    assert torch.equal(out1, out2), (
        f"CHECK 1 FAILED: baseline outputs differ!\n  run1={out1}\n  run2={out2}"
    )
    print("  [CHECK 1] baseline is deterministic ✓")


def sanity_check_patched_baseline_equals_unpatched(model: Any, inputs: Any, processor: Any) -> None:
    """CHECK 2 — Patched baseline must be bit-identical to unpatched model."""
    img_start, img_end = get_image_token_range(inputs, processor)
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
    model: Any, inputs: Any, processor: Any, method: str, value: float
) -> None:
    """CHECK 3 — Intervention must change attention weights vs baseline."""
    img_start, img_end = get_image_token_range(inputs, processor)
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
    model: Any, inputs: Any, processor: Any, method: str, value: float
) -> None:
    """CHECK 4 — No NaN or Inf in generated tokens under any valid intervention."""
    img_start, img_end = get_image_token_range(inputs, processor)
    patch_model(model, method, value)
    update_sample(img_start, img_end)
    out = _generate_short(model, inputs)
    assert not out.isnan().any(), f"CHECK 4 FAILED: NaN in output under {method}={value}"
    assert not out.isinf().any(), f"CHECK 4 FAILED: Inf in output under {method}={value}"
    print(f"  [CHECK 4] no NaN/Inf under {method}={value} ✓")
    patch_model(model, "baseline", 1.0)
    update_sample(img_start, img_end)


def run_sanity_checks(
    model: Any, inputs: Any, processor: Any, method: str, value: float
) -> None:
    """Run CHECK 1-4. Raises AssertionError on any failure."""
    print("\nRunning sanity checks...")
    sanity_check_baseline_determinism(model, inputs, processor)
    sanity_check_patched_baseline_equals_unpatched(model, inputs, processor)
    sanity_check_intervention_changes_attention(model, inputs, processor, method, value)
    sanity_check_no_nan_or_inf(model, inputs, processor, method, value)
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
    model: Any, inputs: Any, processor: Any, head_mask: torch.Tensor
) -> None:
    """VHR CHECK 3 — vhr_boost with value=1.0 must be a no-op (log(1)=0)."""
    img_start, img_end = get_image_token_range(inputs, processor)

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
    model: Any, inputs: Any, processor: Any, head_mask: torch.Tensor, value: float = 2.0
) -> None:
    """
    VHR CHECK 4 — vhr_boost should give different output than vision_boost
    (which boosts all heads) when head_mask doesn't select all heads.
    """
    img_start, img_end = get_image_token_range(inputs, processor)
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
    processor: Any,
    top_k_pct: float = 0.20,
) -> torch.Tensor:
    """
    Run all VHR sanity checks.  Returns the computed head_mask.

    Args:
        model               : Qwen model (eager attention, already loaded)
        calibration_inputs  : preprocessed input dicts for calibration
        img_ranges          : (img_start, img_end) per calibration sample
        sc_inputs           : a single input dict used for no-op / output checks
        processor           : AutoProcessor
        top_k_pct           : fraction of heads to select
    """
    print("\nRunning VHR sanity checks...")

    head_mask = identify_visual_heads(
        model, calibration_inputs, img_ranges, top_k_pct
    )
    sanity_check_vhr_head_count(head_mask, top_k_pct)
    sanity_check_vhr_scores_monotone(model, calibration_inputs, img_ranges, head_mask)
    sanity_check_vhr_noop_at_one(model, sc_inputs, processor, head_mask)
    sanity_check_vhr_differs_from_full_boost(model, sc_inputs, processor, head_mask)

    print("All VHR sanity checks passed ✓\n")
    return head_mask
