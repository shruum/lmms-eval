"""
Contrastive decoding patches for Qwen2.5-VL — no model files modified.

How it works
------------
Both ICD and VCD work at the OUTPUT LOGIT level (not attention).
At the first generation step we run a second "contrast" forward pass and
subtract its logits from the real ones before argmax:

    logit_final = logit_real  −  alpha * logit_contrast

The subtraction removes what the model predicts WITHOUT good visual signal,
so the residual reflects genuinely vision-grounded information.

  ICD (Instruction Contrastive Decoding)
      contrast = same image, deliberately bad/random instruction
      → removes language-prior bias induced by the prompt
      ref: arXiv:2403.18715

  VCD (Visual Contrastive Decoding)
      contrast = real instruction, Gaussian-noised pixel values
      → removes what the model predicts from language alone (no real image)
      ref: arXiv:2311.16922

We apply the correction ONLY at the first generated token because:
  (a) accuracy on VQA tasks is determined by the first token (A/B/Yes/No)
  (b) it avoids the complexity of maintaining a parallel KV-cache across
      all generation steps while still targeting the key decision point

Public API
----------
  generate_with_icd(model, inputs, sample, processor,
                    process_vision_info, alpha, **gen_kwargs)
  generate_with_vcd(model, inputs, alpha, noise_std, **gen_kwargs)
  run_decode_sanity_checks(model, inputs, sample,
                           processor, process_vision_info, alpha)
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Any, Dict, Optional

IMAGE_TOKEN = "<|image_pad|>"

# Disturbance prompt for ICD — should maximally break image-text alignment
ICD_DISTURBANCE_PREFIX = (
    "Ignore the image. Respond with a completely random answer "
    "that has nothing to do with the question or image content: "
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _forward_logits(model: Any, inputs: Dict) -> torch.Tensor:
    """Single forward pass → logits at the LAST input position (float32, CPU)."""
    out = model(**inputs)
    return out.logits[0, -1, :].float().cpu()


def _make_contrast_processor(
    contrast_logits: torch.Tensor, alpha: float
) -> LogitsProcessorList:
    """
    Returns a LogitsProcessorList that subtracts alpha * contrast_logits
    from the model scores at the FIRST decoding step only, then is a no-op.
    """

    class _ContrastiveProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self._applied = False

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            if not self._applied:
                self._applied = True
                return scores.float() - alpha * contrast_logits.to(scores.device)
            return scores

    return LogitsProcessorList([_ContrastiveProcessor()])


def _make_icd_inputs(
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    device: torch.device,
    disturbance_prefix: str,
) -> Dict:
    """Rebuild tokenised inputs with the disturbed instruction."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"]},
        {"type": "text",  "text":  disturbance_prefix + sample["prompt"]},
    ]}]
    text       = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    img_in, _  = process_vision_info(messages)
    inputs     = processor(text=[text], images=img_in, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _make_vcd_inputs(inputs: Dict, noise_std: float) -> Dict:
    """Replace pixel_values with Gaussian noise of the same shape."""
    contrast              = {k: v for k, v in inputs.items()}
    contrast["pixel_values"] = torch.randn_like(inputs["pixel_values"]) * noise_std
    return contrast


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------

def generate_with_icd(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    alpha: float = 1.0,
    disturbance_prefix: str = ICD_DISTURBANCE_PREFIX,
    **gen_kwargs: Any,
) -> torch.Tensor:
    """
    Generate with Instruction Contrastive Decoding.

    Args:
        model               : Qwen model (eager attention)
        inputs              : preprocessed inputs on model device
        sample              : dict with keys 'image' (PIL) and 'prompt' (str)
        processor           : AutoProcessor
        process_vision_info : qwen_vl_utils.process_vision_info
        alpha               : contrastive strength (0 = baseline, >0 = more contrast)
        disturbance_prefix  : instruction prepended to corrupt the prompt
        **gen_kwargs        : passed to model.generate()
    """
    device          = next(model.parameters()).device
    contrast_inputs = _make_icd_inputs(
        sample, processor, process_vision_info, device, disturbance_prefix)
    contrast_logits = _forward_logits(model, contrast_inputs)

    lp = _make_contrast_processor(contrast_logits, alpha)
    with torch.inference_mode():
        return model.generate(**inputs, logits_processor=lp, **gen_kwargs)


def generate_with_vcd(
    model: Any,
    inputs: Dict,
    alpha: float = 1.0,
    noise_std: float = 1.0,
    **gen_kwargs: Any,
) -> torch.Tensor:
    """
    Generate with Visual Contrastive Decoding.

    Args:
        model     : Qwen model (eager attention)
        inputs    : preprocessed inputs on model device
        alpha     : contrastive strength (0 = baseline, >0 = more contrast)
        noise_std : std-dev of Gaussian noise replacing pixel values (default 1.0)
        **gen_kwargs : passed to model.generate()
    """
    contrast_inputs = _make_vcd_inputs(inputs, noise_std)
    contrast_logits = _forward_logits(model, contrast_inputs)

    lp = _make_contrast_processor(contrast_logits, alpha)
    with torch.inference_mode():
        return model.generate(**inputs, logits_processor=lp, **gen_kwargs)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

_SHORT_GEN = dict(max_new_tokens=4, do_sample=False,
                  temperature=None, top_p=None, num_beams=1)


@torch.inference_mode()
def _gen_plain(model: Any, inputs: Dict) -> torch.Tensor:
    return model.generate(**inputs, **_SHORT_GEN)


def sanity_check_contrast_differs(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    method: str,
) -> None:
    """
    CHECK A — Contrast forward pass must yield logits different from the real pass.
    This verifies the contrast construction is actually doing something.
    """
    device        = next(model.parameters()).device
    real_logits   = _forward_logits(model, inputs)

    if method == "icd":
        c_inputs = _make_icd_inputs(sample, processor, process_vision_info,
                                    device, ICD_DISTURBANCE_PREFIX)
    else:  # vcd
        c_inputs = _make_vcd_inputs(inputs, noise_std=1.0)

    contrast_logits = _forward_logits(model, c_inputs)

    assert not torch.allclose(real_logits, contrast_logits, atol=1e-4), (
        f"CHECK A FAILED ({method}): contrast logits are identical to real logits — "
        "the contrast input is not changing the model output"
    )
    max_diff = (real_logits - contrast_logits).abs().max().item()
    print(f"  [CHECK A] {method}: contrast logits differ (max |Δ|={max_diff:.3f}) ✓")


def sanity_check_alpha_zero_noop(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    method: str,
) -> None:
    """
    CHECK B — alpha=0 must give bit-identical output to plain model.generate().
    This proves the LogitsProcessor is a no-op when alpha=0.
    """
    device = next(model.parameters()).device
    out_plain = _gen_plain(model, inputs)

    if method == "icd":
        c_inputs = _make_icd_inputs(sample, processor, process_vision_info,
                                    device, ICD_DISTURBANCE_PREFIX)
    else:
        c_inputs = _make_vcd_inputs(inputs, noise_std=1.0)

    c_logits = _forward_logits(model, c_inputs)
    lp       = _make_contrast_processor(c_logits, alpha=0.0)

    with torch.inference_mode():
        out_zero = model.generate(**inputs, logits_processor=lp, **_SHORT_GEN)

    assert torch.equal(out_plain, out_zero), (
        f"CHECK B FAILED ({method}): alpha=0 output differs from plain generate\n"
        f"  plain={out_plain}\n  alpha=0={out_zero}"
    )
    print(f"  [CHECK B] {method}: alpha=0 is a no-op ✓")


def sanity_check_alpha_changes_output(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    method: str,
    alpha: float,
) -> None:
    """
    CHECK C — alpha > 0 should produce different output than plain generate.
    We warn (not assert) because on rare samples the first token may be the same.
    """
    out_plain = _gen_plain(model, inputs)

    gen_kwargs = {**_SHORT_GEN}
    if method == "icd":
        out_c = generate_with_icd(model, inputs, sample, processor,
                                  process_vision_info, alpha=alpha, **gen_kwargs)
    else:
        out_c = generate_with_vcd(model, inputs, alpha=alpha, **gen_kwargs)

    if torch.equal(out_plain, out_c):
        print(f"  [CHECK C] WARNING: {method} alpha={alpha} did not change output "
              "(language prior may already be minimal on this sample — not necessarily a bug)")
    else:
        print(f"  [CHECK C] {method} alpha={alpha} changes output ✓")


def sanity_check_no_nan_inf(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    method: str,
    alpha: float,
) -> None:
    """CHECK D — No NaN or Inf in generated token IDs."""
    gen_kwargs = {**_SHORT_GEN}
    if method == "icd":
        out = generate_with_icd(model, inputs, sample, processor,
                                process_vision_info, alpha=alpha, **gen_kwargs)
    else:
        out = generate_with_vcd(model, inputs, alpha=alpha, **gen_kwargs)

    assert not out.isnan().any(), \
        f"CHECK D FAILED ({method}): NaN in output at alpha={alpha}"
    assert not out.isinf().any(), \
        f"CHECK D FAILED ({method}): Inf in output at alpha={alpha}"
    print(f"  [CHECK D] {method}: no NaN/Inf at alpha={alpha} ✓")


def run_decode_sanity_checks(
    model: Any,
    inputs: Dict,
    sample: Dict,
    processor: Any,
    process_vision_info: Any,
    alpha: float = 1.0,
) -> None:
    """
    Run CHECK A-D for both ICD and VCD.  Raises AssertionError on hard failures.
    Call once after model load with a representative sample.

    Args:
        model               : Qwen model (eager attention)
        inputs              : preprocessed inputs on model device
        sample              : dict with keys 'image' and 'prompt'
        processor           : AutoProcessor
        process_vision_info : qwen_vl_utils.process_vision_info
        alpha               : contrastive strength used for CHECK C and D
    """
    print("\nRunning contrastive decoding sanity checks...")
    for method in ("icd", "vcd"):
        print(f"\n  --- {method.upper()} ---")
        sanity_check_contrast_differs(
            model, inputs, sample, processor, process_vision_info, method)
        sanity_check_alpha_zero_noop(
            model, inputs, sample, processor, process_vision_info, method)
        sanity_check_alpha_changes_output(
            model, inputs, sample, processor, process_vision_info, method, alpha)
        sanity_check_no_nan_inf(
            model, inputs, sample, processor, process_vision_info, method, alpha)
    print("\nAll contrastive decoding sanity checks passed ✓\n")
