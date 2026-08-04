"""
VAF — Visual Amplification Fusion for Qwen2.5-VL.

Adapted from ClearSight https://github.com/ustc-hyin/ClearSight (CVPR 2025)
  Paper: Yin et al., "ClearSight: Visual Signal Enhancement for Object
         Hallucination Mitigation in Multimodal Large Language Models"

Core idea: in mid-depth decoder layers, boost attention to image tokens and
suppress attention to system-prompt tokens — no CLIP, no calibration.

Original (LLaVA / LlamaAttention):
  attn_weights[..., img_tokens] *= enh_para          (multiplicative)
  attn_weights[..., sys_tokens] *= sup_para

Our adaptation (Qwen2.5-VL, additive-logit formulation):
  logits[..., img_tokens] += vaf_alpha               (pre-softmax, equivalent direction)
  logits[..., sys_tokens] -= vaf_beta

Applied via qwen_attn_patch "vaf" method — all heads, specific layer range.
Interface matches srf.py so eval.py can dispatch with --method vaf.
"""
from __future__ import annotations

import torch
import qwen_attn_patch as patch
import config as CFG

# ---------------------------------------------------------------------------
# Defaults (Qwen2.5-VL-3B proportional mapping from paper's LLaVA-1.5-7B)
# Paper: enh_para=1.15, sup_para=0.95, layers 8-15 of 32
# Additive equivalent: alpha≈4.0, beta≈0.30  (same relative scale as SRF)
# Layer range: 6-14 of 28  (proportional: 8/32*28 ≈ 7,  15/32*28 ≈ 13)
# ---------------------------------------------------------------------------

_VAF_ALPHA:       float = 4.0    # image token logit boost
_VAF_BETA:        float = 0.30   # system token logit suppression
_VAF_LAYER_START: int   = 6
_VAF_LAYER_END:   int   = 14


# ---------------------------------------------------------------------------
# Public API (matches srf.py interface)
# ---------------------------------------------------------------------------

def setup(model, processor, calib_dataset: str = "pope") -> None:
    """Install VAF attention patch — no calibration needed (all heads)."""
    patch.patch_model(model, method="vaf", value=_VAF_ALPHA)
    # All heads: set head_mask = None
    patch._STATE["head_mask"] = None


def reset_for_dataset(
    dataset: str = "pope",
    *,
    alpha:      float | None = None,
    vaf_beta:   float | None = None,
    layer_start: int  | None = None,
    layer_end:   int  | None = None,
    **kwargs,   # absorb unused SRF overrides
) -> None:
    a  = alpha      if alpha      is not None else _VAF_ALPHA
    b  = vaf_beta   if vaf_beta   is not None else _VAF_BETA
    ls = layer_start if layer_start is not None else _VAF_LAYER_START
    le = layer_end   if layer_end   is not None else _VAF_LAYER_END

    patch._STATE["method"]          = "vaf"
    patch._STATE["value"]           = a
    patch._STATE["vaf_beta"]        = b
    patch._STATE["vaf_layer_start"] = ls
    patch._STATE["vaf_layer_end"]   = le
    patch._STATE["head_mask"]       = None   # all heads, no calibration


def prepare_sample(inp, img_start, img_end, image, question, model, processor, **kwargs) -> None:
    """Set image token range for current sample."""
    patch._STATE["method"]    = "vaf"
    patch._STATE["img_start"] = img_start
    patch._STATE["img_end"]   = img_end - 1   # inclusive end (patch convention)
    patch._STATE["sys_end"]   = img_start - 1  # tokens before image = system + BOS


def cleanup() -> None:
    patch._STATE["method"]    = "baseline"
    patch._STATE["img_start"] = None
    patch._STATE["img_end"]   = None
    patch._STATE["sys_end"]   = None


# ---------------------------------------------------------------------------
# Dispatch hooks — called by eval.py via method_get_logits / method_generate
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, beta: float = None, **kwargs) -> torch.Tensor:
    """Single forward pass with VAF patch active."""
    patch._STATE["method"] = "vaf"
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def generate_contrastive(
    model, inp: dict, processor,
    beta: float = None,
    max_new_tokens: int = 20,
    **kwargs,
) -> list[int]:
    """Standard greedy generation with VAF attention patch active."""
    patch._STATE["method"] = "vaf"
    with torch.inference_mode():
        out_ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return out_ids[0, inp["input_ids"].shape[1]:].tolist()
