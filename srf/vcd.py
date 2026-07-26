"""
VCD — Visual Contrastive Decoding for Qwen2.5-VL.

Adapted from https://github.com/DAMO-NLP-SG/VCD (CVPR 2024)
  Paper: Leng et al., "Mitigating Object Hallucinations in Large Vision-Language
         Models through Visual Contrastive Decoding"

Core idea: contrast logits from the original image against a noise-corrupted image.
  logits_vcd = (1 + α) * logits_orig - α * logits_noisy
  with Adaptive Plausibility Constraint: tokens unlikely under original are masked.

Noise: diffusion-schedule Gaussian noise at step t (default t=500 of 1000).
Source: vcd_add_noise.add_diffusion_noise — imported directly from VCD repo.

Interface matches srf.py so eval.py can dispatch with --method vcd.
"""
from __future__ import annotations

import sys
import pathlib

# Import noise utility directly from cloned VCD repo
_VCD_REPO = pathlib.Path("/volumes2/mllm/VCD/vcd_utils")
if str(_VCD_REPO) not in sys.path:
    sys.path.insert(0, str(_VCD_REPO))

import torch
import qwen_attn_patch as patch
from vcd_add_noise import add_diffusion_noise   # from VCD repo

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_NOISE_STEP: int   = 500    # diffusion timestep (0–999); higher = more noise
_CD_ALPHA:   float = 1.0    # contrastive strength
_CD_BETA:    float = 0.1    # Adaptive Plausibility Constraint threshold

_noisy_inp: dict | None = None   # noisy-image input dict for current sample


# ---------------------------------------------------------------------------
# Public API (matches srf.py interface)
# ---------------------------------------------------------------------------

def setup(model, processor, calib_dataset: str = "pope") -> None:
    """Activate patch in baseline mode — VCD needs no calibration."""
    patch.patch_model(model, method="baseline", value=1.0)


def reset_for_dataset(
    dataset: str = "pope",
    *,
    noise_step: int   | None = None,
    cd_alpha:   float | None = None,
    cd_beta:    float | None = None,
    **kwargs,   # absorb unused SRF overrides
) -> None:
    global _NOISE_STEP, _CD_ALPHA, _CD_BETA
    if noise_step is not None:
        _NOISE_STEP = noise_step
    if cd_alpha is not None:
        _CD_ALPHA = cd_alpha
    if cd_beta is not None:
        _CD_BETA = cd_beta


def prepare_sample(inp, img_start, img_end, image, question, model, processor) -> None:
    """Create diffusion-noisy pixel_values for the current sample."""
    global _noisy_inp
    patch._STATE["method"] = "baseline"

    if "pixel_values" in inp:
        pv = inp["pixel_values"]
        noisy_pv = add_diffusion_noise(pv.float(), _NOISE_STEP)
        noisy_pv = noisy_pv.to(dtype=pv.dtype, device=pv.device)
        _noisy_inp = {**inp, "pixel_values": noisy_pv}
    else:
        _noisy_inp = dict(inp)


def cleanup() -> None:
    global _noisy_inp
    _noisy_inp = None
    patch._STATE["method"] = "baseline"


# ---------------------------------------------------------------------------
# Contrastive helpers — called by eval.py via method_get_logits / method_generate
# ---------------------------------------------------------------------------

def _combine(logits: torch.Tensor, logits_cd: torch.Tensor) -> torch.Tensor:
    """VCD logit combination with Adaptive Plausibility Constraint."""
    cutoff = torch.log(torch.tensor(_CD_BETA, device=logits.device)) \
             + logits.max(dim=-1, keepdim=True).values
    diffs  = (1.0 + _CD_ALPHA) * logits - _CD_ALPHA * logits_cd
    return diffs.masked_fill(logits < cutoff, -float("inf"))


def get_contrastive_logits(model, inp: dict, beta: float = None) -> torch.Tensor:
    """Single-step VCD: two forward passes → combined logits [1, vocab]."""
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        logits    = model(**inp).logits[:, -1, :].float()
        logits_cd = model(**_noisy_inp).logits[:, -1, :].float()
    return _combine(logits, logits_cd)


def generate_contrastive(
    model, inp: dict, processor,
    beta: float = None,
    max_new_tokens: int = 20,
) -> list[int]:
    """
    Greedy VCD generation — two KV caches (original + noisy) run in parallel.

    Step 0: prefill both branches with full inputs.
    Step 1+: feed the same greedy token to both KV caches, combine logits.
    """
    device    = next(model.parameters()).device
    input_len = inp["input_ids"].shape[1]

    eos_ids: set[int] = set()
    eid = getattr(model.config, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif isinstance(eid, (list, tuple)):
        eos_ids.update(eid)

    generated: list[int] = []
    past_orig = None
    past_cd   = None

    patch._STATE["method"] = "baseline"

    for step in range(max_new_tokens):
        seq_len = input_len + step

        if step == 0:
            kw_orig = dict(inp,        use_cache=True)
            kw_cd   = dict(_noisy_inp, use_cache=True)
        else:
            new_tok   = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            attn_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            kw_orig   = dict(input_ids=new_tok, attention_mask=attn_mask,
                             past_key_values=past_orig, use_cache=True)
            kw_cd     = dict(input_ids=new_tok, attention_mask=attn_mask,
                             past_key_values=past_cd,   use_cache=True)

        with torch.inference_mode():
            out_orig = model(**kw_orig)
            out_cd   = model(**kw_cd)

        logits    = out_orig.logits[:, -1, :].float()
        logits_cd = out_cd.logits[:, -1, :].float()

        next_token = int(_combine(logits, logits_cd).argmax(dim=-1).item())
        generated.append(next_token)

        past_orig = out_orig.past_key_values
        past_cd   = out_cd.past_key_values

        if next_token in eos_ids:
            break

    return generated
