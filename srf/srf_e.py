#!/usr/bin/env python3
"""
SRF-E (Evidence-Amplified SRF) — two-pass visual evidence amplification.

Extends SRF base with a contrastive second pass that isolates image
evidence from language priors:

    logits_final = logits_full + β * (logits_full - logits_noval)

where logits_noval is a forward pass with pixel_values zeroed (no-visual).
The difference (logits_full - logits_noval) captures what the image adds
over the language prior; β amplifies this signal.

All setup/calibration/saliency logic lives in srf.py. This file adds only
the two-pass inference functions.

Public interface (identical to srf.py, plus two extra functions):
    setup(model, processor, calib_dataset="pope")
    reset_for_dataset(phase, alpha, layer_end, eps, dataset)
    prepare_sample(inputs, img_start, img_end, image, question, model, processor)
    get_contrastive_logits(model, inp, beta, mode) -> Tensor [1, vocab]
    generate_contrastive(model, inp, processor, beta, mode, max_new_tokens) -> list[int]
    cleanup()
"""
from __future__ import annotations

import pathlib
import sys

_SRF_DIR      = pathlib.Path(__file__).parent           # srf/
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"         # my_analysis/
sys.path.insert(0, str(_SRF_DIR / "saliency"))           # clip_salience, hssa_salience
sys.path.insert(0, str(_SRF_DIR))                         # config, noun_extract, srf
sys.path.insert(0, str(_ANALYSIS_DIR))                    # qwen_attn_patch

import torch
import qwen_attn_patch as patch

# Re-export entire SRF base interface — callers can import from srf_e exclusively
from srf import (
    setup,
    reset_for_dataset,
    prepare_sample,
    cleanup,
    BIAS,
    SALIENCY,
)


# ---------------------------------------------------------------------------
# No-visual input construction
# ---------------------------------------------------------------------------

def _make_noval_inp(inp: dict, mode: str = "all") -> dict:
    """
    Build the no-visual input for Pass 2.

    mode="all"  — zero ALL pixel_values → ViT sees a black image (language-only pass)
    mode="clip" — (future) zero only CLIP-salient token positions; falls back to "all"
    """
    if mode == "clip":
        # CLIP-conditioned masking requires pixel-level ViT patch mapping.
        # Not yet implemented — fall through to "all".
        pass

    # "all": zero the entire pixel tensor
    inp_noval = dict(inp)
    if "pixel_values" in inp_noval:
        inp_noval["pixel_values"] = torch.zeros_like(inp["pixel_values"])
    return inp_noval


# ---------------------------------------------------------------------------
# Contrastive inference
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, beta: float = 1.0,
                           mode: str = "all") -> torch.Tensor:
    """
    Single-step contrastive decoding — returns logits [1, vocab] for the first token.

    Pass 1 (SRF + full image):   logits_full
    Pass 2 (baseline + no image): logits_noval

    logits_final = logits_full + β * (logits_full - logits_noval)

    Use for single-token answers (POPE yes/no, MMVP A/B).
    For multi-token answers, use generate_contrastive().
    """
    # Pass 1: SRF active, full image
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out_full = model(**inp)
    logits_full = out_full.logits[:, -1, :].float().clone()

    # Pass 2: baseline, no-visual
    inp_noval = _make_noval_inp(inp, mode=mode)
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out_noval = model(**inp_noval)
    logits_noval = out_noval.logits[:, -1, :].float()

    # Restore state
    patch._STATE["method"] = "srf"

    return logits_full + beta * (logits_full - logits_noval)


def generate_contrastive(model, inp: dict, processor,
                         beta: float = 1.0, mode: str = "all",
                         max_new_tokens: int = 20) -> list[int]:
    """
    Step-by-step contrastive generation — for multi-token answers (VLM Bias).

    Each step:
      - Forward pass 1 (SRF + full-image KV cache)
      - Forward pass 2 (baseline + no-visual KV cache)
      - Apply contrastive combination, pick next token via greedy argmax
      - Feed same token to both KV caches

    Returns list of generated token IDs (prompt not included).

    Note: suppresses language-prior format tokens (e.g. '{' in VLM Bias answer
    templates) — use SRF base (srf.py) for format-sensitive generation instead.
    """
    device    = next(model.parameters()).device
    input_len = inp["input_ids"].shape[1]

    eos_ids: set[int] = set()
    eid = getattr(model.config, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif isinstance(eid, (list, tuple)):
        eos_ids.update(eid)

    inp_noval  = _make_noval_inp(inp, mode=mode)
    generated  : list[int] = []
    past_full  = None
    past_noval = None

    for step in range(max_new_tokens):
        seq_len = input_len + step

        if step == 0:
            kw_full  = dict(inp,       use_cache=True)
            kw_noval = dict(inp_noval, use_cache=True)
        else:
            new_tok  = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            attn_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
            kw_full  = dict(input_ids=new_tok, attention_mask=attn_mask,
                            past_key_values=past_full,  use_cache=True)
            kw_noval = dict(input_ids=new_tok, attention_mask=attn_mask,
                            past_key_values=past_noval, use_cache=True)

        patch._STATE["method"] = "srf"
        with torch.inference_mode():
            out_full = model(**kw_full)
        logits_full = out_full.logits[:, -1, :].float()
        past_full   = out_full.past_key_values

        patch._STATE["method"] = "baseline"
        with torch.inference_mode():
            out_noval = model(**kw_noval)
        logits_noval = out_noval.logits[:, -1, :].float()
        past_noval   = out_noval.past_key_values

        logits_final = logits_full + beta * (logits_full - logits_noval)
        next_token   = int(logits_final.argmax(dim=-1).item())
        generated.append(next_token)

        if next_token in eos_ids:
            break

    patch._STATE["method"] = "srf"
    return generated
