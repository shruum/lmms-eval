"""
srf_c_v2.py — SRF-C v2: salient-region masking contrastive decoding.

Improvement over SRF-E (srf_e.py / SRF-C):
  Instead of zeroing the ENTIRE image in Pass 2 (→ garbage features on LLaVA),
  we zero only the CLIP-salient regions, keeping the background intact.

  Pass 1: full image + SRF attention boost         → logits_full
  Pass 2: salient pixels zeroed, background kept   → logits_nosaliency
  Final:  logits_full + γ * (logits_full − logits_nosaliency)

The contrastive signal captures "what does the salient object contribute?"
rather than "what does the whole image add to the language prior?" — more
targeted, preserves scene context for spatial/count reasoning.

Fallback: if no saliency mask is available (CLIP gate failed / object absent),
Pass 2 becomes a zero-image pass (same as SRF-E), since there is no salient
region to mask selectively.

Interface: identical to srf_e.py.
  eval.py dispatches with --method srfc2.
  eval.py injects: method_mod.patch = patch  (before setup())
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import clip_salience as clip_sal
import srf as _srf

patch = None  # injected by eval.py before setup()

# Re-export full SRF base interface
from srf import (
    setup,
    reset_for_dataset,
    prepare_sample,
    cleanup,
    BIAS,
    SALIENCY,
)

_SPATIAL = 2  # Qwen spatial merge factor; ignored for LLaVA (get_grid_dims handles it)


def _model_type() -> str:
    return "llava" if "llava" in _srf._model_id.lower() else "qwen"


def _make_nosaliency_inp(inp: dict) -> dict:
    """
    Build Pass 2 input by zeroing CLIP-salient pixel regions.

    Reads salience_mask from patch._STATE (set by srf.prepare_sample).
    If no mask is available (CLIP gate failed), falls back to zeroing
    the entire image (same behaviour as SRF-E).

    Returns a new dict sharing all tensors except pixel_values.
    """
    salience_mask = patch._STATE.get("salience_mask")

    if salience_mask is None:
        # Fallback: no saliency → zero whole image (SRF-E behaviour)
        inp_out = dict(inp)
        if "pixel_values" in inp_out:
            inp_out["pixel_values"] = torch.zeros_like(inp["pixel_values"])
        return inp_out

    # Upsample coarse saliency grid (6×6 for LLaVA) to full pixel resolution
    mt = _model_type()
    try:
        grid_h, grid_w = clip_sal.get_grid_dims(inp, _SPATIAL, mt)
    except Exception:
        # Fallback: zero whole image
        inp_out = dict(inp)
        if "pixel_values" in inp_out:
            inp_out["pixel_values"] = torch.zeros_like(inp["pixel_values"])
        return inp_out

    pv = inp["pixel_values"]           # [1, 3, H, W]
    H, W = pv.shape[-2], pv.shape[-1]

    # Upsample saliency mask to pixel space
    sal_2d   = salience_mask.reshape(grid_h, grid_w).unsqueeze(0).unsqueeze(0).float()
    sal_full = F.interpolate(sal_2d, size=(H, W), mode="bilinear", align_corners=False)
    sal_mask = (sal_full.squeeze() > 0.5)  # (H, W) bool: True = salient

    # Zero salient regions in a cloned tensor
    new_pv = pv.clone()
    new_pv[:, :, sal_mask] = 0.0
    inp_out = dict(inp)
    inp_out["pixel_values"] = new_pv
    return inp_out


# ---------------------------------------------------------------------------
# Contrastive inference
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, gamma: float = 1.0,
                           mode: str = "salient") -> torch.Tensor:
    """
    Single-step contrastive decoding for first-token answers (MME yes/no, POPE, MMVP).

    Pass 1 (SRF + full image):           logits_full
    Pass 2 (salient regions zeroed):     logits_nosaliency
    logits_final = logits_full + γ * (logits_full − logits_nosaliency)
    """
    # Pass 1: SRF active, full image
    patch._STATE["method"] = "srf"
    with torch.inference_mode():
        out_full = model(**inp)
    logits_full = out_full.logits[:, -1, :].float().clone()

    # Pass 2: baseline, salient regions zeroed
    inp_nosaliency = _make_nosaliency_inp(inp)
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out_nosaliency = model(**inp_nosaliency)
    logits_nosaliency = out_nosaliency.logits[:, -1, :].float()

    patch._STATE["method"] = "srf"
    return logits_full + gamma * (logits_full - logits_nosaliency)


def generate_contrastive(model, inp: dict, processor,
                         gamma: float = 1.0, mode: str = "salient",
                         max_new_tokens: int = 20,
                         content_offset: int = 0) -> list[int]:
    """
    Step-by-step contrastive generation for multi-token answers (MMHal).

    Same KV-cache structure as srf_e.generate_contrastive, but Pass 2
    uses salient-region-masked pixel_values instead of zero image.
    """
    device    = next(model.parameters()).device
    input_len = inp["input_ids"].shape[1]

    eos_ids: set[int] = set()
    eid = getattr(model.config, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif isinstance(eid, (list, tuple)):
        eos_ids.update(eid)

    inp_nosaliency = _make_nosaliency_inp(inp)
    generated  : list[int] = []
    past_full  = None
    past_noval = None

    for step in range(max_new_tokens):
        seq_len = input_len + step

        if step == 0:
            kw_full  = dict(inp,            use_cache=True)
            kw_noval = dict(inp_nosaliency, use_cache=True)
        else:
            new_tok   = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
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
        logits_nosaliency = out_noval.logits[:, -1, :].float()
        past_noval        = out_noval.past_key_values

        if step >= content_offset:
            logits_final = logits_full + gamma * (logits_full - logits_nosaliency)
        else:
            logits_final = logits_full
        next_token = int(logits_final.argmax(dim=-1).item())
        generated.append(next_token)

        if next_token in eos_ids:
            break

    patch._STATE["method"] = "srf"
    return generated
