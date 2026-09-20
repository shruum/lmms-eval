"""
srf_c_v3.py — SRF-C v3: embedding-space visual token zeroing.

Root cause of v1/v2 failure on LLaVA:
  Zeroing pixels (whole or salient) sends an OOD input to the frozen CLIP-L
  encoder, producing garbage embeddings that corrupt logits_noval.

Fix:
  Run the full CLIP+MLP pipeline ONCE to get the 576 projected visual tokens
  in LLM embedding space. For Pass 2, zero those token positions directly —
  CLIP is never called with a corrupted image.

  Pass 1: full image → CLIP → MLP → 576 LLM tokens + SRF attention boost  → logits_full
  Pass 2: same merged embeddings, visual positions [img_start:img_end] zeroed → logits_null
  Final:  logits_full + γ * (logits_full − logits_null)

For salient-only zeroing (when salience_mask is available):
  Only the salient token positions within [img_start:img_end] are zeroed.
  This targets "what does the salient object contribute?" rather than
  "what does any visual content contribute?"

Interface: identical to srf_e.py and srf_c_v2.py.
  eval.py dispatches with --method srfc3.
  eval.py injects: method_mod.patch = patch  (before setup())
"""
from __future__ import annotations

import pathlib
import sys

import torch

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import srf as _srf

patch = None  # injected by eval.py before setup()

# Re-export full SRF base interface
from srf import (
    setup,
    cleanup,
    BIAS,
    SALIENCY,
)

# Module-level storage for image token range (set in prepare_sample)
_IMG_RANGE: tuple[int, int] | None = None


def reset_for_dataset(*args, **kwargs):
    global _IMG_RANGE
    _IMG_RANGE = None
    _srf.reset_for_dataset(*args, **kwargs)


def prepare_sample(inp, img_start, img_end, image, query, model, processor):
    global _IMG_RANGE
    _IMG_RANGE = (img_start, img_end)
    _srf.prepare_sample(inp, img_start, img_end, image, query, model, processor)


# ---------------------------------------------------------------------------
# Embedding-space helpers
# ---------------------------------------------------------------------------

def _get_transformer(model) -> torch.nn.Module:
    """Return the LlamaModel backbone regardless of transformers version.
    Transformers 4.x: model.language_model.model
    Transformers 5.x: model.model.language_model.model
    """
    if hasattr(model, "language_model"):
        lm = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
    else:
        raise AttributeError(f"Cannot find language_model in {type(model)}")
    return lm.model if hasattr(lm, "model") else lm


def _capture_merged_embeds(model, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run one forward pass and capture the merged inputs_embeds (text + projected
    visual tokens) from inside LlavaForConditionalGeneration, before the LLM layers.

    Returns: (merged_embeds [1, seq, d], logits_full [1, vocab])
    """
    captured: dict = {}

    def _hook(module, args, kwargs):
        ie = kwargs.get("inputs_embeds")
        if ie is not None:
            captured["embeds"] = ie.detach().clone()

    # Hook at LlamaModel level — receives inputs_embeds after image merging
    handle = _get_transformer(model).register_forward_pre_hook(
        _hook, with_kwargs=True
    )
    try:
        patch._STATE["method"] = "srf"
        with torch.inference_mode():
            out = model(**inp)
        logits_full = out.logits[:, -1, :].float().clone()
    finally:
        handle.remove()

    return captured.get("embeds"), logits_full


def _make_null_embeds(merged_embeds: torch.Tensor,
                      img_start: int,
                      img_end: int,
                      salience_mask) -> torch.Tensor:
    """
    Zero visual token positions in LLM embedding space.
    If salience_mask is available, zero only the salient token indices.
    """
    null = merged_embeds.clone()
    if salience_mask is not None:
        sal_idx = salience_mask.nonzero(as_tuple=True)[0]  # indices within visual grid
        # sal_idx maps to positions img_start + sal_idx in the full sequence
        abs_idx = img_start + sal_idx
        # clamp to valid range
        abs_idx = abs_idx[abs_idx < img_end]
        null[:, abs_idx, :] = 0.0
    else:
        null[:, img_start:img_end, :] = 0.0
    return null


def _run_null_forward(model, inp: dict, null_embeds: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the LLM using pre-computed null embeddings.
    Passes inputs_embeds directly, bypassing vision tower entirely.
    """
    seq_len = null_embeds.shape[1]
    attn_mask = torch.ones(1, seq_len, device=null_embeds.device, dtype=torch.long)

    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out = model(
            inputs_embeds=null_embeds,
            attention_mask=attn_mask,
        )
    return out.logits[:, -1, :].float()


# ---------------------------------------------------------------------------
# Contrastive inference
# ---------------------------------------------------------------------------

def get_contrastive_logits(model, inp: dict, gamma: float = 1.0) -> torch.Tensor:
    """
    Single-step contrastive decoding for first-token answers (MME, POPE, MMVP).

    Pass 1: SRF + full image  → logits_full   (captured via hook, one forward pass)
    Pass 2: visual tokens zeroed in embed space → logits_null
    Final:  logits_full + γ * (logits_full − logits_null)
    """
    img_start, img_end = _IMG_RANGE or (0, 0)
    salience_mask = patch._STATE.get("salience_mask")

    # Pass 1: full forward, capture embeddings
    merged_embeds, logits_full = _capture_merged_embeds(model, inp)

    if merged_embeds is None:
        # Hook failed (shouldn't happen) — fall back to logits_full only
        return logits_full

    # Pass 2: zero visual tokens, run LLM only
    null_embeds  = _make_null_embeds(merged_embeds, img_start, img_end, salience_mask)
    logits_null  = _run_null_forward(model, inp, null_embeds)

    patch._STATE["method"] = "srf"
    return logits_full + gamma * (logits_full - logits_null)


def generate_contrastive(model, inp: dict, processor,
                         gamma: float = 1.0,
                         max_new_tokens: int = 20,
                         content_offset: int = 0) -> list[int]:
    """
    Step-by-step contrastive generation (MMHal open-ended).

    Step 0: capture merged embeddings via hook, run both full and null forward passes.
    Step 1+: extend with new token using KV cache; null pass uses language_model directly.
    """
    device    = next(model.parameters()).device
    input_len = inp["input_ids"].shape[1]
    img_start, img_end = _IMG_RANGE or (0, 0)
    salience_mask = patch._STATE.get("salience_mask")

    eos_ids: set[int] = set()
    eid = getattr(model.config, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif isinstance(eid, (list, tuple)):
        eos_ids.update(eid)

    generated: list[int] = []
    past_full  = None
    past_null  = None

    # Pre-compute null embeddings (reused every step)
    merged_embeds, _ = _capture_merged_embeds(model, inp)
    null_embeds_full = (_make_null_embeds(merged_embeds, img_start, img_end, salience_mask)
                        if merged_embeds is not None
                        else None)

    seq_len_0 = merged_embeds.shape[1] if merged_embeds is not None else input_len

    for step in range(max_new_tokens):
        seq_len = seq_len_0 + step

        # ── Pass 1: SRF, full image ──────────────────────────────────────────
        if step == 0:
            kw_full = dict(inp, use_cache=True)
        else:
            new_tok  = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            attn_full = torch.ones(1, seq_len, device=device, dtype=torch.long)
            kw_full  = dict(input_ids=new_tok, attention_mask=attn_full,
                            past_key_values=past_full, use_cache=True)

        patch._STATE["method"] = "srf"
        with torch.inference_mode():
            out_full = model(**kw_full)
        logits_full = out_full.logits[:, -1, :].float()
        past_full   = out_full.past_key_values

        # ── Pass 2: null visual tokens (via LLM only after step 0) ───────────
        if null_embeds_full is not None:
            if step == 0:
                attn_null = torch.ones(1, seq_len_0, device=device, dtype=torch.long)
                kw_null = dict(inputs_embeds=null_embeds_full,
                               attention_mask=attn_null, use_cache=True)
            else:
                new_tok   = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
                attn_null = torch.ones(1, seq_len, device=device, dtype=torch.long)
                kw_null   = dict(input_ids=new_tok, attention_mask=attn_null,
                                 past_key_values=past_null, use_cache=True)

            patch._STATE["method"] = "baseline"
            with torch.inference_mode():
                out_null = model(**kw_null)
            logits_null = out_null.logits[:, -1, :].float()
            past_null   = out_null.past_key_values

            if step >= content_offset:
                logits_final = logits_full + gamma * (logits_full - logits_null)
            else:
                logits_final = logits_full
        else:
            logits_final = logits_full

        next_token = int(logits_final.argmax(dim=-1).item())
        generated.append(next_token)

        if next_token in eos_ids:
            break

    patch._STATE["method"] = "srf"
    return generated
