#!/usr/bin/env python3
"""
SRF method for MMVP — AGENT MODIFIES THIS FILE.

CLIP-only saliency. All objects present → no suppress logic.
MMVP tests fine-grained visual attributes (orientation, state, color, shape).
CLIP localises the subject object; we boost attention to those tokens so the
VLM "looks harder" at the right region rather than defaulting to language priors.

Public interface:
    setup(model, processor)
    prepare_sample(inputs, img_start, img_end, image, question, model, processor)
    cleanup()
"""
from __future__ import annotations

import pathlib
import random
import sys

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import torch
import qwen_attn_patch as patch
import clip_salience   as clip_sal
from noun_extract import extract_clip_noun

# Warmstart: VLM Bias best (alpha=8.0, eps=0.5, generation-only, layers 8-14, head_top_k=0.20)
# Adjusted for 7B (32 layers): layer_end → 15 to stay in same relative range.
SALIENCY = {
    "clip_coarse_grid":     7,
    "clip_top_k_pct":       0.30,
    "clip_use_soft":        True,
    "clip_fallback_thresh": 0.20,
}

BIAS = {
    "layer_start":      8,
    "layer_end":        15,     # 28 layers total; 8-15 is one wider than VLM Bias best (8-14) — good warmstart
    "head_top_k_pct":   0.20,
    "sys_beta":         0.10,
    "text_beta":        0.0,    # disabled — confirmed no effect on VLM Bias
    "text_layer_start": 24,     # proportionally deep for 32-layer model
    "text_layer_end":   31,
    "bias_mode":        "additive_logit",
    "boost_alpha":      4.0,
    "background_eps":   0.2,   # lower suppression — MMVP needs full visual context, not just salient region
    "interp_lambda":    1.0,
    "prob_floor":       0.005,
    "img_scale":        1.5,
    "srf_apply_phase":  "both",   # MMVP is single A/B token — decision made at prefill, not generation
}

_model = _processor = None
_spatial = 2


def _extract_noun(question: str) -> str:
    return extract_clip_noun(question, mode="mmvp")


def setup(model, processor) -> None:
    global _model, _processor, _spatial
    _model, _processor = model, processor
    _spatial   = getattr(model.config.vision_config, "spatial_merge_size", 2)
    _calibrate_heads()
    patch.patch_model(model, "vaf", max(float(BIAS["boost_alpha"]), 1e-6))
    patch._STATE["vaf_layer_start"]    = BIAS["layer_start"]
    patch._STATE["vaf_layer_end"]      = BIAS["layer_end"]
    patch._STATE["vaf_beta"]           = BIAS["sys_beta"]
    patch._STATE["srf_background_eps"] = BIAS["background_eps"]
    patch._STATE["srf_bias_mode"]      = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"]  = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]     = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]      = BIAS["img_scale"]
    patch._STATE["srf_apply_phase"]    = BIAS["srf_apply_phase"]
    patch._STATE["srf_text_beta"]         = BIAS["text_beta"]
    patch._STATE["srf_text_layer_start"]  = BIAS["text_layer_start"]
    patch._STATE["srf_text_layer_end"]    = BIAS["text_layer_end"]


def _calibrate_heads() -> None:
    if BIAS["head_top_k_pct"] <= 0.0:
        patch._STATE["head_mask"] = None
        return
    from qwen_vl_utils import process_vision_info as pvi
    print("  [SRF] Calibrating heads on MMVP samples (20, seed=0)…")
    import pandas as pd
    df = pd.read_csv("/volumes2/hugging_face_cache/mmvp_questions/Questions.csv")
    from datasets import load_dataset as hf_load
    img_ds = hf_load("MMVP/MMVP", split="train")
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {csv_1idx: hf_idx for hf_idx, csv_1idx in enumerate(lex_sorted)}

    img_id = _processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(_model.parameters()).device

    rng = random.Random(0)
    idxs = list(range(1, 301))
    rng.shuffle(idxs)
    calib_inputs, img_ranges = [], []
    for csv_1idx in idxs[:20]:
        row = df.iloc[csv_1idx - 1]
        img = img_ds[csv_to_hf[csv_1idx]]["image"].convert("RGB")
        import re as _re
        opt_matches = _re.findall(r'\(([ab])\)\s*([^(]+)', str(row["Options"]), _re.IGNORECASE)
        choices  = {m[0].upper(): m[1].strip() for m in opt_matches}
        opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
        prompt   = f"{row['Question']}\n{opt_text}\nAnswer with the option's letter from the given choices directly."
        msgs     = [{"role": "user", "content": [{"type": "image", "image": img},
                                                   {"type": "text",  "text":  prompt}]}]
        text     = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _   = pvi(msgs)
        inp      = _processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)
        ids      = inp["input_ids"][0].tolist()
        calib_inputs.append(inp)
        img_ranges.append((ids.index(img_id), len(ids)-1-ids[::-1].index(img_id)))

    patch.identify_visual_heads(_model, calib_inputs, img_ranges, BIAS["head_top_k_pct"])
    n_sel = int(patch._STATE["head_mask"].sum().item())
    print(f"  [SRF] {n_sel} heads selected (top {BIAS['head_top_k_pct']*100:.0f}%)")
    del calib_inputs, img_ranges
    torch.cuda.empty_cache()


def prepare_sample(inputs, img_start, img_end, image, question, model, processor) -> None:
    patch.update_sample(img_start, img_end)
    patch._STATE["value"]             = BIAS["boost_alpha"]
    patch._STATE["method"]            = "srf"
    patch._STATE["srf_bias_mode"]     = BIAS["bias_mode"]
    patch._STATE["srf_interp_lambda"] = BIAS["interp_lambda"]
    patch._STATE["srf_prob_floor"]    = BIAS["prob_floor"]
    patch._STATE["srf_img_scale"]     = BIAS["img_scale"]
    patch._STATE["srf_text_beta"]     = BIAS["text_beta"]
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
    noun   = _extract_noun(question)
    result = clip_sal.compute_clip_salience(image, noun, grid_h, grid_w,
                 top_k_pct=SALIENCY["clip_top_k_pct"], coarse_n=SALIENCY["clip_coarse_grid"])
    if result.max_sim < SALIENCY["clip_fallback_thresh"]:
        patch._STATE["salience_mask"] = None
    else:
        patch._STATE["salience_mask"] = result.saliency if SALIENCY["clip_use_soft"] else result.mask


def cleanup() -> None:
    patch._STATE["salience_mask"] = None
    patch._STATE["method"]        = "vaf"
