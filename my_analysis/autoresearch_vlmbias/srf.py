#!/usr/bin/env python3
"""
SRF method for VLM Bias — AGENT MODIFIES THIS FILE.

CLIP-only (HSSA removed — confirmed dead from POPE autoresearch).
Object always present → no suppress logic; low CLIP max_sim → uniform boost fallback.

Public interface:
    setup(model, processor)
    prepare_sample(inputs, img_start, img_end, image, question, model, processor)
    cleanup()
"""
from __future__ import annotations

import pathlib
import random
import re
import sys

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import torch
import qwen_attn_patch as patch
import clip_salience   as clip_sal

SALIENCY = {
    "clip_coarse_grid":     7,
    "clip_top_k_pct":       0.30,
    "clip_use_soft":        True,
    "clip_fallback_thresh": 0.20,
}

BIAS = {
    "layer_start":      8,
    "layer_end":        14,
    "head_top_k_pct":   0.20,
    "sys_beta":         0.10,
    "text_beta":        0.0,    # post-image text suppression strength (0.0 = disabled; tried 0.5/2.0 — no effect, language prior is MLP not attention)
    "text_layer_start": 20,    # ALL heads, deep layers (language prior forms here)
    "text_layer_end":   27,
    "bias_mode":        "additive_logit",
    "boost_alpha":      8.0,
    "background_eps":   0.5,
    "interp_lambda":    1.0,
    "prob_floor":       0.005,
    "img_scale":        1.5,
    "srf_apply_phase":  "generation",   # generation-only: skip prefill, only boost during token decoding
}

VIS_SAMPLES = 0   # disable vis during sweep for speed
VIS_DIR     = pathlib.Path(__file__).parent / "vis"
_model = _processor = None
_spatial = 2
_vis_count = 0


_GENERIC_NOUNS = {"thing", "things", "item", "items", "object", "objects", "image", "picture", "photo"}

def _extract_noun(question: str) -> str:
    """Extract the most CLIP-queryable noun from a VLM Bias question.

    Priority: specific counted object > scene container > fallback.
    E.g. 'How many logos are on this image?' → 'logos' (not 'image').
    """
    q = question.split("Answer")[0].strip().lower()
    # Priority 1: the specific thing being counted — best CLIP target
    m = re.search(r'how many (\w+(?:\s+\w+)?) (?:are|is|have|does)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun
    m = re.search(r'count the (\w+(?:\s+\w+)?) (?:pieces|on|in)', q)
    if m: return m.group(1).strip()
    # Priority 2: scene/container (less specific but better than nothing)
    m = re.search(r'(?:on|in) this (\w+(?:\s+\w+)?)', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun
    m = re.search(r'(?:on|in) the (\w+)', q)
    if m: return m.group(1).strip()
    words = re.findall(r'\b[a-z]{4,}\b', q)
    return words[0] if words else "object"


def setup(model, processor) -> None:
    global _model, _processor, _spatial, _vis_count
    _model, _processor = model, processor
    _spatial   = getattr(model.config.vision_config, "spatial_merge_size", 2)
    _vis_count = 0
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
    from datasets import load_dataset as hf_load
    from qwen_vl_utils import process_vision_info as pvi
    print("  [SRF] Calibrating heads on VLM Bias samples (20, seed=0)…")
    ds = hf_load("anvo25/vlms-are-biased", split="main")
    rows = list(ds)
    random.Random(0).shuffle(rows)
    img_id = _processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    device = next(_model.parameters()).device
    calib_inputs, img_ranges = [], []
    for r in rows[:20]:
        img  = r["image"].convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  r["prompt"]}]}]
        text   = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vis, _ = pvi(msgs)
        inp    = _processor(text=[text], images=vis, return_tensors="pt", padding=True).to(device)
        ids    = inp["input_ids"][0].tolist()
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
