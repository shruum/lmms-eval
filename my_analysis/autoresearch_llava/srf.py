#!/usr/bin/env python3
"""SRF method for LLaVA — AGENT MODIFIES THIS FILE."""
from __future__ import annotations
import pathlib, sys, torch

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import qwen_attn_patch as patch
import clip_salience as clip_sal

SALIENCY = {"source": "clip", "clip_coarse_grid": 7, "clip_top_k_pct": 0.30, "clip_use_soft": True}

BIAS = {
    "layer_start": 8, "layer_end": 15, "head_top_k_pct": 0.0, "sys_beta": 0.10,
    "bias_mode": "additive_logit", "boost_alpha": 4.0,  # STRONG BOOST
    "background_eps": 0.0, "srf_apply_phase": "prefill",
}

VIS_SAMPLES = 0  # Disable vis for speed
VIS_DIR = pathlib.Path(__file__).parent / "vis"
_model, _processor = None, None

def setup(model, processor) -> None:
    global _model, _processor
    _model, _processor = model, processor
    patch._STATE["head_mask"] = None
    print("  [SRF] head_top_k_pct=0 → all heads, boost_alpha=4.0")
    patch.patch_model(model, "vaf", BIAS["boost_alpha"])
    patch._STATE["vaf_layer_start"] = BIAS["layer_start"]
    patch._STATE["vaf_layer_end"] = BIAS["layer_end"]
    patch._STATE["vaf_beta"] = BIAS["sys_beta"]
    patch._STATE["srf_background_eps"] = BIAS["background_eps"]
    patch._STATE["srf_bias_mode"] = BIAS["bias_mode"]
    patch._STATE["srf_apply_phase"] = BIAS["srf_apply_phase"]

def prepare_sample(inputs: dict, img_start: int, img_end: int, image, question: str, model, processor) -> None:
    patch.update_sample(img_start, img_end)
    patch._STATE["value"] = BIAS["boost_alpha"]
    patch._STATE["method"] = "srf"
    patch._STATE["srf_bias_mode"] = BIAS["bias_mode"]
    grid_h, grid_w = 24, 24
    result = clip_sal.compute_clip_salience(image, question, grid_h, grid_w,
        top_k_pct=SALIENCY["clip_top_k_pct"], coarse_n=SALIENCY["clip_coarse_grid"])
    patch._STATE["salience_mask"] = result.saliency if SALIENCY["clip_use_soft"] else result.mask

def cleanup() -> None:
    patch._STATE["salience_mask"] = None
    patch._STATE["method"] = "vaf"
