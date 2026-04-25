#!/usr/bin/env python3
"""SRF method for LLaVA — AGENT MODIFIES THIS FILE."""
from __future__ import annotations
import pathlib, sys, torch

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import llava_attn_patch as patch
import clip_salience as clip_sal

# =============================================================================
# STAGE 1: SALIENCY (CLIP-based for LLaVA)
# =============================================================================
SALIENCY = {
    "source": "clip",  # CLIP ViT-B/32 for LLaVA
    "clip_coarse_grid": 9,
    "clip_top_k_pct": 0.2,
    "clip_absence_thresh": 0.20,
    "clip_use_soft": True,
}

# =============================================================================
# STAGE 2: ATTENTION BIAS
# =============================================================================
BIAS = {
    "layer_start": 10,  # ClearSight's optimal range for LLaVA
    "layer_end": 14,
    "bias_mode": "prob_scale",  # Start with additive logit
    "boost_alpha": 3.0,
    "sys_beta": 0.15,  # System prompt suppression
    "head_top_k_pct": 0.0,  # Apply to all heads initially
    "srf_background_eps": 0.0,
}

_model, _processor = None, None

def setup(model, processor) -> None:
    global _model, _processor
    _model, _processor = model, processor

    print(f"  [SRF] Mode: {BIAS['bias_mode']}, boost={BIAS['boost_alpha']}")
    print(f"  [SRF] Layers: {BIAS['layer_start']}-{BIAS['layer_end']}, saliency={SALIENCY['source']}")

    # Configure the patch
    patch._STATE["vaf_layer_start"] = BIAS["layer_start"]
    patch._STATE["vaf_layer_end"] = BIAS["layer_end"]
    patch._STATE["vaf_beta"] = BIAS["sys_beta"]
    patch._STATE["srf_bias_mode"] = BIAS["bias_mode"]
    patch._STATE["srf_background_eps"] = BIAS.get("srf_background_eps", 0.0)

    # Initialize the patch (but don't enable yet)
    patch.patch_model(model, "baseline", 1.0)

def prepare_sample(inputs: dict, img_start: int, img_end: int, image, question: str, model, processor) -> None:
    """Compute saliency and configure the patch for this sample."""
    # Update image token range
    patch.update_sample(img_start, img_end)

    # Compute CLIP saliency
    grid_h, grid_w = 24, 24  # LLaVA uses 24x24 = 576 image tokens
    result = clip_sal.compute_clip_salience(
        image, question,
        grid_h, grid_w,
        top_k_pct=SALIENCY["clip_top_k_pct"],
        coarse_n=SALIENCY["clip_coarse_grid"],
    )

    # Set saliency mask
    sal = result.saliency if SALIENCY["clip_use_soft"] else result.mask
    patch._STATE["salience_mask"] = sal

    # Set boost strength
    patch._STATE["value"] = BIAS["boost_alpha"]

    # Enable SRF intervention
    patch._STATE["enabled"] = True
    patch._STATE["method"] = "srf"

def cleanup() -> None:
    """Reset patch state after sample."""
    patch._STATE["salience_mask"] = None
    patch._STATE["enabled"] = False
    patch._STATE["method"] = "baseline"
