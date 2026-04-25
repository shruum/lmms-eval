"""
SRF method for LLaVA — AGENT MODIFIES THIS FILE.
Using ClearSight's VAF approach (multiplicative scaling on attention logits).
"""
from __future__ import annotations
import pathlib, sys, torch

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import llava_attn_patch_working as patch
import clip_salience as clip_sal

# =============================================================================
# STAGE 1: SALIENCY (CLIP-based for LLaVA)
# =============================================================================
SALIENCY = {
    "source": "clip",  # CLIP ViT-B/32 for LLaVA
    "clip_coarse_grid": 5,
    "clip_top_k_pct": 0.4,
    "clip_absence_thresh": 0.20,
    "clip_use_soft": True,
}

# =============================================================================
# STAGE 2: ATTENTION BIAS (ClearSight VAF parameters)
# =============================================================================
BIAS = {
    "layer_start": 10,  # ClearSight's optimal range for LLaVA
    "layer_end": 16,
    "enh_para": 1.05,  # Visual enhancement (boost)
    "sup_para": 1.0,  # System suppression
}

_model, _processor = None, None

def setup(model, processor) -> None:
    global _model, _processor
    _model, _processor = model, processor

    print(f"  [SRF] enh_para={BIAS['enh_para']}, sup_para={BIAS['sup_para']}")
    print(f"  [SRF] Layers: {BIAS['layer_start']}-{BIAS['layer_end']}, saliency={SALIENCY['source']}")

    # Initialize the patch with baseline
    patch.patch_model(model, "baseline", 1.0, 1.0)

def prepare_sample(inputs: dict, img_start: int, img_end: int, image, question: str, model, processor) -> None:
    """Compute saliency and configure the patch for this sample."""
    # Update image token range (sys_len, img_len)
    patch.update_sample(img_start, img_end)

    # For now, use uniform enhancement (no per-token saliency)
    # TODO: Could integrate CLIP saliency later for per-token scaling

    # Enable VAF with configured parameters
    patch._STATE["method"] = "srf"
    patch._STATE["enh_para"] = BIAS["enh_para"]
    patch._STATE["sup_para"] = BIAS["sup_para"]

def cleanup() -> None:
    """Reset patch state after sample."""
    patch._STATE["method"] = "baseline"
