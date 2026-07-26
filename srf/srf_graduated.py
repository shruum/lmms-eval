"""
Graduated multi-stage attention SRF implementation.
Different α levels for different saliency tiers.
"""
import torch
from typing import Optional
import sys
sys.path.insert(0, 'srf')
sys.path.insert(0, 'srf/saliency')

# Import base SRF functionality
from srf import prepare_sample_base, BIAS, SALIENCY, patch
import clip_salience as clip_sal

def prepare_sample_graduated(
    inputs,
    img_start: int,
    img_end: int,
    image,
    question: str,
    model,
    processor,
    alpha_high: float = 4.0,  # For highly salient tokens (top 20%)
    alpha_mid: float = 2.0,   # For medium salience (20%-60%)
    alpha_low: float = 1.0,   # For low salience (bottom 40%)
    eps: float = 0.2,
) -> None:
    """
    Graduated multi-stage attention boost.
    Different enhancement levels based on saliency percentile.
    """
    patch.update_sample(img_start, img_end)
    patch._STATE["method"] = "srf"

    # Compute standard CLIP saliency
    model_type = "llava" if "llava" in model.config._name_or_path.lower() else "qwen"
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, 2, model_type)  # spatial=2

    result = clip_sal.compute_clip_salience(
        image, question, grid_h, grid_w,
        top_k_pct=SALIENCY["clip_top_k_pct"],
        coarse_n=SALIENCY["clip_coarse_grid"],
    )

    # Compute saliency percentiles for graduated boost
    saliency = result.saliency
    n_tokens = len(saliency)

    # Top 20% → high boost
    # 20%-60% → medium boost
    # Bottom 40% → low/no boost
    sorted_vals, _ = torch.sort(saliency, descending=True)

    high_thresh = sorted_vals[int(n_tokens * 0.2)]  # 80th percentile
    mid_thresh = sorted_vals[int(n_tokens * 0.6)]   # 60th percentile

    # Create per-token enhancement factors
    enh_factors = torch.ones(n_tokens, dtype=torch.float32)

    # High saliency → strong boost
    high_mask = saliency >= high_thresh
    enh_factors[high_mask] = alpha_high

    # Medium saliency → medium boost
    mid_mask = (saliency >= mid_thresh) & (saliency < high_thresh)
    enh_factors[mid_mask] = alpha_mid

    # Low saliency → minimal boost (could be < 1.0 for suppression)
    low_mask = saliency < mid_thresh
    enh_factors[low_mask] = alpha_low

    # Apply absence-aware logic
    suppress_thresh = BIAS.get("clip_suppress_thresh", 0.0)
    suppress_alpha = BIAS.get("clip_suppress_alpha", 5.0)

    if suppress_thresh > 0.0 and result.max_sim < suppress_thresh:
        # Object likely absent → suppress all image tokens
        enh_factors = torch.ones(n_tokens, dtype=torch.float32) / (1.0 + suppress_alpha)

    # Set enhancement factors as salience mask (for multiplicative scaling)
    patch._STATE["salience_mask"] = enh_factors.to(inputs["input_ids"].device)
    patch._STATE["enh_para"] = 2.0  # Default fallback (will be overridden by mask)
