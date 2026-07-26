"""
Absence detection utilities for SRF.

Provides entropy-based object absence detection to improve POPE performance
on object-absent samples.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any


def compute_saliency_entropy(saliency_map: torch.Tensor) -> float:
    """
    Compute entropy of saliency distribution.

    High entropy → diffuse saliency → object likely absent
    Low entropy → focused saliency → object likely present

    Args:
        saliency_map: Tensor of shape (H, W) or (H*W,) with saliency scores

    Returns:
        float: Entropy value (higher = more diffuse/uniform)
    """
    # Flatten if needed
    if saliency_map.dim() > 1:
        saliency_flat = saliency_map.flatten()
    else:
        saliency_flat = saliency_map

    # Normalize to [0, 1] to form probability distribution
    saliency_sum = saliency_flat.sum() + 1e-10  # Avoid division by zero
    saliency_norm = saliency_flat / saliency_sum

    # Compute entropy: H = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    entropy = -torch.sum(saliency_norm * torch.log(saliency_norm + 1e-10))

    return entropy.item()


def detect_absence_entropy(
    clip_saliency: torch.Tensor,
    clip_confidence: float,
    entropy_thresh: float = 3.5,
    conf_thresh_low: float = 0.25,
    conf_thresh_high: float = 0.30
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Detect object absence using entropy + confidence (two-criteria detection).

    More robust than using confidence or entropy alone.

    Args:
        clip_saliency: CLIP saliency map (H, W) or (H*W,)
        clip_confidence: Maximum CLIP similarity score
        entropy_thresh: Entropy threshold for absence (default 3.5)
        conf_thresh_low: Confidence threshold for "absent" (default 0.25)
        conf_thresh_high: Confidence threshold for "present" (default 0.30)

    Returns:
        Tuple of:
            - mode: "absent", "present", or "uncertain"
            - entropy: Entropy value (for debugging/analysis)
            - info: Dict with additional info (confidence, thresholds used)
    """
    # Compute entropy
    entropy = compute_saliency_entropy(clip_saliency)

    # Two-criteria detection (both must agree)
    if clip_confidence < conf_thresh_low and entropy > entropy_thresh:
        # Low confidence AND high entropy → object absent
        mode = "absent"
    elif clip_confidence > conf_thresh_high and entropy < entropy_thresh:
        # High confidence AND low entropy → object present
        mode = "present"
    else:
        # Mixed signals or mid-range → uncertain
        mode = "uncertain"

    # Additional info for debugging
    info = {
        "confidence": clip_confidence,
        "entropy": entropy,
        "entropy_thresh": entropy_thresh,
        "conf_thresh_low": conf_thresh_low,
        "conf_thresh_high": conf_thresh_high
    }

    return mode, entropy, info


def detect_absence_max_saliency(
    clip_saliency: torch.Tensor,
    clip_confidence: float,
    max_thresh: float = 0.3
) -> Tuple[str, Dict[str, Any]]:
    """
    Simpler absence detection using max saliency value only.

    Fallback method if entropy doesn't work well.

    Args:
        clip_saliency: CLIP saliency map
        clip_confidence: Maximum CLIP similarity score
        max_thresh: Threshold for max saliency (default 0.3)

    Returns:
        Tuple of (mode, info) where mode is "absent" or "present"
    """
    max_sal = clip_saliency.max().item()

    if clip_confidence < 0.25 and max_sal < max_thresh:
        mode = "absent"
    else:
        mode = "present"

    info = {
        "max_saliency": max_sal,
        "confidence": clip_confidence
    }

    return mode, info


def compute_saliency_statistics(saliency_map: torch.Tensor) -> Dict[str, float]:
    """
    Compute various statistics of saliency map for analysis.

    Args:
        saliency_map: Saliency map tensor

    Returns:
        Dict with statistics: entropy, max, mean, std, focused_ratio
    """
    if saliency_map.dim() > 1:
        saliency_flat = saliency_map.flatten()
    else:
        saliency_flat = saliency_map

    entropy = compute_saliency_entropy(saliency_map)
    max_val = saliency_flat.max().item()
    mean_val = saliency_flat.mean().item()
    std_val = saliency_flat.std().item()

    # Focused ratio: fraction of saliency in top 10% pixels
    k = max(1, len(saliency_flat) // 10)
    top_k_values = torch.topk(saliency_flat, k).values
    focused_ratio = top_k_values.sum().item() / (saliency_flat.sum().item() + 1e-10)

    return {
        "entropy": entropy,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "focused_ratio": focused_ratio
    }
