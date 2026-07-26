"""
Multi-metric absence detection combining top 3 separating metrics:
1. Peak-to-mean ratio (Cohen's d = 0.838)
2. Mean saliency (Cohen's d = 0.810)
3. Top concentration (Cohen's d = 0.559)
"""

import torch
import numpy as np


def compute_multi_metrics(saliency_map: torch.Tensor) -> dict:
    """
    Compute multiple saliency metrics for robust detection.

    Returns dict with:
    - peak_to_mean: max / mean ratio
    - mean_saliency: average saliency intensity
    - top_concentration: ratio of top-10% to total saliency
    """
    saliency_flat = saliency_map.flatten()

    # 1. Peak-to-mean ratio
    peak_val = saliency_map.max().item()
    mean_val = saliency_map.mean().item()
    peak_to_mean = peak_val / (mean_val + 1e-10)

    # 2. Mean saliency intensity
    mean_saliency = mean_val

    # 3. Top-10% concentration
    top_k = int(len(saliency_flat) * 0.1)
    top_values = torch.topk(saliency_flat, top_k).values
    top_concentration = top_values.sum().item() / (saliency_map.sum().item() + 1e-10)

    return {
        "peak_to_mean": peak_to_mean,
        "mean_saliency": mean_saliency,
        "top_concentration": top_concentration,
    }


def detect_absence_multi_metric(
    clip_saliency: torch.Tensor,
    clip_confidence: float,
    peak_thresh: float = 2.1,
    mean_thresh: float = 0.49,
    top_conc_thresh: float = 0.182,
    conf_thresh_low: float = 0.23,
    conf_thresh_high: float = 0.30,
) -> tuple:
    """
    Multi-metric absence detection for robustness.

    Uses weighted voting from 3 metrics + CLIP confidence:
    - Metric 1: Peak-to-mean ratio (high = absent)
    - Metric 2: Mean saliency (low = absent)
    - Metric 3: Top concentration (high = absent)
    - Metric 4: CLIP confidence (low = absent)

    Detection logic:
    - Count "absent votes" from each metric
    - ABSENT if >= 3 votes for absence
    - PRESENT if >= 3 votes for presence
    - UNCERTAIN if tied (2-2)

    Args:
        clip_saliency: CLIP saliency map
        clip_confidence: CLIP max similarity score
        peak_thresh: Peak-to-mean threshold (default: 2.1)
        mean_thresh: Mean saliency threshold (default: 0.49)
        top_conc_thresh: Top concentration threshold (default: 0.182)
        conf_thresh_low: CLIP confidence low threshold (default: 0.23)
        conf_thresh_high: CLIP confidence high threshold (default: 0.30)

    Returns:
        tuple: (mode, votes, info)
        - mode: "absent", "present", or "uncertain"
        - votes: dict with individual metric votes
        - info: dict with metric values
    """
    metrics = compute_multi_metrics(clip_saliency)

    # Count votes for absence
    absent_votes = 0
    present_votes = 0

    # Vote 1: Peak-to-mean ratio
    if metrics["peak_to_mean"] > peak_thresh:
        absent_votes += 1
    elif metrics["peak_to_mean"] < peak_thresh:
        present_votes += 1

    # Vote 2: Mean saliency
    if metrics["mean_saliency"] < mean_thresh:
        absent_votes += 1
    elif metrics["mean_saliency"] > mean_thresh:
        present_votes += 1

    # Vote 3: Top concentration
    if metrics["top_concentration"] > top_conc_thresh:
        absent_votes += 1
    elif metrics["top_concentration"] < top_conc_thresh:
        present_votes += 1

    # Vote 4: CLIP confidence
    if clip_confidence < conf_thresh_low:
        absent_votes += 1
    elif clip_confidence > conf_thresh_high:
        present_votes += 1

    # Decision by majority vote
    if absent_votes >= 3:
        mode = "absent"
    elif present_votes >= 3:
        mode = "present"
    else:
        mode = "uncertain"

    votes = {
        "absent_votes": absent_votes,
        "present_votes": present_votes,
        "peak_vote": "absent" if metrics["peak_to_mean"] > peak_thresh else "present" if metrics["peak_to_mean"] < peak_thresh else "neutral",
        "mean_vote": "absent" if metrics["mean_saliency"] < mean_thresh else "present" if metrics["mean_saliency"] > mean_thresh else "neutral",
        "top_conc_vote": "absent" if metrics["top_concentration"] > top_conc_thresh else "present" if metrics["top_concentration"] < top_conc_thresh else "neutral",
        "conf_vote": "absent" if clip_confidence < conf_thresh_low else "present" if clip_confidence > conf_thresh_high else "neutral",
    }

    info = {
        **metrics,
        "confidence": clip_confidence,
        "absent_votes": absent_votes,
        "present_votes": present_votes,
    }

    return mode, votes, info
