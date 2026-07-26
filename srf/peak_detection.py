"""
Peak-to-mean ratio based absence detection.
This metric showed best separation (Cohen's d = 0.838) in our analysis.
"""

import torch


def compute_peak_to_mean(saliency_map: torch.Tensor) -> float:
    """
    Compute peak-to-mean ratio of saliency map.

    High ratio = sharp peaks with low background (absent object pattern)
    Low ratio = distributed saliency with background (present object pattern)

    Args:
        saliency_map: Tensor of any shape with saliency values

    Returns:
        float: Peak-to-mean ratio
    """
    peak_val = saliency_map.max().item()
    mean_val = saliency_map.mean().item()
    ratio = peak_val / (mean_val + 1e-10)
    return ratio


def detect_absence_peak_ratio(
    clip_saliency: torch.Tensor,
    clip_confidence: float,
    peak_thresh: float = 2.1,
    conf_thresh_low: float = 0.23,
    conf_thresh_high: float = 0.30,
) -> tuple:
    """
    Detect object absence using peak-to-mean ratio + CLIP confidence.

    Detection logic:
    - ABSENT: peak_ratio > threshold AND confidence < low_thresh
    - PRESENT: peak_ratio < threshold AND confidence > high_thresh
    - UNCERTAIN: everything else

    Args:
        clip_saliency: CLIP saliency map
        clip_confidence: CLIP max similarity score
        peak_thresh: Peak-to-mean ratio threshold (default: 2.1)
        conf_thresh_low: CLIP confidence threshold for absence (default: 0.23)
        conf_thresh_high: CLIP confidence threshold for presence (default: 0.30)

    Returns:
        tuple: (mode: str, peak_ratio: float, info: dict)
    """
    peak_ratio = compute_peak_to_mean(clip_saliency)

    info = {
        "peak_ratio": peak_ratio,
        "confidence": clip_confidence,
        "mean_saliency": clip_saliency.mean().item(),
        "max_saliency": clip_saliency.max().item(),
    }

    # Dual-criteria detection (more robust)
    if peak_ratio > peak_thresh and clip_confidence < conf_thresh_low:
        mode = "absent"
    elif peak_ratio < peak_thresh and clip_confidence > conf_thresh_high:
        mode = "present"
    else:
        mode = "uncertain"

    return mode, peak_ratio, info
