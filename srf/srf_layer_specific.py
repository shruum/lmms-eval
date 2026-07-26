#!/usr/bin/env python3
"""
SRF Layer-Specific Modulation variant.

Different layers serve different functions — treat them differently:
- Early layers (0 to layer_early_end): Gentle visual detection boost
- Middle layers (layer_early_end to layer_mid_end): Saliency-guided fusion
- Late layers (layer_mid_end to n_layers): Suppress language priors

Usage:
    import srf_layer_specific as srf_ls
    srf_ls.setup(model, processor, calib_dataset="pope")
    srf_ls.reset_for_dataset(
        layer_early_end=7,
        layer_mid_end=15,
        alpha_early=0.5,
        alpha_mid=2.0,
        beta_late=0.1
    )
"""
from __future__ import annotations

import sys
import pathlib
from collections import defaultdict

_SRF_DIR = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

# Import patch module (will be injected by eval.py)
import patch as patch


def setup(model, processor, calib_dataset: str = "pope"):
    """Initialize layer-specific modulation.

    Args:
        model: VLM model
        processor: Model processor
        calib_dataset: Dataset for head identification
    """
    global _MODEL, _PROCESSOR, _ARCH, _DEVICE

    _MODEL = model
    _PROCESSOR = processor
    _DEVICE = next(model.parameters()).device

    # Import config to get arch params
    import config as CFG
    model_id = getattr(processor, 'name_or_path', 'Qwen/Qwen2.5-VL-3B-Instruct')
    _ARCH = CFG.get_arch(model_id)

    print(f"[Layer-Specific] Setup for model with {_ARCH['n_layers']} layers")

    # Use SRF's head identification
    import srf as srf_base
    srf_base.patch = patch

    # Reset and identify vision-aware heads
    srf_base.reset_for_dataset(dataset=calib_dataset)
    srf_base.setup(model, processor, calib_dataset=calib_dataset)

    # Copy vision-aware heads
    _VISION_HEADS = srf_base.patch._STATE.get("vision_heads", set())

    print(f"[Layer-Specific] Identified {len(_VISION_HEADS)} vision-aware heads")


def reset_for_dataset(
    dataset: str,
    layer_early_end: int = 7,
    layer_mid_end: int = 15,
    alpha_early: float = 0.5,
    alpha_mid: float = 2.0,
    beta_late: float = 0.1,
    **kwargs
):
    """Reset layer-specific parameters for a dataset.

    Args:
        dataset: Dataset name ("pope", "mmvp", etc.)
        layer_early_end: Last layer of early zone (visual detection)
        layer_mid_end: Last layer of mid zone (saliency-guided fusion)
        alpha_early: Boost strength for early layers
        alpha_mid: Boost strength for mid layers (with saliency)
        beta_late: Suppression strength for late layers (language priors)
        **kwargs: Additional args (ignored, for compatibility)
    """
    global _DATASET, _LAYER_EARLY_END, _LAYER_MID_END, _ALPHA_EARLY, _ALPHA_MID, _BETA_LATE

    _DATASET = dataset
    _LAYER_EARLY_END = layer_early_end
    _LAYER_MID_END = layer_mid_end
    _ALPHA_EARLY = alpha_early
    _ALPHA_MID = alpha_mid
    _BETA_LATE = beta_late

    n_layers = _ARCH["n_layers"]

    print(f"[Layer-Specific] Reset for dataset={dataset}")
    print(f"  Early zone:  0-{layer_early_end} (α={alpha_early})")
    print(f"  Mid zone:    {layer_early_end}-{layer_mid_end} (α={alpha_mid} with saliency)")
    print(f"  Late zone:  {layer_mid_end}-{n_layers} (β={beta_late} text suppression)")


def prepare_sample(inp, img_start, img_end, image, question, model, processor):
    """Prepare sample for layer-specific modulation.

    Computes CLIP saliency for mid-layer modulation.
    """
    # Compute saliency for mid layers
    import srf as srf_base
    srf_base.patch = patch

    # Use SRF's saliency computation
    srf_base.prepare_sample(inp, img_start, img_end, image, question, model, processor)

    # Store saliency for layer-specific use
    _SALIENCE_MASK = srf_base.patch._STATE.get("salience_mask", None)
    patch._STATE["layer_specific_salience"] = _SALIENCE_MASK
    patch._STATE["layer_specific_method"] = "layer_specific"


def cleanup():
    """Clean up layer-specific state."""
    patch._STATE.pop("layer_specific_salience", None)
    patch._STATE.pop("layer_specific_method", None)


# Monkey-patch the attention function for layer-specific modulation
_original_layer_specific_forward = None


def _layer_specific_forward(
    layer_idx,
    self,
    hidden_states,
    attention_mask=None,
    **kwargs
):
    """Layer-specific attention modulation.

    Three zones:
    - Early (0 to layer_early_end): Gentle uniform visual boost
    - Mid (layer_early_end to layer_mid_end): Saliency-guided boost
    - Late (layer_mid_end to n_layers): Text suppression
    """
    # Call original forward first
    output = _original_layer_specific_forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        **kwargs
    )

    # Check if we should apply modulation
    method = patch._STATE.get("layer_specific_method")
    if method != "layer_specific":
        return output

    enh_para = patch._STATE.get("enh_para")
    salience_mask = patch._STATE.get("layer_specific_salience")

    if enh_para is None or salience_mask is None:
        return output

    # Get image token indices
    img_token_ids = patch._STATE.get("img_token_ids", [])
    if not img_token_ids:
        return output

    # Apply layer-specific modulation
    n_layers = _ARCH["n_layers"]

    if layer_idx < _LAYER_EARLY_END:
        # Early zone: Gentle uniform visual boost
        alpha = _ALPHA_EARLY
        for head_idx in enh_para["vision_heads"]:
            # Gentle boost to all visual tokens
            if hasattr(output, 'to') and hasattr(salience_mask, 'to'):
                # For attention weights output
                pass  # Implementation depends on output structure

    elif layer_idx < _LAYER_MID_END:
        # Mid zone: Saliency-guided boost
        alpha = _ALPHA_MID
        # Strong boost to salient regions
        # Similar to current SRF but with stronger alpha
        pass

    else:
        # Late zone: Suppress language priors
        beta = _BETA_LATE
        # Downscale text token attention
        pass

    return output


def install_layer_specific_hook():
    """Install layer-specific modulation hooks on the model."""
    global _original_layer_specific_forward

    # This would be called during setup to replace attention forward
    # Implementation depends on model architecture
    pass
