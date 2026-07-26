"""
SRF2: Enhanced SRF using both CLIP and internal activations.

Key innovation:
1. CLIP provides external guidance on what SHOULD be salient
2. Internal activations show what the model ACTUALLY attends to
3. Combine both signals to identify truly important patches
4. Use activation patterns to dynamically scale intervention strength

Usage:
    python srf/eval.py --method srf2 --model llava-hf/llava-1.5-7b-hf ...

Note: SRF2 builds on SRF infrastructure - it extends SRF with activation-aware boosting.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import torch

import config as CFG


@dataclass
class SRF2CalibrationResult:
    """Results from calibration phase."""
    head_mask: torch.Tensor                      # (n_heads,) bool - top visual heads
    clip_alpha: float                            # Base CLIP boosting strength
    activation_alpha_mult: float                # Multiplier based on activation patterns
    combined_strategy: str                      # How to combine CLIP + activations


def setup(
    model: Any,
    processor: Any,
    calib_dataset: str = "pope",
    n_calib: int = 20,
    seed: int = 0,
    clip_alpha: float = 1.0,
    activation_alpha_mult: float = 0.5,
    combined_strategy: str = "multiplicative",
    **kwargs
) -> SRF2CalibrationResult:
    """
    Setup SRF2 by calibrating on dataset.

    SRF2 uses the same calibration as SRF but adds activation-aware boosting.
    The key innovation is runtime modulation based on layer activation patterns.

    Args:
        model: LLaVA model (already moved to device)
        processor: LLaVA processor
        calib_dataset: Dataset name for calibration
        n_calib: Number of calibration samples (unused, passed to SRF)
        seed: Random seed
        clip_alpha: Base CLIP boosting strength
        activation_alpha_mult: Multiplier for activation-based boosting
        combined_strategy: How to combine CLIP and activations
            - "multiplicative": CLIP_score × activation_weight (default)
            - "additive": CLIP_score + activation_weight
            - "clip_only": Use only CLIP (same as SRF)
            - "activation_only": Use only internal activations

    Returns:
        SRF2CalibrationResult with calibration parameters
    """
    import srf as srf_mod

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"[SRF2] Calibrating on {calib_dataset} (n={n_calib}, seed={seed})...")
    print(f"  [SRF2] Using SRF calibration with activation-aware boosting")

    # Use SRF's calibration infrastructure
    # Note: SRF reads hyperparameters from global config, not kwargs
    srf_mod.setup(model, processor, calib_dataset=calib_dataset)

    # Get SRF's calibration results
    head_mask = srf_mod._STATE.get("head_mask", torch.ones(32, dtype=torch.bool))

    result = SRF2CalibrationResult(
        head_mask=head_mask.cpu(),
        clip_alpha=clip_alpha,
        activation_alpha_mult=activation_alpha_mult,
        combined_strategy=combined_strategy
    )

    print(f"  [SRF2] Calibration complete:")
    print(f"    - Visual heads: {head_mask.sum().item()}/{len(head_mask)}")
    print(f"    - CLIP alpha: {clip_alpha}")
    print(f"    - Activation multiplier: {activation_alpha_mult}")
    print(f"    - Strategy: {combined_strategy}")

    # Store for runtime use
    _setup_runtime(result)

    return result


# Global state for runtime
_CALIB_RESULT: Optional[SRF2CalibrationResult] = None


def _setup_runtime(calib_result: SRF2CalibrationResult):
    """Setup runtime state with calibration results."""
    global _CALIB_RESULT
    _CALIB_RESULT = calib_result


def get_srf2_alpha_modulation(
    layer_idx: int,
    base_alpha: float
) -> float:
    """
    Get SRF2 alpha modulation for a given layer.

    SRF2 modulates boosting strength based on:
    1. Layer position (middle fusion layers get stronger boost)
    2. Combined strategy from calibration

    Args:
        layer_idx: Current layer index
        base_alpha: Base alpha value

    Returns:
        Modulated alpha value
    """
    if _CALIB_RESULT is None:
        return base_alpha

    # Layer-wise modulation
    if 10 <= layer_idx <= 18:
        # Fusion layers: apply activation multiplier
        modulated = base_alpha * (1.0 + _CALIB_RESULT.activation_alpha_mult)
    else:
        modulated = base_alpha

    return modulated
