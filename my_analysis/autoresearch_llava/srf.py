#!/usr/bin/env python3
"""SRF method for LLaVA — AGENT MODIFIES THIS FILE."""
from __future__ import annotations
import pathlib, sys, torch

_ANALYSIS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ANALYSIS_DIR))

import qwen_attn_patch as patch

SALIENCY = {"source": "clip"}  # Minimal
BIAS = {"bias_mode": "baseline"}  # NO-OP mode

_model, _processor = None, None

def setup(model, processor) -> None:
    global _model, _processor
    _model, _processor = model, processor
    print("  [SRF] Using baseline mode (no-op)")
    patch.patch_model(model, "baseline", 1.0)

def prepare_sample(inputs: dict, img_start: int, img_end: int, image, question: str, model, processor) -> None:
    pass

def cleanup() -> None:
    pass
