#!/usr/bin/env python3
"""
Single test experiment for LLaVA to validate the setup.

Tests just one configuration:
- Layers 8-12
- Boost 1.5, Suppress 3.0 (gentle intervention)
- CLIP saliency (7x7 grid, top-30%)
"""

import os
import sys
from pathlib import Path

# Set environment BEFORE any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/anna2/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/anna2/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/home/anna2/.cache/huggingface/hub"

# Add paths
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch")

print("=" * 60)
print("Single LLaVA Test Experiment")
print("=" * 60)
print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"HF_HOME: {os.environ['HF_HOME']}")
print()

# Test configuration
test_config = {
    "layer_start": 8,
    "layer_end": 12,
    "boost_alpha": 1.5,
    "suppress_alpha": 3.0,
    "head_top_k_pct": 0.20,
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.30,
    "clip_use_soft": True,
    "saliency_method": "clip",
}

print("Test Configuration:")
print(f"  Layers: {test_config['layer_start']}-{test_config['layer_end']}")
print(f"  Boost/Suppress: {test_config['boost_alpha']}/{test_config['suppress_alpha']}")
print(f"  Head top-k: {test_config['head_top_k_pct']}")
print(f"  Saliency: CLIP {test_config['clip_coarse_grid']}x{test_config['clip_coarse_grid']}, top-{test_config['clip_top_k_pct']*100:.0f}%")
print()

# Import and modify SRF config
print("Importing modules...")
from pope_srf_eval import main, SRF, ARCH_DEFAULTS

print("Updating SRF configuration...")
SRF.update(test_config)
ARCH_DEFAULTS["llava"]["layer_start"] = test_config["layer_start"]
ARCH_DEFAULTS["llava"]["layer_end"] = test_config["layer_end"]

print("Starting evaluation...")
print("-" * 60)

# Set up command line args
sys.argv = [
    "test_single_llava.py",
    "--arch", "llava",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--n_samples", "20",  # Small sample for quick test
    "--mode", "both",
    "--output", "/home/anna2/shruthi/lmms-eval/my_analysis/test_result.json",
]

# Run
try:
    main()
    print("-" * 60)
    print("✅ Test completed successfully!")
    print(f"Results saved to: test_result.json")
except Exception as e:
    print("-" * 60)
    print(f"❌ Test failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
