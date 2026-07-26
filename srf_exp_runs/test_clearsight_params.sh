#!/usr/bin/env bash
# =============================================================================
# run_clearsight_vaf.sh — Test ClearSight's VAF method on Qwen-7B
#
# Based on ClearSight paper: α=0.15, β=0.1, middle layers, top 50% heads
# =============================================================================

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR="srf_exp_runs/results/clearsight_vaf"
N_POPE=100  # Quick test

# ClearSight parameters (from paper)
ALPHA=0.15  # Visual enhancement
BETA=0.1    # System suppression
LAYER_START=10
LAYER_END=15
HEAD_TOP_K_PCT=0.50  # Top 50% visual heads

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Testing ClearSight VAF on Qwen-7B (POPE adversarial)"
echo "Parameters: α=${ALPHA}, β=${BETA}, layers ${LAYER_START}-${LAYER_END}"
echo "Top ${HEAD_TOP_K_PCT} visual heads"
echo "============================================================"

# Test VAF method using llava_attn_patch_fixed.py approach
python - <<PYEOF
import sys
sys.path.insert(0, 'my_analysis')
import llava_attn_patch_fixed as patch
import torch
import json

# Setup VAF parameters
patch.patch_model(None, method="srf", enh_para=1.0 + ALPHA, sup_para=1.0 - BETA)
patch._STATE["layer_start"] = LAYER_START
patch._STATE["layer_end"] = LAYER_END

print(f"[VAF] Testing with ClearSight parameters")
print(f"[VAF] α={ALPHA} (enh_para={1.0 + ALPHA})")
print(f"[VAF] β={BETA} (sup_para={1.0 - BETA})")
print(f"[VAF] Layers: {LAYER_START}-{LAYER_END}")
print(f"[VAF] This requires full integration with eval pipeline")
PYEOF

echo "VAF setup complete. Need to integrate with eval pipeline."
