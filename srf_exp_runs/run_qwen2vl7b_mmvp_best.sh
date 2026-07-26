#!/usr/bin/env bash
# =============================================================================
# run_qwen2vl7b_mmvp_best.sh — MMVP on Qwen2.5-VL-7B-Instruct with best params
# =============================================================================
#
# Best params from 3B autoresearch (44% pair acc):
#   alpha=4.0, eps=0.2, phase=both
#   layer_start=8, layer_end=15 (3B has 28 layers)
#
# For 7B (32 layers), arch params are already in config.py:
#   layer_start=9, layer_end=17 (proportionally scaled)
#   head_top_k_pct=0.20, clip_coarse_grid=7, clip_top_k_pct=0.30
#
# Methods: baseline, SRF (alpha=4.0 eps=0.2), SRF-E (beta=2.0)
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

CONDA_ENV="mllm"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR="srf_exp_runs/results/qwen2vl7b_mmvp"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "MMVP Evaluation on Qwen2.5-VL-7B-Instruct"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Method 1: Baseline
echo ""
echo "[1/3] Running Baseline..."
PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python -u srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline.log"

echo ""
echo "[2/3] Running SRF (alpha=4.0, eps=0.2, phase=both)..."
echo "Arch params from config: layer_start=9, layer_end=17, head_top_k_pct=0.20"
PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    2>&1 | tee "${OUT_DIR}/srf.log"

echo ""
echo "[3/3] Running SRF-E (beta=2.0, alpha=4.0, eps=0.2, phase=both)..."
PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python -u srf/eval.py \
    --method srfe \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/srfe/" \
    --beta 2.0 \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    2>&1 | tee "${OUT_DIR}/srfe.log"

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results: ${OUT_DIR}"
echo ""
echo "Summary:"
echo "  Baseline: ${OUT_DIR}/baseline/mmvp.json"
echo "  SRF:      ${OUT_DIR}/srf/mmvp.json"
echo "  SRF-E:    ${OUT_DIR}/srfe/mmvp.json"
echo "============================================================"
