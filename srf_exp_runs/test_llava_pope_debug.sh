#!/usr/bin/env bash
# =============================================================================
# test_llava_pope_debug.sh — Test LLaVA-1.5-7B on POPE (debug mode)
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/llava_debug"
N_SAMPLES=100

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Testing LLaVA-1.5-7B on POPE (${N_SAMPLES} samples)"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Test 1: Baseline
echo ""
echo "[Test 1/3] Baseline (${N_SAMPLES} samples)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/baseline/" \
    --n_pope ${N_SAMPLES} \
    --alpha 0.0 \
    2>&1 | tee "${OUT_DIR}/baseline.log"

# Test 2: SRF
echo ""
echo "[Test 2/3] SRF alpha=1.0 eps=0.1 (${N_SAMPLES} samples)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/srf_test/" \
    --n_pope ${N_SAMPLES} \
    --alpha 1.0 \
    --eps 0.1 \
    --phase both \
    2>&1 | tee "${OUT_DIR}/srf_test.log"

# Test 3: SRF-E
echo ""
echo "[Test 3/3] SRF-E beta=1.0 (${N_SAMPLES} samples)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srfe \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/srfe_test/" \
    --n_pope ${N_SAMPLES} \
    --beta 1.0 \
    2>&1 | tee "${OUT_DIR}/srfe_test.log"

echo ""
echo "============================================================"
echo "Debug tests complete! Check logs for errors."
echo "============================================================"
