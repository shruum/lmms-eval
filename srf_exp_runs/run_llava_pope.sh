#!/usr/bin/env bash
# =============================================================================
# run_llava_pope.sh — POPE evaluation on LLaVA-1.5-7B
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/llava_pope"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "POPE Evaluation on LLaVA-1.5-7B"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

echo ""
echo "[1/3] Running Baseline (100 samples, adversarial split)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/baseline/" \
    --pope_splits adversarial \
    --n_pope 100 \
    2>&1 | tee "${OUT_DIR}/baseline_100.log"

echo ""
echo "============================================================"
echo "Baseline test complete! Check results above."
echo "If accuracy is good (~85%+), run full evaluation:"
echo "  bash srf_exp_runs/run_llava_pope_full.sh"
echo "============================================================"
