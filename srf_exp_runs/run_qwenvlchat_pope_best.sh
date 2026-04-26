#!/usr/bin/env bash
# =============================================================================
# run_qwenvlchat_pope_best.sh — POPE on Qwen-VL-Chat with best known params
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="Qwen/Qwen-VL-Chat"
OUT_DIR="srf_exp_runs/results/qwenvlchat_pope"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Full POPE Evaluation on Qwen-VL-Chat"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# # Method 1: Baseline
# echo ""
# echo "[1/3] Running Baseline..."
# PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python srf/eval.py \
#     --method baseline \
#     --model "${MODEL}" \
#     --datasets pope \
#     --output "${OUT_DIR}/baseline/" \
#     2>&1 | tee "${OUT_DIR}/baseline.log"

# echo ""
# echo "[2/3] Running SRF (alpha=4.0, eps=0.2, phase=both, layer_end=17)..."
# PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python srf/eval.py \
#     --method srf \
#     --model "${MODEL}" \
#     --datasets pope \
#     --output "${OUT_DIR}/srf/" \
#     --alpha 4.0 \
#     --eps 0.2 \
#     --phase both \
#     --layer_end 17 \
#     2>&1 | tee "${OUT_DIR}/srf.log"

# echo ""
echo "[3/3] Running SRF-E (beta=2.0)..."
PYTHONUNBUFFERED=1 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srfe \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/srfe/" \
    --beta 2.0 \
    2>&1 | tee "${OUT_DIR}/srfe.log"

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results: ${OUT_DIR}"
echo "============================================================"
