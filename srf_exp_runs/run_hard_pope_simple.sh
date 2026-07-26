#!/bin/bash
# =============================================================================
# Simple Hard POPE Sweep - Test key configurations on hard samples
# =============================================================================

set -euo pipefail

export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/hard_pope_simple"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "HARD POPE SIMPLE SWEEP"
echo "Model: ${MODEL}"
echo "Samples: 100 hard samples"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Baseline on hard samples
echo ""
echo "[1/9] Baseline..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline.log"

# SRF - Post-softmax (Idea 1 from Next_steps.md)
echo ""
echo "[2/9] SRF Post-softmax (alpha=0.15)..."
CUDA_VISIBLE_DEVICES=1 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 0.15 \
    --eps 0.0 \
    --layer_start 10 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/post_softmax_a0.15/" \
    2>&1 | tee "${OUT_DIR}/post_softmax_a0.15.log"

echo ""
echo "[3/9] SRF Post-softmax (alpha=0.5)..."
CUDA_VISIBLE_DEVICES=2 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 0.5 \
    --eps 0.0 \
    --layer_start 10 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/post_softmax_a0.5/" \
    2>&1 | tee "${OUT_DIR}/post_softmax_a0.5.log"

# Layer-specific (Idea 2 from Next_steps.md)
echo ""
echo "[4/9] Layer-Specific (early=5, mid=15)..."
CUDA_VISIBLE_DEVICES=3 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 2.0 \
    --eps 0.0 \
    --layer_start 5 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/layer_specific_5_15/" \
    2>&1 | tee "${OUT_DIR}/layer_specific_5_15.log"

echo ""
echo "[5/9] Layer-Specific (early=8, mid=15)..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 2.0 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/layer_specific_8_15/" \
    2>&1 | tee "${OUT_DIR}/layer_specific_8_15.log"

echo ""
echo "[6/9] Layer-Specific (early=10, mid=20)..."
CUDA_VISIBLE_DEVICES=1 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 2.0 \
    --eps 0.0 \
    --layer_start 10 \
    --layer_end 20 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/layer_specific_10_20/" \
    2>&1 | tee "${OUT_DIR}/layer_specific_10_20.log"

# VAF-like (weaker config from Next_steps.md)
echo ""
echo "[7/9] VAF-like (alpha=0.15, layers 10-15)..."
CUDA_VISIBLE_DEVICES=2 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 0.15 \
    --eps 0.0 \
    --layer_start 10 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/vaf_like/" \
    2>&1 | tee "${OUT_DIR}/vaf_like.log"

# Strong boost
echo ""
echo "[8/9] Strong boost (alpha=2.0, layers 8-15)..."
CUDA_VISIBLE_DEVICES=3 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 2.0 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.2 \
    --output "${OUT_DIR}/strong_boost/" \
    2>&1 | tee "${OUT_DIR}/strong_boost.log"

# With suppression
echo ""
echo "[9/9] With text suppression..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.2 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/with_suppression/" \
    2>&1 | tee "${OUT_DIR}/with_suppression.log"

echo ""
echo "============================================================"
echo "SWEEP COMPLETE!"
echo "Results: ${OUT_DIR}"
echo ""
echo "Compare results:"
echo "  cat ${OUT_DIR}/*/pope.json | jq -s 'map({method: .n, accuracy: .baseline.accuracy})'"
echo "============================================================"
