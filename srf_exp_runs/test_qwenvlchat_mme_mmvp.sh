#!/usr/bin/env bash
# =============================================================================
# test_qwenvlchat_mme_mmvp.sh — Test Qwen-VL-Chat on MME/MMVP (few samples)
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export HF_HOME="/home/anna2/.cache/huggingface"

CONDA_ENV="mllm"
MODEL="Qwen/Qwen-VL-Chat"
OUT_DIR="srf_exp_runs/results/qwenvlchat_test"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "TEST: Qwen-VL-Chat on MME/MMVP (limited samples)"
echo "Model: ${MODEL}"
echo "============================================================"

echo ""
echo "[1/4] Testing MMVP Baseline (10 pairs)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline_mmvp.log" | tail -30

echo ""
echo "[2/4] Testing MMVP SRF (10 pairs)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    --layer_end 17 \
    2>&1 | tee "${OUT_DIR}/srf_mmvp.log" | tail -30

echo ""
echo "[3/4] Testing MME Baseline (20 samples)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mme \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline_mme.log" | tail -30

echo ""
echo "[4/4] Testing MME SRF (20 samples)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mme \
    --output "${OUT_DIR}/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    --layer_end 17 \
    2>&1 | tee "${OUT_DIR}/srf_mme.log" | tail -30

echo ""
echo "============================================================"
echo "Test complete! Check results above."
echo "If working, run full: bash srf_exp_runs/run_qwenvlchat_mme_mmvp.sh"
echo "============================================================"
