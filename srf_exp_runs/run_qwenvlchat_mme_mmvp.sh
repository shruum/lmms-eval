#!/usr/bin/env bash
# =============================================================================
# run_qwenvlchat_mme_mmvp.sh — MME and MMVP evaluation on Qwen-VL-Chat
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="Qwen/Qwen-VL-Chat"
OUT_DIR="srf_exp_runs/results/qwenvlchat_mme_mmvp"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "MME and MMVP Evaluation on Qwen-VL-Chat"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# ── Baseline ────────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Running Baseline on MMVP..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline_mmvp.log"

echo ""
echo "[2/6] Running Baseline on MME..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mme \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline_mme.log"

# ── SRF (best params from config) ────────────────────────────────────────────
echo ""
echo "[3/6] Running SRF on MMVP (alpha=4.0, eps=0.2, phase=both)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    --layer_end 17 \
    2>&1 | tee "${OUT_DIR}/srf_mmvp.log"

echo ""
echo "[4/6] Running SRF on MME (alpha=4.0, eps=0.2, phase=both)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mme \
    --output "${OUT_DIR}/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    --layer_end 17 \
    2>&1 | tee "${OUT_DIR}/srf_mme.log"

# ── SRF-E (beta=2.0) ────────────────────────────────────────────────────────
echo ""
echo "[5/6] Running SRF-E on MMVP (beta=2.0)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srfe \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/srfe/" \
    --beta 2.0 \
    2>&1 | tee "${OUT_DIR}/srfe_mmvp.log"

echo ""
echo "[6/6] Running SRF-E on MME (beta=2.0)..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method srfe \
    --model "${MODEL}" \
    --datasets mme \
    --output "${OUT_DIR}/srfe/" \
    --beta 2.0 \
    2>&1 | tee "${OUT_DIR}/srfe_mme.log"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results: ${OUT_DIR}"
echo ""
echo "Quick summary:"
echo "  MMVP baseline:  $(cat ${OUT_DIR}/baseline/summary.json | jq -r '.mmvp.baseline_pair // .mmvp.baseline_img // .mmvp // "N/A" 2>/dev/null)"
echo "  MME baseline:   $(cat ${OUT_DIR}/baseline/summary.json | jq -r '.mme.baseline_acc // "N/A" 2>/dev/null)"
echo "============================================================"
