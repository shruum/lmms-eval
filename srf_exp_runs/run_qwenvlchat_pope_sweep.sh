#!/usr/bin/env bash
# =============================================================================
# run_qwenvlchat_pope_sweep.sh — POPE evaluation on Qwen-VL-Chat with param sweep
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="Qwen/Qwen-VL-Chat"
OUT_DIR="srf_exp_runs/results/qwenvlchat_pope_sweep"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "POPE Parameter Sweep on Qwen-VL-Chat"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# ── Baseline ────────────────────────────────────────────────────────────────
echo ""
echo "[Step 1/3] Running Baseline..."
conda run -n ${CONDA_ENV} python srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets pope \
    --output "${OUT_DIR}/baseline/" \
    2>&1 | tee "${OUT_DIR}/baseline.log"

# ── SRF Parameter Sweep ──────────────────────────────────────────────────────
echo ""
echo "[Step 2/3] SRF Parameter Sweep..."
echo "Sweeping alpha, eps, and layer_end..."

# Alpha sweep (keep eps=0.2, layer_end=17)
for alpha in 2.0 4.0 6.0 8.0; do
    echo ""
    echo "  SRF: alpha=${alpha}, eps=0.2, layer_end=17"
    conda run -n ${CONDA_ENV} python srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/srf_alpha${alpha}/" \
        --alpha ${alpha} \
        --eps 0.2 \
        --phase both \
        --layer_end 17 \
        2>&1 | tee "${OUT_DIR}/srf_alpha${alpha}.log"
done

# Epsilon sweep (keep alpha=4.0, layer_end=17)
for eps in 0.1 0.2 0.3 0.5; do
    echo ""
    echo "  SRF: alpha=4.0, eps=${eps}, layer_end=17"
    conda run -n ${CONDA_ENV} python srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/srf_eps${eps}/" \
        --alpha 4.0 \
        --eps ${eps} \
        --phase both \
        --layer_end 17 \
        2>&1 | tee "${OUT_DIR}/srf_eps${eps}.log"
done

# Layer end sweep (keep alpha=4.0, eps=0.2)
for layer_end in 15 17 19 21; do
    echo ""
    echo "  SRF: alpha=4.0, eps=0.2, layer_end=${layer_end}"
    conda run -n ${CONDA_ENV} python srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/srf_layer${layer_end}/" \
        --alpha 4.0 \
        --eps 0.2 \
        --phase both \
        --layer_end ${layer_end} \
        2>&1 | tee "${OUT_DIR}/srf_layer${layer_end}.log"
done

# ── SRF-E Parameter Sweep ────────────────────────────────────────────────────
echo ""
echo "[Step 3/3] SRF-E Parameter Sweep..."
echo "Sweeping beta values..."

for beta in 0.5 1.0 2.0 3.0 4.0; do
    echo ""
    echo "  SRF-E: beta=${beta}"
    conda run -n ${CONDA_ENV} python srf/eval.py \
        --method srfe \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/srfe_beta${beta}/" \
        --beta ${beta} \
        2>&1 | tee "${OUT_DIR}/srfe_beta${beta}.log"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "Results: ${OUT_DIR}"
echo ""
echo "To compare results:"
echo "  cat ${OUT_DIR}/*/pope.json | jq -s 'map({method: .n, accuracy: .baseline.accuracy})'"
echo "============================================================"
