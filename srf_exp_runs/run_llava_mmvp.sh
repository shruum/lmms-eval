#!/bin/bash
# =============================================================================
# LLAVA MMVP Baseline + SRF Evaluation
# =============================================================================

set -euo pipefail

export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/llava_mmvp"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "LLAVA MMVP BASELINE + SRF EVALUATION"
echo "Model: ${MODEL}"
echo "Dataset: MMVP (150 pairs, 300 images, full)"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Baseline
echo ""
echo "Step 1/2: Running Baseline..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} python -u srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets mmvp \
    --output "${OUT_DIR}/baseline/" \
    | tee "${OUT_DIR}/baseline.log"

echo ""
echo "Step 2/2: Running SRF (alpha=4.0, eps=0.2, phase=both)..."
echo "Using best config from Qwen autoresearch (layer 8-15, boost_alpha=4.0)"
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets mmvp \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.20 \
    --output "${OUT_DIR}/srf/" \
    | tee "${OUT_DIR}/srf.log"

echo ""
echo "============================================================"
echo "EXPERIMENTS COMPLETE!"
echo ""
echo "Results:"
echo "  Baseline: ${OUT_DIR}/baseline/mmvp.json"
echo "  SRF:      ${OUT_DIR}/srf/mmvp.json"
echo ""
echo "Logs:"
echo "  Baseline: ${OUT_DIR}/baseline.log"
echo "  SRF:      ${OUT_DIR}/srf.log"
echo "============================================================"

# Quick summary
if [ -f "${OUT_DIR}/baseline/mmvp.json" ] && [ -f "${OUT_DIR}/srf/mmvp.json" ]; then
    echo ""
    echo "QUICK SUMMARY:"
    python3 - <<EOF
import json
with open("${OUT_DIR}/baseline/mmvp.json") as f:
    base = json.load(f)
with open("${OUT_DIR}/srf/mmvp.json") as f:
    srf = json.load(f)

base_acc = base.get('baseline_pair', 0) * 100
srf_acc = srf.get('method_pair', {}).get('0.0', base.get('baseline_pair', 0)) * 100
delta = srf_acc - base_acc

print(f"  Baseline: {base_acc:.2f}%")
print(f"  SRF:      {srf_acc:.2f}%")
print(f"  Delta:    {delta:+.2f}%")
EOF
fi
