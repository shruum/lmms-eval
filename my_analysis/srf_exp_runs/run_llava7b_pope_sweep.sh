#!/usr/bin/env bash
# =============================================================================
# run_llava7b_pope_sweep.sh — LLaVA-1.5-7B parameter sweep on POPE
#
# Grid search for best SRF/SRF-E parameters on LLaVA architecture.
# LLaVA has 32 layers (vs Qwen's 28), so we adjust layer ranges accordingly.
# =============================================================================

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/llava_7b_pope_sweep"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "LLaVA-1.5-7B POPE Parameter Sweep"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# LLaVA architecture: 32 layers, middle layers = 10-24
# Conservative grid based on ClearSight paper insights

# Sweep 1: Layer range + alpha
echo ""
echo "[Sweep 1/3] Testing different layer ranges and alpha values..."
for layer_start in 8 10 12; do
    for layer_end in 20 24 28; do
        for alpha in 1.0 2.0 4.0; do
            config="ls${layer_start}_le${layer_end}_a${alpha}"
            echo ""
            echo "Running: ${config}"

            python srf/eval.py \
                --method srf \
                --model "${MODEL}" \
                --datasets pope \
                --output "${OUT_DIR}/${config}/" \
                --alpha ${alpha} \
                --eps 0.2 \
                --phase both \
                --layer_start ${layer_start} \
                --layer_end ${layer_end} \
                2>&1 | tee "${OUT_DIR}/${config}.log"
        done
    done
done

# Sweep 2: Eps values
echo ""
echo "[Sweep 2/3] Testing different eps values..."
for eps in 0.1 0.15 0.2 0.25; do
    config="eps${eps}"
    echo ""
    echo "Running: ${config}"

    python srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/${config}/" \
        --alpha 2.0 \
        --eps ${eps} \
        --phase both \
        --layer_start 10 \
        --layer_end 24 \
        2>&1 | tee "${OUT_DIR}/${config}.log"
done

# Sweep 3: SRF-E beta values
echo ""
echo "[Sweep 3/3] Testing SRF-E with different beta values..."
for beta in 0.5 1.0 1.5 2.0; do
    config="srfe_beta${beta}"
    echo ""
    echo "Running: ${config}"

    python srf/eval.py \
        --method srfe \
        --model "${MODEL}" \
        --datasets pope \
        --output "${OUT_DIR}/${config}/" \
        --beta ${beta} \
        2>&1 | tee "${OUT_DIR}/${config}.log"
done

echo ""
echo "============================================================"
echo "Grid search complete!"
echo "Results: ${OUT_DIR}"
echo "Analyze results to find best configuration."
echo "============================================================"
