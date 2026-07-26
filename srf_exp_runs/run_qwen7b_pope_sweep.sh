#!/usr/bin/env bash
# =============================================================================
# run_qwen7b_pope_sweep.sh — Qwen-7B POPE parameter sweep
#
# Tests different layer ranges and head selection percentages to find best SRF params
# Runs on GPU 0
# =============================================================================

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

CONDA_ENV="mllm"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUT_DIR="srf_exp_runs/results/qwen7b_sweep"
N_POPE=100  # Quick sweep with 100 samples

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Qwen-7B POPE Parameter Sweep (GPU ${CUDA_VISIBLE_DEVICES})"
echo "Model: ${MODEL}"
echo "Samples per config: ${N_POPE}"
echo "Output: ${OUT_DIR}/"
echo "============================================================"

# Parameter grid to test
# Layer ranges: start from 8/10/12, end at 16/18/20
# Head percentages: 15%, 20%, 25%

declare -a CONFIGS=(
    # Format: "layer_start layer_end head_top_k_pct"
    "8 16 0.15"
    "8 16 0.20"
    "8 18 0.20"
    "10 18 0.20"
    "10 20 0.20"
    "8 20 0.15"
    "8 20 0.25"
)

echo ""
echo "Running ${#CONFIGS[@]} SRF configurations..."
echo ""

# Run SRF sweep
for config in "${CONFIGS[@]}"; do
    read -r layer_start layer_end head_top_k_pct <<< "${config}"

    config_tag="ls${layer_start}_le${layer_end}_hk${head_top_k_pct}"
    log_file="${OUT_DIR}/srf_${config_tag}.log"

    echo "[$(date '+%H:%M:%S')] SRF config: ${config_tag}"

    conda run -n "${CONDA_ENV}" python srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope \
        --pope_splits adversarial \
        --n_pope "${N_POPE}" \
        --output "${OUT_DIR}/srf_${config_tag}" \
        --layer_start "${layer_start}" \
        --layer_end "${layer_end}" \
        --head_top_k_pct "${head_top_k_pct}" \
        --clip_coarse_grid 7 \
        --clip_top_k_pct 0.30 \
        --clip_fallback_thresh 0.20 \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed - check ${log_file}"
    fi
done

# Run SRF-E sweep with beta values (using middle-range SRF params)
echo ""
echo "[$(date '+%H:%M:%S')] Running SRF-E beta sweep..."
echo ""

declare -a BETAS=(0.5 1.0 2.0)

for beta in "${BETAS[@]}"; do
    config_tag="srfe_beta${beta}"
    log_file="${OUT_DIR}/${config_tag}.log"

    echo "[$(date '+%H:%M:%S')] SRF-E beta=${beta}"

    conda run -n "${CONDA_ENV}" python srf/eval.py \
        --method srfe \
        --model "${MODEL}" \
        --datasets pope \
        --pope_splits adversarial \
        --beta "${beta}" \
        --n_pope "${N_POPE}" \
        --output "${OUT_DIR}/${config_tag}" \
        --layer_start 8 \
        --layer_end 18 \
        --head_top_k_pct 0.20 \
        --clip_coarse_grid 7 \
        --clip_top_k_pct 0.30 \
        --clip_fallback_thresh 0.20 \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed - check ${log_file}"
    fi
done

echo ""
echo "============================================================"
echo "SWEEP COMPLETE!"
echo "Results saved to: ${OUT_DIR}/"
echo "Check logs to find the best configuration"
echo "============================================================"
