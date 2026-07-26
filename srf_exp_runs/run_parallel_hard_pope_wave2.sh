#!/bin/bash
# =============================================================================
# Parallel Hard POPE Wave 2 - Different configurations
# =============================================================================

set -euo pipefail

export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/parallel_hard_pope_wave2"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "PARALLEL HARD POPE WAVE 2 - 8 GPUs"
echo "Testing: Different layer ranges, head %, alphas"
echo "============================================================"

# GPU 0 - Layers 5-10
echo "Launching GPU 0: Layers 5-10..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.0 \
    --layer_start 5 \
    --layer_end 10 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu0_layers_5_10/" \
    > "${OUT_DIR}/gpu0_layers_5_10.log" 2>&1 &
echo "  GPU 0 PID: $!"

# GPU 1 - Layers 15-25
echo "Launching GPU 1: Layers 15-25..."
CUDA_VISIBLE_DEVICES=1 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.0 \
    --layer_start 15 \
    --layer_end 25 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu1_layers_15_25/" \
    > "${OUT_DIR}/gpu1_layers_15_25.log" 2>&1 &
echo "  GPU 1 PID: $!"

# GPU 2 - Head 30%
echo "Launching GPU 2: Head 30%..."
CUDA_VISIBLE_DEVICES=2 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.3 \
    --output "${OUT_DIR}/gpu2_head_30/" \
    > "${OUT_DIR}/gpu2_head_30.log" 2>&1 &
echo "  GPU 2 PID: $!"

# GPU 3 - Head 80%
echo "Launching GPU 3: Head 80%..."
CUDA_VISIBLE_DEVICES=3 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.8 \
    --output "${OUT_DIR}/gpu3_head_80/" \
    > "${OUT_DIR}/gpu3_head_80.log" 2>&1 &
echo "  GPU 3 PID: $!"

# GPU 4 - Alpha 0.3
echo "Launching GPU 4: Alpha 0.3..."
CUDA_VISIBLE_DEVICES=4 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 0.3 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu4_alpha_0.3/" \
    > "${OUT_DIR}/gpu4_alpha_0.3.log" 2>&1 &
echo "  GPU 4 PID: $!"

# GPU 5 - Alpha 4.0
echo "Launching GPU 5: Alpha 4.0..."
CUDA_VISIBLE_DEVICES=5 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 4.0 \
    --eps 0.0 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu5_alpha_4.0/" \
    > "${OUT_DIR}/gpu5_alpha_4.0.log" 2>&1 &
echo "  GPU 5 PID: $!"

# GPU 6 - With suppression
echo "Launching GPU 6: With suppression..."
CUDA_VISIBLE_DEVICES=6 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu6_with_suppression/" \
    > "${OUT_DIR}/gpu6_with_suppression.log" 2>&1 &
echo "  GPU 6 PID: $!"

# GPU 7 - Epsilon 0.5
echo "Launching GPU 7: Epsilon 0.5..."
CUDA_VISIBLE_DEVICES=7 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.5 \
    --layer_start 8 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu7_eps_0.5/" \
    > "${OUT_DIR}/gpu7_eps_0.5.log" 2>&1 &
echo "  GPU 7 PID: $!"

echo ""
echo "============================================================"
echo "WAVE 2 LAUNCHED - 8 more experiments"
echo "============================================================"
