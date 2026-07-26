#!/bin/bash
# =============================================================================
# Parallel Hard POPE Sweep - All 8 GPUs simultaneously
# =============================================================================

set -euo pipefail

export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUT_DIR="srf_exp_runs/results/parallel_hard_pope"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "PARALLEL HARD POPE SWEEP - 8 GPUs"
echo "Model: ${MODEL}"
echo "Samples: 100 hard samples per experiment"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Launch 8 experiments in parallel on different GPUs

# GPU 0 - Baseline
echo "Launching GPU 0: Baseline..."
CUDA_VISIBLE_DEVICES=0 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method baseline \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output "${OUT_DIR}/gpu0_baseline/" \
    > "${OUT_DIR}/gpu0_baseline.log" 2>&1 &
echo "  GPU 0 PID: $!"

# GPU 1 - Post-softmax alpha=0.15
echo "Launching GPU 1: Post-softmax (alpha=0.15)..."
CUDA_VISIBLE_DEVICES=1 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu1_post_a015/" \
    > "${OUT_DIR}/gpu1_post_a015.log" 2>&1 &
echo "  GPU 1 PID: $!"

# GPU 2 - Post-softmax alpha=0.5
echo "Launching GPU 2: Post-softmax (alpha=0.5)..."
CUDA_VISIBLE_DEVICES=2 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu2_post_a05/" \
    > "${OUT_DIR}/gpu2_post_a05.log" 2>&1 &
echo "  GPU 2 PID: $!"

# GPU 3 - Post-softmax alpha=1.0
echo "Launching GPU 3: Post-softmax (alpha=1.0)..."
CUDA_VISIBLE_DEVICES=3 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
    --method srf \
    --model "${MODEL}" \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --alpha 1.0 \
    --eps 0.0 \
    --layer_start 10 \
    --layer_end 15 \
    --head_top_k_pct 0.5 \
    --output "${OUT_DIR}/gpu3_post_a10/" \
    > "${OUT_DIR}/gpu3_post_a10.log" 2>&1 &
echo "  GPU 3 PID: $!"

# GPU 4 - Layer-specific 8-15
echo "Launching GPU 4: Layer-specific (8-15)..."
CUDA_VISIBLE_DEVICES=4 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu4_layer_8_15/" \
    > "${OUT_DIR}/gpu4_layer_8_15.log" 2>&1 &
echo "  GPU 4 PID: $!"

# GPU 5 - Layer-specific 10-20
echo "Launching GPU 5: Layer-specific (10-20)..."
CUDA_VISIBLE_DEVICES=5 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu5_layer_10_20/" \
    > "${OUT_DIR}/gpu5_layer_10_20.log" 2>&1 &
echo "  GPU 5 PID: $!"

# GPU 6 - VAF-like (weak boost)
echo "Launching GPU 6: VAF-like..."
CUDA_VISIBLE_DEVICES=6 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu6_vaf_like/" \
    > "${OUT_DIR}/gpu6_vaf_like.log" 2>&1 &
echo "  GPU 6 PID: $!"

# GPU 7 - Strong boost
echo "Launching GPU 7: Strong boost..."
CUDA_VISIBLE_DEVICES=7 conda run -n ${CONDA_ENV} nohup python -u srf/eval.py \
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
    --output "${OUT_DIR}/gpu7_strong/" \
    > "${OUT_DIR}/gpu7_strong.log" 2>&1 &
echo "  GPU 7 PID: $!"

echo ""
echo "============================================================"
echo "ALL 8 EXPERIMENTS LAUNCHED IN PARALLEL!"
echo "Monitor individual logs:"
echo "  tail -f ${OUT_DIR}/gpu*.log"
echo ""
echo "Monitor GPU utilization:"
echo "  watch -n 1 nvidia-smi"
echo "============================================================"

# Create monitoring script
cat > "${OUT_DIR}/monitor.sh" << 'EOF'
#!/bin/bash
echo "============================================"
echo "Parallel Hard POPE Sweep - Status"
echo "============================================"
echo ""
echo "GPU Utilization:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "GPU %s: %s%% utilized, %sMiB/%sMiB\n", $1, $2, $3, $4}'
echo ""
echo "Experiment Progress:"
for gpu in 0 1 2 3 4 5 6 7; do
    if [ -f "gpu${gpu}_baseline.log" ]; then
        log="gpu${gpu}_baseline.log"
    elif [ -f "gpu${gpu}_post_a015.log" ]; then
        log="gpu${gpu}_post_a015.log"
    elif [ -f "gpu${gpu}_post_a05.log" ]; then
        log="gpu${gpu}_post_a05.log"
    elif [ -f "gpu${gpu}_post_a10.log" ]; then
        log="gpu${gpu}_post_a10.log"
    elif [ -f "gpu${gpu}_layer_8_15.log" ]; then
        log="gpu${gpu}_layer_8_15.log"
    elif [ -f "gpu${gpu}_layer_10_20.log" ]; then
        log="gpu${gpu}_layer_10_20.log"
    elif [ -f "gpu${gpu}_vaf_like.log" ]; then
        log="gpu${gpu}_vaf_like.log"
    elif [ -f "gpu${gpu}_strong.log" ]; then
        log="gpu${gpu}_strong.log"
    else
        continue
    fi

    echo "GPU ${gpu}:"
    tail -3 "$log" | grep -E "\[.*\]|Accuracy|Saved" | tail -1
done
echo ""
echo "Completed experiments:"
ls -1 gpu_*_baseline/pope.json gpu_*/pope.json 2>/dev/null | wc -l
echo "of 8 expected"
EOF
chmod +x "${OUT_DIR}/monitor.sh"

echo ""
echo "Quick monitoring: ${OUT_DIR}/monitor.sh"
echo ""
