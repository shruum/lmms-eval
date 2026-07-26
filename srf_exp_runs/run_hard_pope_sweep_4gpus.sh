#!/bin/bash
# =============================================================================
# Hard POPE Sample Sweep - GPUs 0-3
# Tests 193 SRF configurations on 100 hard samples
# =============================================================================

set -euo pipefail

# Environment
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
OUT_DIR="srf_exp_runs/results/hard_pope_sweep_4gpus"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "HARD POPE SWEEP - GPUs 0-3"
echo "Configurations: 193"
echo "Samples: 100 hard samples"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Launch 4 parallel workers
for GPU_ID in 0 1 2 3; do
    echo ""
    echo "Launching GPU ${GPU_ID} worker..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n ${CONDA_ENV} nohup python -u srf/sweep_hard_pope_worker.py \
        --hard_samples srf/hard_samples_pope.json \
        --model llava-hf/llava-1.5-7b-hf \
        --configs "${OUT_DIR}/gpu_${GPU_ID}_configs.json" \
        --output "${OUT_DIR}/gpu_${GPU_ID}_results.json" \
        --gpu_id ${GPU_ID} \
        > "${OUT_DIR}/gpu_${GPU_ID}.log" 2>&1 &

    echo "  GPU ${GPU_ID}: PID $!"
done

echo ""
echo "============================================================"
echo "All 4 workers launched!"
echo "Monitor with: tail -f ${OUT_DIR}/gpu_*.log"
echo "============================================================"

# Create simple monitoring script
cat > "${OUT_DIR}/monitor.sh" << 'EOF'
#!/bin/bash
echo "Hard POPE Sweep - GPU Status"
echo "================================"
for gpu in 0 1 2 3; do
    echo ""
    echo "GPU ${gpu}:"
    if [ -f "gpu_${gpu}.log" ]; then
        tail -3 "gpu_${gpu}.log"
    else
        echo "  No log file yet"
    fi
done
echo ""
echo "Results so far:"
ls -1 gpu_*_results.json 2>/dev/null | wc -l
echo "of 4 expected result files"
EOF
chmod +x "${OUT_DIR}/monitor.sh"

echo ""
echo "Monitor: ${OUT_DIR}/monitor.sh"
echo ""
