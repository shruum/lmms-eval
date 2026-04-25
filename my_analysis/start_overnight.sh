#!/usr/bin/env bash
# =============================================================================
# Start Overnight Experiments - Launch multiple model investigations in parallel
#
# This will run experiments all night on different GPUs.
# Results are automatically saved incrementally.
#
# Usage:
#   ./start_overnight.sh        # Run all 3 GPUs
#   ./start_overnight.sh 0      # Run only GPU 0
# =============================================================================

set -euo pipefail

PYTHON_PATH="/home/anna2/miniconda3/envs/mllm/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Function to run experiments on a GPU
run_gpu() {
    local gpu_id=$1
    local model=$2
    local arch=$3
    local focus=$4

    echo "[$(date '+%H:%M:%S')] Starting GPU ${gpu_id}: ${model} (focus: ${focus})"

    nohup ${PYTHON_PATH} overnight_experiments.py \
        --gpu ${gpu_id} \
        --model "${model}" \
        --arch ${arch} \
        --focus ${focus} \
        --n_samples 100 \
        --max_experiments 50 \
        > "overnight_gpu${gpu_id}.log" 2>&1 &

    echo "[$(date '+%H:%M:%S')] GPU ${gpu_id} started (PID: $!)"
    echo "   Log: overnight_gpu${gpu_id}.log"
}

# =============================================================================
# Configuration
# =============================================================================

# GPU 0: Qwen-7B (explore around layers 14-20 - our best finding)
GPU_0_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
GPU_0_ARCH="qwen"
GPU_0_FOCUS="late"

# GPU 1: Qwen-3B (explore around layers 8-14 - validate 3B works well too)
GPU_1_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
GPU_1_ARCH="qwen"
GPU_1_FOCUS="mid"

# GPU 2: LLaVA (explore around layers 10-16)
GPU_2_MODEL="llava-hf/llava-1.5-7b-hf"
GPU_2_ARCH="llava"
GPU_2_FOCUS="standard"

# =============================================================================
# Launch
# =============================================================================

echo "🌙 STARTING OVERNIGHT EXPERIMENTS"
echo "=================================="
echo "Start time: $(date)"
echo ""

# Check which GPUs to run
if [[ $# -eq 0 ]]; then
    # Run all GPUs
    echo "Launching all 3 GPUs..."
    run_gpu 0 "${GPU_0_MODEL}" ${GPU_0_ARCH} ${GPU_0_FOCUS}
    sleep 5
    run_gpu 1 "${GPU_1_MODEL}" ${GPU_1_ARCH} ${GPU_1_FOCUS}
    sleep 5
    run_gpu 2 "${GPU_2_MODEL}" ${GPU_2_ARCH} ${GPU_2_FOCUS}
else
    # Run specific GPU
    gpu_num=$1
    echo "Launching GPU ${gpu_num} only..."

    case $gpu_num in
        0)
            run_gpu 0 "${GPU_0_MODEL}" ${GPU_0_ARCH} ${GPU_0_FOCUS}
            ;;
        1)
            run_gpu 1 "${GPU_1_MODEL}" ${GPU_1_ARCH} ${GPU_1_FOCUS}
            ;;
        2)
            run_gpu 2 "${GPU_2_MODEL}" ${GPU_2_ARCH} ${GPU_2_FOCUS}
            ;;
        *)
            echo "Error: Invalid GPU number. Use 0, 1, 2, or no argument for all."
            exit 1
            ;;
    esac
fi

echo ""
echo "=================================="
echo "[$(date '+%H:%M:%S')] All experiments started!"
echo ""
echo "Monitor progress with:"
echo "  tail -f overnight_gpu0.log"
echo "  tail -f overnight_gpu1.log"
echo "  tail -f overnight_gpu2.log"
echo ""
echo "Check results in the morning:"
echo "  cat overnight_late_Qwen2.5-VL-7B-Instruct/results_partial.json | grep improvement"
echo "  cat overnight_mid_Qwen2.5-VL-3B-Instruct/results_partial.json | grep improvement"
echo "  cat overnight_standard_llava-1.5-7b-hf/results_partial.json | grep improvement"
echo ""
echo "Each GPU will run ~50 experiments (~8-10 hours)"
echo "=================================="
