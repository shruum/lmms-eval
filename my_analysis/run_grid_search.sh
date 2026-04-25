#!/usr/bin/env bash
# =============================================================================
# Grid Search Launcher for LLaVA SRF Parameters
# =============================================================================

set -euo pipefail

# Configuration
CONDA_ENV="mllm"
PYTHON_PATH="$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/grid_search_results"
N_GPUS=2  # Number of GPUs to use
MAX_EXPERIMENTS=50  # Experiments per GPU

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "LLaVA Grid Search Launcher"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "GPUs: ${N_GPUS}"
echo "Max experiments per GPU: ${MAX_EXPERIMENTS}"
echo "========================================"

# Launch experiments on each GPU
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    echo "Launching GPU ${gpu_id}..."

    "${PYTHON_PATH}" "${SCRIPT_DIR}/grid_search_llava.py" \
        --gpu ${gpu_id} \
        --total_gpus ${N_GPUS} \
        --max_experiments ${MAX_EXPERIMENTS} \
        --output_dir "${OUTPUT_DIR}" \
        --conda_env "${CONDA_ENV}" \
        > "${OUTPUT_DIR}/gpu_${gpu_id}.log" 2>&1 &

    echo "GPU ${gpu_id} launched (PID: $!)"
done

echo ""
echo "========================================"
echo "All ${N_GPUS} GPUs launched!"
echo "========================================"
echo "Monitor progress with:"
echo "  tail -f ${OUTPUT_DIR}/gpu_0.log"
echo "  tail -f ${OUTPUT_DIR}/gpu_1.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_DIR}/results_gpu*_final.json"
echo ""

# Wait for all background processes
wait

echo ""
echo "========================================"
echo "Grid search complete!"
echo "========================================"
echo "Analyze results with:"
echo "  python ${SCRIPT_DIR}/analyze_grid_search.py --output_dir ${OUTPUT_DIR}"
