#!/usr/bin/env bash
# =============================================================================
# Fast Grid Search Launcher - LLaVA SRF Parameters
# =============================================================================

set -euo pipefail

echo "🚀 FAST GRID SEARCH FOR LLAVA"
echo "========================================"
echo "Stage 1: Coarse Search"
echo "  18 experiments total (~1.5-3 hours on 2 GPUs)"
echo "  9 experiments per GPU"
echo ""
echo "Testing:"
echo "  • 3 layer ranges (8-12, 10-16, 12-18)"
echo "  • 3 boost/suppress ratios (1.5/3.0, 2.0/4.0, 2.5/5.0)"
echo "  • 2 saliency methods (CLIP vs DINO)"
echo "========================================"
echo ""

cd ~/shruthi/lmms-eval/my_analysis
mkdir -p fast_grid_search

# Launch on GPU 2
echo "Starting GPU 2 (physical)..."
~/miniconda3/envs/mllm/bin/python fast_grid_search_llava.py \
    --gpu 0 \
    --total_gpus 2 \
    --physical_gpu 2 \
    --max_experiments 9 \
    --output_dir fast_grid_search \
    --stage 1 \
    > fast_grid_search/gpu_2.log 2>&1 &

echo "  GPU 2 launched (PID: $!)"

sleep 2

# Launch on GPU 3
echo "Starting GPU 3 (physical)..."
~/miniconda3/envs/mllm/bin/python fast_grid_search_llava.py \
    --gpu 1 \
    --total_gpus 2 \
    --physical_gpu 3 \
    --max_experiments 9 \
    --output_dir fast_grid_search \
    --stage 1 \
    > fast_grid_search/gpu_3.log 2>&1 &

echo "  GPU 3 launched (PID: $!)"
echo ""

echo "========================================"
echo "🎯 RUNNING 18 EXPERIMENTS ON 2 GPUs"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f fast_grid_search/gpu_2.log"
echo "  tail -f fast_grid_search/gpu_3.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Expected completion: 1.5-3 hours"
echo ""

# Wait for completion
wait

echo ""
echo "========================================"
echo "✅ GRID SEARCH COMPLETE!"
echo "========================================"
echo ""
echo "Analyze results:"
echo "  python fast_grid_search_llava.py --output_dir fast_grid_search"
echo ""
echo "Check best configs:"
echo "  cat fast_grid_search/results_gpu*_final.json"
