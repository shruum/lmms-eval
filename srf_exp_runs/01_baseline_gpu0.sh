#!/bin/bash
# Script 1: LLaVA-1.5-7B-HF POPE Baseline - GPU 0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export CUDA_VISIBLE_DEVICES=0

echo "=================================================="
echo "LLaVA-1.5-7B-HF POPE BASELINE"
echo "GPU: 0"
echo "=================================================="

python srf/eval.py \
    --method baseline \
    --model "llava-hf/llava-1.5-7b-hf" \
    --datasets pope \
    --output "srf_exp_runs/results/llava_baseline_gpu0/"

echo ""
echo "✓ BASELINE COMPLETE"
echo "Results: srf_exp_runs/results/llava_baseline_gpu0/summary.json"
