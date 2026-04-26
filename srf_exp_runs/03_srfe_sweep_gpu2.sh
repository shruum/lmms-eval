#!/bin/bash
# Script 3: LLaVA-1.5-7B-HF POPE SRF-E Sweep - GPU 2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export CUDA_VISIBLE_DEVICES=2

BASE_DIR="srf_exp_runs/results/llava_srfe_gpu2"
mkdir -p "$BASE_DIR"

echo "=================================================="
echo "LLaVA-1.5-7B-HF POPE SRF-E SWEEP"
echo "GPU: 2"
echo "=================================================="

# SRF-E beta values
BETAS=(0.5 1.0 1.5 2.0)

for BETA in "${BETAS[@]}"; do
    echo ""
    echo "SRF-E: beta=$BETA"
    
    python srf/eval.py \
        --method srfe \
        --model "llava-hf/llava-1.5-7b-hf" \
        --datasets pope \
        --output "$BASE_DIR/srfe_beta${BETA}/" \
        --beta "$BETA" \
        --alpha 4.0 \
        --eps 0.2 \
        --phase both
done

echo ""
echo "✓ SRF-E SWEEP COMPLETE"
echo "Results in: $BASE_DIR"
