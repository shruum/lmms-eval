#!/bin/bash
# Script 2: LLaVA-1.5-7B-HF POPE SRF Sweep - GPU 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export CUDA_VISIBLE_DEVICES=1

BASE_DIR="srf_exp_runs/results/llava_srf_gpu1"
mkdir -p "$BASE_DIR"

echo "=================================================="
echo "LLaVA-1.5-7B-HF POPE SRF SWEEP"
echo "GPU: 1"
echo "=================================================="

# SRF parameter combinations
ALPHAS=(2.0 4.0 8.0)
EPS_VALS=(0.1 0.2 0.3)

for ALPHA in "${ALPHAS[@]}"; do
    for EPS in "${EPS_VALS[@]}"; do
        echo ""
        echo "SRF: alpha=$ALPHA eps=$EPS"
        
        python srf/eval.py \
            --method srf \
            --model "llava-hf/llava-1.5-7b-hf" \
            --datasets pope \
            --output "$BASE_DIR/srf_alpha${ALPHA}_eps${EPS}/" \
            --alpha "$ALPHA" \
            --eps "$EPS" \
            --phase both
    done
done

echo ""
echo "✓ SRF SWEEP COMPLETE"
echo "Results in: $BASE_DIR"
