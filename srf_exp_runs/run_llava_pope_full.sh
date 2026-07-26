#!/bin/bash
# Full POPE evaluation for LLaVA-1.5-7B
# GPU=0

set -e

BASE_DIR="srf_exp_runs/results/llava15_7b_pope"
mkdir -p "$BASE_DIR"/{baseline,srf,srfe}

echo "=================================================="
echo "POPE FULL EVAL - LLaVA-1.5-7B"
echo "GPU: 0"
echo "=================================================="

# Baseline
echo ""
echo "=== BASELINE ==="
CUDA_VISIBLE_DEVICES=0 conda run -n mllm python srf/eval.py \
    --method baseline \
    --model "liuhaotian/llava-v1.5-7b" \
    --datasets pope \
    --output "$BASE_DIR/baseline/" \
    > "$BASE_DIR/baseline/run.log" 2>&1

if [ -f "$BASE_DIR/baseline/summary.json" ]; then
    echo "  ✓ Baseline completed"
else
    echo "  ✗ Baseline failed"
fi

# SRF
echo ""
echo "=== SRF (alpha=4.0, eps=0.2) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n mllm python srf/eval.py \
    --method srf \
    --model "liuhaotian/llava-v1.5-7b" \
    --datasets pope \
    --output "$BASE_DIR/srf/" \
    --alpha 4.0 \
    --eps 0.2 \
    --phase both \
    > "$BASE_DIR/srf/run.log" 2>&1

if [ -f "$BASE_DIR/srf/summary.json" ]; then
    echo "  ✓ SRF completed"
else
    echo "  ✗ SRF failed"
fi

# SRF-E sweep
echo ""
echo "=== SRF-E PARAMETER SWEEP ==="
BETAS=(0.5 1.0 1.5 2.0)

for BETA in "${BETAS[@]}"; do
    echo "  Beta=$BETA..."
    CUDA_VISIBLE_DEVICES=0 conda run -n mllm python srf/eval.py \
        --method srfe \
        --model "liuhaotian/llava-v1.5-7b" \
        --datasets pope \
        --output "$BASE_DIR/srfe_beta${BETA}/" \
        --beta "$BETA" \
        --alpha 4.0 \
        --eps 0.2 \
        --phase both \
        > "$BASE_DIR/srfe_beta${BETA}/run.log" 2>&1

    if [ -f "$BASE_DIR/srfe_beta${BETA}/summary.json" ]; then
        echo "    ✓ Completed"
    else
        echo "    ✗ Failed"
    fi
done

echo ""
echo "=================================================="
echo "POPE EVAL COMPLETE"
echo "Results in: $BASE_DIR"
echo "=================================================="
