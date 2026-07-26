#!/bin/bash
# MMVP parameter sweep for Qwen-VL-Chat
# GPU=2

set -e

BASE_DIR="srf_exp_runs/results/qwenvlchat_mmvp"
mkdir -p "$BASE_DIR"

echo "=================================================="
echo "MMVP PARAMETER SWEEP - Qwen-VL-Chat"
echo "GPU: 2"
echo "=================================================="

# SRF sweep: alpha x eps combinations
echo ""
echo "=== SRF SWEEP: alpha x eps ==="
ALPHAS=(2.0 4.0 8.0)
EPS_VALS=(0.1 0.2 0.3)

for ALPHA in "${ALPHAS[@]}"; do
    for EPS in "${EPS_VALS[@]}"; do
        OUT_DIR="$BASE_DIR/srf_alpha${ALPHA}_eps${EPS}"
        mkdir -p "$OUT_DIR"

        echo ""
        echo "[$(date +%H:%M:%S)] SRF: alpha=$ALPHA eps=$EPS"
        CUDA_VISIBLE_DEVICES=2 conda run -n mllm python srf/eval.py \
            --method srf \
            --model "Qwen/Qwen-VL-Chat" \
            --datasets mmvp \
            --output "$OUT_DIR/" \
            --alpha "$ALPHA" \
            --eps "$EPS" \
            --phase both \
            --layer_end 17 \
            > "$OUT_DIR/run.log" 2>&1

        # Check if run succeeded
        if [ -f "$OUT_DIR/summary.json" ]; then
            echo "  ✓ Completed"
            grep "pair_acc" "$OUT_DIR/summary.json" || echo "  Warning: No pair_acc in summary"
        else
            echo "  ✗ Failed - check $OUT_DIR/run.log"
        fi
    done
done

# SRF-E sweep: beta values
echo ""
echo "=== SRF-E SWEEP: beta values ==="
BETAS=(0.5 1.0 1.5 2.0)

for BETA in "${BETAS[@]}"; do
    OUT_DIR="$BASE_DIR/srfe_beta${BETA}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "[$(date +%H:%M:%S)] SRF-E: beta=$BETA"
    CUDA_VISIBLE_DEVICES=2 conda run -n mllm python srf/eval.py \
        --method srfe \
        --model "Qwen/Qwen-VL-Chat" \
        --datasets mmvp \
        --output "$OUT_DIR/" \
        --beta "$BETA" \
        --alpha 4.0 \
        --eps 0.2 \
        --phase both \
        --layer_end 17 \
        > "$OUT_DIR/run.log" 2>&1

    # Check if run succeeded
    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "  ✓ Completed"
        grep "pair_acc" "$OUT_DIR/summary.json" || echo "  Warning: No pair_acc in summary"
    else
        echo "  ✗ Failed - check $OUT_DIR/run.log"
    fi
done

echo ""
echo "=================================================="
echo "MMVP SWEEP COMPLETE"
echo "Results in: $BASE_DIR"
echo "=================================================="
