#!/bin/bash
# Test script to compare absence-aware vs simple boost configurations

echo "=== Test 1: Disable absence-aware (match your working mmvp-srf config) ==="
python srf/eval.py \
    --method srf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 10 \
    --alpha 4.0 \
    --eps 0.2 \
    --clip_suppress_thresh 0.0 \
    --output results/test_no_absence/

echo ""
echo "=== Test 2: Enable absence-aware (current default) ==="
python srf/eval.py \
    --method srf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 10 \
    --alpha 2.0 \
    --eps 0.0 \
    --clip_suppress_thresh 0.248 \
    --clip_suppress_alpha 5.0 \
    --output results/test_with_absence/

echo ""
echo "=== Test 3: Try intermediate threshold ==="
python srf/eval.py \
    --method srf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 10 \
    --alpha 3.0 \
    --eps 0.1 \
    --clip_suppress_thresh 0.20 \
    --clip_suppress_alpha 3.0 \
    --output results/test_intermediate/

echo ""
echo "Compare results in results/test_*/ directories"
