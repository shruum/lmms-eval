#!/bin/bash
# Quick launcher for parallel experiments on 8 GPUs
# Usage: bash srf/run_all_parallel.sh

set -e

echo "=========================================="
echo "Parallel Experiments Launcher (8 GPUs)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllm

# Change to repo directory
cd /home/anna2/shruthi/lmms-eval

# Create logs directory
mkdir -p logs

echo ""
echo "This will launch TWO experiments in parallel:"
echo "  - GPUs 0-3: Hard POPE sweep (193 configs)"
echo "  - GPUs 4-7: Layer-specific modulation (244 configs)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Check if hard samples exist
if [ ! -f "srf/hard_samples_pope.json" ]; then
    echo ""
    echo "Hard samples not found. Finding them now..."
    echo ""

    python srf/find_hard_pope_samples.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --pope_splits adversarial \
        --n_samples 100 \
        --output srf/hard_samples_pope.json

    echo ""
    echo "Hard samples found!"
    echo ""
fi

# Launch both experiments
echo "Launching experiments..."
echo ""

# Experiment 1: Hard POPE sweep (GPUs 0-3)
echo "Starting Hard POPE sweep on GPUs 0-3..."
nohup python srf/sweep_hard_pope_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/sweep_hard_pope_parallel/ \
    --num_gpus 4 \
    --layer_ranges 5-10 8-15 10-15 15-25 \
    --head_pcts 0.3 0.5 0.8 \
    --alphas 0.15 0.5 1.0 2.0 \
    --with_text_suppression \
    --with_sys_suppression \
    --include_post_softmax \
    > logs/hard_pope.log 2>&1 &
HARD_POPE_PID=$!

# Experiment 2: Layer-specific modulation (GPUs 4-7)
echo "Starting Layer-specific modulation on GPUs 4-7..."
nohup python srf/sweep_layer_specific_parallel.py \
    --hard_samples srf/hard_samples_pope.json \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output results/sweep_layer_specific/ \
    --num_gpus 4 \
    --early_ends 5 7 10 \
    --mid_ends 15 17 20 \
    --alpha_early 0.3 0.5 1.0 \
    --alpha_mid 1.0 2.0 4.0 \
    --beta_late 0.05 0.1 0.2 \
    > logs/layer_specific.log 2>&1 &
LAYER_SPEC_PID=$!

echo ""
echo "=========================================="
echo "Both experiments launched!"
echo "=========================================="
echo ""
echo "Hard POPE sweep (GPUs 0-3): PID $HARD_POPE_PID"
echo "  Logs: tail -f logs/hard_pope.log"
echo ""
echo "Layer-specific sweep (GPUs 4-7): PID $LAYER_SPEC_PID"
echo "  Logs: tail -f logs/layer_specific.log"
echo ""
echo "Monitor GPU utilization:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Estimated completion: 2-3 hours"
echo ""
