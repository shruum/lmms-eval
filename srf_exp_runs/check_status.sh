#!/bin/bash
# Quick status check for all nightly runs

echo "=================================================="
echo "EXPERIMENT STATUS CHECK"
date
echo "=================================================="
echo ""

# GPU Status
echo "=== GPU USAGE ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# MMVP Sweep Status
echo "=== MMVP SWEEP (Qwen-VL-Chat, GPU 2) ==="
if [ -f "srf_exp_runs/mmvp_sweep.log" ]; then
    echo "Progress:"
    tail -5 srf_exp_runs/mmvp_sweep.log
    echo ""
    echo "Completed runs:"
    find srf_exp_runs/results/qwenvlchat_mmvp -name "summary.json" | wc -l
    echo "Total runs: 13"
else
    echo "No log file found"
fi
echo ""

# LLaVA POPE Status
echo "=== POPE EVAL (LLaVA-1.5-7B, GPU 0) ==="
if [ -f "srf_exp_runs/llava_pope_full.log" ]; then
    echo "Progress:"
    tail -5 srf_exp_runs/llava_pope_full.log
    echo ""
    echo "Completed runs:"
    find srf_exp_runs/results/llava15_7b_pope -name "summary.json" | wc -l
    echo "Total runs: 6"
else
    echo "No log file found"
fi
echo ""

# LLaVA MMVP Status
echo "=== MMVP EVAL (LLaVA-1.5-7B, Auto GPU) ==="
if [ -f "srf_exp_runs/llava_mmvp_auto.log" ]; then
    echo "Status:"
    tail -3 srf_exp_runs/llava_mmvp_auto.log
    echo ""
    echo "Completed runs:"
    find srf_exp_runs/results/llava15_7b_mmvp -name "summary.json" | wc -l
else
    echo "No log file (not started yet)"
fi
echo ""

echo "=================================================="
echo "QUICK RESULTS SUMMARY"
echo "=================================================="
echo ""

# Qwen-VL-Chat MMVP Results
echo "Qwen-VL-Chat MMVP:"
for f in srf_exp_runs/results/qwenvlchat_mmvp/*/summary.json; do
    if [ -f "$f" ]; then
        dir=$(basename $(dirname "$f"))
        pair_acc=$(grep -o '"baseline_pair":[0-9.]*' "$f" | cut -d: -f2)
        srf_acc=$(grep -o '"pair_acc":[0-9.]*' "$f" | head -1 | cut -d: -f2)
        echo "  $dir: baseline=$pair_acc, srf=$srf_acc"
    fi
done
echo ""

# LLaVA POPE Results  
echo "LLaVA-1.5-7B POPE:"
for f in srf_exp_runs/results/llava15_7b_pope/*/summary.json; do
    if [ -f "$f" ]; then
        dir=$(basename $(dirname "$f"))
        acc=$(grep -o '"accuracy":[0-9.]*' "$f" | head -1 | cut -d: -f2)
        f1=$(grep -o '"f1":[0-9.]*' "$f" | head -1 | cut -d: -f2)
        echo "  $dir: acc=$acc, f1=$f1"
    fi
done
echo ""

echo "=================================================="
