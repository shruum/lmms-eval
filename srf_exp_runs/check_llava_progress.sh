#!/bin/bash
# Quick progress check for LLaVA POPE sweep

BASE_DIR="srf_exp_runs/results/llava15_7b_pope_full"

echo "=================================================="
echo "LLaVA POPE SWEEP PROGRESS"
date
echo "=================================================="

# Check current progress
baseline=$([ -f "$BASE_DIR/baseline/summary.json" ] && echo "✓" || echo "Running...")
srf_count=$(find "$BASE_DIR" -name "srf_alpha*" -name "summary.json" 2>/dev/null | wc -l)
srfe_count=$(find "$BASE_DIR" -name "srfe_beta*" -name "summary.json" 2>/dev/null | wc -l)

echo "Baseline: $baseline"
echo "SRF runs: $srf_count/9"
echo "SRF-E runs: $srfe_count/4"

# Check if still running
proc_count=$(ps aux | grep "llava.*pope" | grep -v grep | wc -l)
echo ""
echo "Active processes: $proc_count"

# Show current log if available
if [ -f "$BASE_DIR/baseline/run.log" ]; then
    echo ""
    echo "Latest output:"
    tail -10 "$BASE_DIR/baseline/run.log"
fi

echo ""
echo "=================================================="
