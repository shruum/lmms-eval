#!/bin/bash
# Final monitoring script for LLaVA POPE sweep

while true; do
    clear
    echo "=================================================="
    echo "LLaVA POPE PARALLEL SWEEP STATUS"
    date
    echo "=================================================="
    
    # Count processes per GPU
    echo ""
    echo "PROCESSES:"
    ps aux | grep "srf/eval.py" | grep -v grep | wc -l
    echo "total eval.py processes running"
    
    # GPU usage
    echo ""
    echo "GPU MEMORY USAGE:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -3
    
    # Progress
    echo ""
    echo "COMPLETED RUNS:"
    BASE_DIR="/home/anna2/shruthi/lmms-eval/srf_exp_runs/results/llava15_7b_pope_full"
    baseline=$([ -f "$BASE_DIR/baseline/summary.json" ] && echo "✓" || echo "running")
    srf_count=$(find "$BASE_DIR" -name "srf_alpha*" -name "summary.json" 2>/dev/null | wc -l)
    srfe_count=$(find "$BASE_DIR" -name "srfe_beta*" -name "summary.json" 2>/dev/null | wc -l)
    
    echo "Baseline: $baseline"
    echo "SRF: $srf_count/9"
    echo "SRF-E: $srfe_count/4"
    
    # Check if all done
    if [ "$baseline" = "✓" ] && [ $srf_count -eq 9 ] && [ $srfe_count -eq 4 ]; then
        echo ""
        echo "🎉 ALL EXPERIMENTS COMPLETED!"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done
