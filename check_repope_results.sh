#!/bin/bash
OUTPUT_BASE="results/repope_sequential_sweep"

echo "📊 REPOPE SEQUENTIAL SWEEP STATUS"
echo "================================"
echo ""

TOTAL=$(find "$OUTPUT_BASE" -name "run.log" | wc -l)
COMPLETED=$(grep -l "accuracy" "$OUTPUT_BASE"/*/run.log 2>/dev/null | wc -l)
RUNNING=$(ps aux | grep "eval.py" | grep "pope_vcd" | grep -v grep | wc -l)

echo "📈 Progress: $COMPLETED/$TOTAL completed, $RUNNING running"
echo ""

echo "🎮 GPU Usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | nl
echo ""

if [ $COMPLETED -gt 0 ]; then
    echo "🏆 RESULTS:"
    echo "----------------------------------------"

    for log_file in "$OUTPUT_BASE"/*/run.log; do
        if grep -q "accuracy" "$log_file" 2>/dev/null; then
            category=$(basename $(dirname "$log_file") | sed 's/_alpha.*//')
            baseline=$(grep -oP '"baseline":\{[^}]*"accuracy": \K[0-9.]*' "$log_file" | head -1)
            method=$(grep -oP '"method":\{[^}]*"0\.0":\{[^}]*"accuracy": \K[0-9.]*' "$log_file" | head -1)

            if [ -n "$method" ]; then
                baseline_pct=$(echo "$baseline * 100" | bc 2>/dev/null || echo "0")
                method_pct=$(echo "$method * 100" | bc 2>/dev/null || echo "0")
                delta=$(echo "$method_pct - $baseline_pct" | bc 2>/dev/null || echo "0")
                echo "$method_pct|$category|$delta"
            fi
        fi
    done | sort -t'|' -k1 -rn | head -10 | while IFS='|' read acc cat delta; do
        printf "  %6.2f%% | %15s | %+5.2f%%\n" "$acc" "$cat" "$delta"
    done
    echo ""
fi

echo "💡 Monitor: tail -f $OUTPUT_BASE/<dir>/run.log"