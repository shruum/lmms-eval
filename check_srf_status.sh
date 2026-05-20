#!/bin/bash
# =============================================================================
# Quick status check for SRF experiments
# =============================================================================

echo "============================================================"
echo "SRF EXPERIMENTS - QUICK STATUS"
echo "============================================================"
echo ""

# Check if monitoring is running
if ps -p 918094 > /dev/null 2>&1; then
    echo "✅ Monitoring script: RUNNING (PID: 918094)"
else
    echo "❌ Monitoring script: NOT RUNNING"
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "  GPU %s: %s utilization, %s memory\n", $1, $2, $3}'

echo ""
echo "Running SRF Processes:"
ps aux | grep "eval.py.*srf" | grep "Rl" | grep -v grep | wc -l
echo "active evaluation processes"

echo ""
echo "Experiment Progress:"
echo "--------------------"

for dataset in coco aokvqa gqa; do
    echo "${dataset^^}:"
    for split in adversarial popular random; do
        result_file="/home/anna2/shruthi/lmms-eval/results/llava_pope_sampling_srf/${dataset}/${split}/pope_${dataset}_${split}_srf.json"
        log_file="/home/anna2/shruthi/lmms-eval/results/llava_pope_sampling_srf/${dataset}/${split}/srf_sampling.log"

        if [ -f "$result_file" ]; then
            # Completed
            acc=$(python3 -c "import json; print(f'{json.load(open('$result_file'))[\"accuracy\"]:.2f}%')" 2>/dev/null)
            echo "  ✅ ${split^^}: COMPLETED (${acc})"
        elif [ -f "$log_file" ]; then
            # Check if running or has errors
            if grep -q "ERROR\|Exception\|Traceback" "$log_file" 2>/dev/null; then
                echo "  ⚠️  ${split^^}: HAS ERRORS"
            elif grep -q "Evaluating" "$log_file" 2>/dev/null; then
                progress=$(grep "Evaluating" "$log_file" | tail -1 | grep -oP '\d+%' || echo "in progress")
                echo "  🔄 ${split^^}: RUNNING (${progress##*:})"
            else
                echo "  ❓ ${split^^}: LOADING"
            fi
        else
            echo "  ❌ ${split^^}: NOT STARTED"
        fi
    done
    echo ""
done

echo "============================================================"
echo "Monitoring Commands:"
echo "  Full monitoring log: tail -f /home/anna2/shruthi/lmms-eval/results/srf_monitoring.log"
echo "  Any experiment log:  tail -f /home/anna2/shruthi/lmms-eval/results/llava_pope_sampling_srf/coco/adversarial/srf_sampling.log"
echo "  GPU usage:           watch -n 10 nvidia-smi"
echo ""
echo "Stop monitoring: kill 918094"
echo "============================================================"
