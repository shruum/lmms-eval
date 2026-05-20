#!/bin/bash
# =============================================================================
# Detailed Status Check for SRF Hyperparameter Sweep
# =============================================================================

echo "============================================================"
echo "SRF HYPERPARAMETER SWEEP - DETAILED STATUS"
echo "============================================================"
echo ""

# GPU Status
echo "🖥️  GPU STATUS:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "  GPU %s: %s%% GPU, %s / %s memory\n", $1, $2, $3, $4}'
echo ""

# Running processes
echo "⚙️  RUNNING PROCESSES:"
ps aux | grep "eval.py.*srf" | grep "Rl" | wc -l
echo "active evaluation processes running"
echo ""

# Count experiment status
echo "📊 EXPERIMENT PROGRESS (COCO Adversarial):"
echo "----------------------------------------"

total=45
completed=0
running=0
failed=0
low_acc=0
good_acc=0
excellent_acc=0

baseline_target=79.30
min_acceptable=76.0
good_threshold=78.0

for i in $(seq 0 44); do
    result_file="/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep/coco_adversarial_config${i}/pope_coco_adversarial_srf.json"
    log_file="/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep/coco_adversarial_config${i}/run.log"

    if [ -f "$result_file" ]; then
        completed=$((completed + 1))

        # Get accuracy
        acc=$(python3 -c "import json; print(f'{json.load(open(\"$result_file\"))[\"method\"].get(\"0.0\", {}).get(\"accuracy\", 0) * 100:.2f}')" 2>/dev/null)

        if [ ! -z "$acc" ]; then
            if (( $(echo "$acc >= $baseline_target" | bc -l) )); then
                excellent_acc=$((excellent_acc + 1))
                echo "  ✅ Config ${i}: ${acc}% (EXCEEDS baseline!)"
            elif (( $(echo "$acc >= $good_threshold" | bc -l) )); then
                good_acc=$((good_acc + 1))
                echo "  ✓ Config ${i}: ${acc}% (Good, above threshold)"
            elif (( $(echo "$acc >= $min_acceptable" | bc -l) )); then
                low_acc=$((low_acc + 1))
                echo "  ⚠️  Config ${i}: ${acc}% (Acceptable, below target)"
            else
                low_acc=$((low_acc + 1))
                echo "  ❌ Config ${i}: ${acc}% (Too low)"
            fi
        fi

    elif [ -f "$log_file" ]; then
        if grep -q "Evaluating" "$log_file" 2>/dev/null; then
            running=$((running + 1))
            # Get progress if available
            progress=$(grep "Evaluating" "$log_file" | tail -1 2>/dev/null | grep -oP '\d+%' || echo "loading")
            echo "  🔄 Config ${i}: RUNNING (${progress##*:})"
        elif grep -q "ERROR\|Exception\|Traceback" "$log_file" 2>/dev/null; then
            failed=$((failed + 1))
            echo "  ❌ Config ${i}: FAILED (has errors)"
        fi
    else
        echo "  ⏳ Config ${i}: NOT STARTED"
    fi
done

echo ""
echo "Summary:"
echo "--------"
echo "  Completed:  $completed/$total"
echo "  Running:   $running/$total"
echo "  Failed:    $failed/$total"
echo ""
echo "Performance:"
echo "  ≥ ${baseline_target}% (baseline target): $excellent_acc/45"
echo "  ≥ ${good_threshold}% (good threshold): $((good_acc + excellent_acc))/45"
echo "  ≥ ${min_acceptable}% (min acceptable): $((low_acc + good_acc + excellent_acc))/45"
echo "  < ${min_acceptable}% (too low): $low_acc/45"

echo ""
echo "============================================================"

# Show best configs so far
if [ $completed -gt 0 ]; then
    echo ""
    echo "🏆 TOP 5 CONFIGURATIONS SO FAR:"
    echo "----------------------------------------"

    python3 << 'PYEOF' 2>/dev/null
import json
import os

results = []
for i in range(45):
    result_file = f"/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep/coco_adversarial_config{i}/pope_coco_adversarial_srf.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
            method = data.get("method", {}).get("0.0", {})
            acc = method.get("accuracy", 0) * 100
            results.append((i, acc))

results.sort(key=lambda x: x[1], reverse=True)

for i, (config_id, acc) in enumerate(results[:5]):
    print(f"  #{i+1}: Config {config_id} - {acc:.2f}%")
PYEOF
fi

echo ""
echo "Monitor Commands:"
echo "  Full monitor log:  tail -f /home/anna2/shruthi/lmms-eval/results/adaptive_monitor.log"
echo "  Detailed check:  bash check_sweep_detailed.sh"
echo "  GPU usage:       watch -n 5 nvidia-smi"
echo ""
echo "============================================================"
