#!/bin/bash
# Layer-Specific Modulation - Parallel Experiments on GPUs 0,1,2,3
#
# Tests different layer ranges and text suppression values for VLM Bias
# Each GPU runs a different experiment configuration

set -e

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_BASE="results/vlmbias_layer_specific/"
N_SAMPLES=10  # Per category (70 total)

# Free GPUs
GPUS=(0 1 2 6)

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "🔍 LAYER-SPECIFIC MODULATION SWEEP"
echo "=========================================="
echo "GPUs: ${GPUS[@]}"
echo "Samples per category: $N_SAMPLES"
echo "Baseline: 19.00%"
echo "Target: >21.00% (Δ=+2.0%)"
echo "=========================================="

# Experiment configurations
# Each tests different layer ranges + text suppression
declare -A CONFIGS

CONFIGS[0]="--alpha 8.0 --layer_start 20 --layer_end 27 --text_beta 0.8 --phase generation"
CONFIGS[1]="--alpha 8.0 --layer_start 16 --layer_end 27 --text_beta 0.6 --phase generation"
CONFIGS[2]="--alpha 6.0 --layer_start 8 --layer_end 15 --text_beta 0.0 --phase generation"
CONFIGS[3]="--alpha 4.0 --layer_start 8 --layer_end 27 --text_beta 0.5 --phase generation"

declare -A NAMES
NAMES[0]="late_20-27_textBeta0.8"
NAMES[1]="late_16-27_textBeta0.6"
NAMES[2]="mid_8-15_textBeta0.0"
NAMES[3]="wide_8-27_textBeta0.5"

declare -a PIDS
declare -a GPU_ASSIGNMENTS

# Launch experiments in parallel
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    CONFIG=${CONFIGS[$i]}
    NAME=${NAMES[$i]}
    OUTPUT_DIR="${OUTPUT_BASE}${NAME}/"

    echo ""
    echo "🚀 Launching Experiment $((i+1))/4 on GPU $GPU"
    echo "   Config: $NAME"
    echo "   Output: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    # Run in background
    CUDA_VISIBLE_DEVICES=$GPU conda run -n mllm --no-capture-output python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets vlmbias \
        --n_vlmbias_per_cat $N_SAMPLES \
        --output "$OUTPUT_DIR" \
        $CONFIG \
        > "$OUTPUT_DIR/log.txt" 2>&1 &

    PID=$!
    PIDS[$i]=$PID
    GPU_ASSIGNMENTS[$i]="GPU $GPU (PID: $PID) - $NAME"

    echo "   ✅ Started: $NAME (PID: $PID)"
done

echo ""
echo "=========================================="
echo "⏳ All experiments launched"
echo "=========================================="
for i in "${!GPU_ASSIGNMENTS[@]}"; do
    echo "  Exp $((i+1)): ${GPU_ASSIGNMENTS[$i]}"
done

echo ""
echo "Monitoring progress..."
echo "Press Ctrl+C to stop all experiments"

# Monitor function
cleanup() {
    echo ""
    echo "=========================================="
    echo "🛑 Stopping all experiments..."
    echo "=========================================="

    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        NAME=${NAMES[$i]}

        if ps -p $PID > /dev/null 2>&1; then
            echo "  Stopping $NAME (PID: $PID)..."
            kill $PID 2>/dev/null || true
        fi
    done

    # Wait for all processes
    wait 2>/dev/null || true

    echo ""
    echo "=========================================="
    echo "📊 PARTIAL RESULTS"
    echo "=========================================="

    # Parse results from completed experiments
    for i in "${!NAMES[@]}"; do
        NAME=${NAMES[$i]}
        OUTPUT_DIR="${OUTPUT_BASE}${NAME}/"
        LOG_FILE="$OUTPUT_DIR/log.txt"

        if [ -f "$LOG_FILE" ]; then
            # Extract accuracy
            ACC=$(grep -i "accuracy" "$LOG_FILE" | grep -oP "[\d.]+%" | head -1 || echo "N/A")

            if [ "$ACC" != "N/A" ]; then
                echo "  $NAME: $ACC"
            else
                echo "  $NAME: Running/Failed"
            fi
        fi
    done

    echo ""
    echo "Check logs in: $OUTPUT_BASE"
    echo "=========================================="

    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Wait for all experiments to complete
echo ""
WAIT_TIME=0
CHECK_INTERVAL=60  # Check every 60 seconds

while true; do
    # Check if all processes are still running
    ALL_RUNNING=true
    COMPLETED=0

    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        NAME=${NAMES[$i]}

        if ! ps -p $PID > /dev/null 2>&1; then
            ALL_RUNNING=false
            COMPLETED=$((COMPLETED + 1))
        fi
    done

    # Status update
    echo "[$(date +%H:%M:%S)] Running: $((4 - COMPLETED))/4 | Elapsed: ${WAIT_TIME}s"

    if [ $COMPLETED -eq 4 ]; then
        echo ""
        echo "=========================================="
        echo "✅ All experiments completed!"
        echo "=========================================="
        break
    fi

    sleep $CHECK_INTERVAL
    WAIT_TIME=$((WAIT_TIME + CHECK_INTERVAL))
done

# Wait for all background processes
wait

# Final results
echo ""
echo "=========================================="
echo "📊 FINAL RESULTS"
echo "=========================================="

for i in "${!NAMES[@]}"; do
    NAME=${NAMES[$i]}
    OUTPUT_DIR="${OUTPUT_BASE}${NAME}/"

    echo ""
    echo "Experiment: $NAME"

    # Extract accuracy
    if [ -f "$OUTPUT_DIR/log.txt" ]; then
        ACC=$(grep -i "accuracy" "$OUTPUT_DIR/log.txt" | grep -oP "[\d.]+%" | head -1 || echo "N/A")
        echo "  Accuracy: $ACC"

        # Extract timing if available
        TIME=$(grep "eval_time" "$OUTPUT_DIR/log.txt" | grep -oP "[\d.]+" | head -1 || echo "N/A")
        if [ "$TIME" != "N/A" ]; then
            echo "  Time: ${TIME}s"
        fi
    else
        echo "  Status: Failed (no log file)"
    fi
done

echo ""
echo "=========================================="
echo "📁 Results saved to: $OUTPUT_BASE"
echo "=========================================="

# Generate summary
python3 << 'EOF'
import json
import re
from pathlib import Path

OUTPUT_BASE = "results/vlmbias_layer_specific/"
BASELINE = 19.00

results = []

for exp_dir in Path(OUTPUT_BASE).iterdir():
    if not exp_dir.is_dir():
        continue

    log_file = exp_dir / "log.txt"
    if not log_file.exists():
        continue

    try:
        content = log_file.read_text()

        # Extract accuracy
        match = re.search(r'accuracy[=:]?\s*([\d.]+)%?', content, re.IGNORECASE)
        if match:
            acc = float(match.group(1))

            results.append({
                "name": exp_dir.name,
                "accuracy": acc,
                "delta": acc - BASELINE
            })
    except:
        pass

if results:
    print("\n" + "="*70)
    print("🏆 RANKING")
    print("="*70)

    results.sort(key=lambda x: x["accuracy"], reverse=True)

    for i, r in enumerate(results):
        delta_str = f"+{r['delta']:.2f}%" if r['delta'] >= 0 else f"{r['delta']:.2f}%"

        if r['delta'] >= 2.0:
            status = "✅"
        elif r['delta'] >= 1.0:
            status = "🟡"
        elif r['delta'] > 0:
            status = "🟠"
        else:
            status = "🔴"

        print(f"{i+1}. {status} {r['name']:30s}: {r['accuracy']:.2f}% (Δ={delta_str})")

    # Best config
    best = results[0]
    print("\n" + "="*70)
    print("🎯 BEST CONFIGURATION")
    print("="*70)
    print(f"Name: {best['name']}")
    print(f"Accuracy: {best['accuracy']:.2f}%")
    print(f"Delta: {best['delta']:+.2f}%")

    if best['delta'] >= 2.0:
        print("\n✅ TARGET ACHIEVED!")
        print("Next step: Run full evaluation with this config")
    elif best['delta'] >= 1.0:
        print("\n⚠️  Promising (1-2% improvement)")
        print("Next step: Fine-tune around this config")
    else:
        print("\n❌ Target not achieved")
        print("Recommendation: Try alternative approaches")

    # Save summary
    summary = {
        "baseline": BASELINE,
        "target": 21.00,
        "results": results
    }

    with open(f"{OUTPUT_BASE}summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 Summary saved to: {OUTPUT_BASE}summary.json")

EOF

echo ""
echo "=========================================="
