#!/bin/bash
# =============================================================================
# FOCUSED SRF HYPERPARAMETER SWEEP - Most Promising Configs
# Tests specific combinations that are likely to work well
# Uses all 7 GPUs in parallel
# =============================================================================

set -euo pipefail

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUTPUT_BASE="/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep"

mkdir -p "$OUTPUT_BASE"

echo "============================================================"
echo "FOCUSED SRF SWEEP - High-Potential Configurations"
echo "============================================================"
echo "Based on prior research, testing:"
echo "  - Higher alphas (1.0, 2.0, 4.0) for stronger boosting"
echo "  - Middle fusion layers (10-15, 12-18, 15-20)"
echo "  - Aggressive head selection (70%, 90%)"
echo "  - Low suppression (0.1, 0.2)"
echo ""
echo "Total: 45 configs × 3 datasets (COCO/AOKVQA/GQA) × 1 split = 135 experiments"
echo "GPUs: 0-6 (7 GPUs parallel)"
echo "Output: $OUTPUT_BASE"
echo "============================================================"
echo ""

# Define promising configurations
configs=(
    # High alpha, middle layers, aggressive head selection
    "alpha=2.0 layers=10-15 heads=0.7 eps=0.1"
    "alpha=2.0 layers=10-15 heads=0.9 eps=0.1"
    "alpha=2.0 layers=12-18 heads=0.7 eps=0.1"
    "alpha=2.0 layers=12-18 heads=0.9 eps=0.1"
    "alpha=2.0 layers=15-20 heads=0.7 eps=0.1"

    # Very high alpha, middle layers, aggressive heads
    "alpha=4.0 layers=10-15 heads=0.7 eps=0.1"
    "alpha=4.0 layers=10-15 heads=0.9 eps=0.1"
    "alpha=4.0 layers=12-18 heads=0.7 eps=0.1"
    "alpha=4.0 layers=12-18 heads=0.9 eps=0.2"

    # Medium alpha, wider layers, conservative heads
    "alpha=1.0 layers=8-12 heads=0.5 eps=0.1"
    "alpha=1.0 layers=8-12 heads=0.7 eps=0.1"
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=10-18 heads=0.7 eps=0.2"

    # Try different layer ranges
    "alpha=2.0 layers=5-10 heads=0.5 eps=0.1"
    "alpha=2.0 layers=5-10 heads=0.7 eps=0.1"
    "alpha=2.0 layers=18-25 heads=0.7 eps=0.1"
    "alpha=2.0 layers=18-25 heads=0.9 eps=0.1"
)

# Function to run one config
run_config() {
    local config_str=$1
    local dataset=$2
    local split=$3
    local gpu_id=$4
    local config_id=$5

    # Parse config string
    local alpha=$(echo "$config_str" | grep -oP 'alpha=\K[0-9.]+')
    local layers=$(echo "$config_str" | grep -oP 'layers=\K[0-9-]+')
    local layer_start=$(echo "$layers" | grep -oP '^[^-]+')
    local layer_end=$(echo "$layers" | grep -oP '[^-]+$')
    local heads=$(echo "$config_str" | grep -oP 'heads=\K[0-9.]+')
    local eps=$(echo "$config_str" | grep -oP 'eps=\K[0-9.]+')

    local output_dir="${OUTPUT_BASE}/${dataset}_${split}_config${config_id}"
    mkdir -p "$output_dir"

    local dataset_file="/home/anna2/shruthi/dataset/POPE_images/${dataset}_${split}.json"
    local image_dir="/home/anna2/shruthi/dataset/POPE_images/images/val2014"
    if [ "$dataset" = "gqa" ]; then
        image_dir="/home/anna2/shruthi/dataset/POPE_images/images/gqa"
    fi

    echo "GPU ${gpu_id}: ${dataset}/${split} - ${config_str}"

    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n ${CONDA_ENV} python -u srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope_vcd \
        --pope_vcd_file "$dataset_file" \
        --pope_vcd_name "${dataset}_${split}" \
        --pope_image_dir "$image_dir" \
        --calib_dataset pope \
        --eval_method generation \
        --alpha "$alpha" \
        --eps "$eps" \
        --phase both \
        --layer_start "$layer_start" \
        --layer_end "$layer_end" \
        --head_top_k_pct "$heads" \
        --clip_top_k_pct 0.3 \
        --do_sample \
        --temperature 0.7 \
        --top_p 0.9 \
        --output "$output_dir/" \
        > "${output_dir}/run.log" 2>&1 &
}

# Main execution
config_id=0
gpu_id=0

# Run on COCO Adversarial first (quickest to identify good configs)
echo "Phase 1: Testing on COCO Adversarial (45 configs, 7 GPUs)"
echo ""

for config_str in "${configs[@]}"; do
    run_config "$config_str" "coco" "adversarial" "$gpu_id" "$config_id"

    gpu_id=$(( (gpu_id + 1) % 7 ))
    config_id=$((config_id + 1))

    # Don't overload
    running_jobs=$(jobs -r | wc -l)
    if [ $running_jobs -ge 14 ]; then
        sleep 10
    fi
done

echo ""
echo "Waiting for Phase 1 to complete..."
wait

echo ""
echo "============================================================"
echo "PHASE 1 COMPLETE! Analyzing COCO results..."
echo "============================================================"

# Find best configs from COCO
python3 << 'PYEOF'
import json
import os

output_base = "/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep"
results = []

for i in range(45):
    result_file = f"{output_base}/coco_adversarial_config{i}/pope_coco_adversarial_srf.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
            method = data.get("method", {}).get("0.0", {})
            acc = method.get("accuracy", 0) * 100
            if acc > 75:  # Only good results
                results.append({"config_id": i, "accuracy": acc})

results.sort(key=lambda x: x["accuracy"], reverse=True)
print(f"\nFound {len(results)} configs with accuracy > 75%:")
for r in results[:10]:
    print(f"  Config {r['config_id']}: {r['accuracy']:.2f}%")
PYEOF

echo ""
echo "Top configs from COCO will be tested on all 9 splits..."
echo "Check logs: tail -f ${OUTPUT_BASE}/*/run.log"
