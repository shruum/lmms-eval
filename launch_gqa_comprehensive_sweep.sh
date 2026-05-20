#!/bin/bash
# GQA Adversarial Comprehensive Sweep - Based on COCO Round 1 Analysis
# Target: ≥76.5% (baseline 75.5% + 1%)

set -euo pipefail

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUTPUT_BASE="/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep"

echo "============================================================"
echo "GQA ADVERSARIAL COMPREHENSIVE SWEEP"
echo "============================================================"
echo "Target: ≥76.5% (baseline 75.5% + 1%)"
echo "Strategy: Test ultra-low alpha + variations"
echo "============================================================"

configs=(
    # ========== ULTRA-LOW ALPHA (0.05-0.3) ==========
    # Based on COCO finding: lower alpha works better
    "alpha=0.05 layers=10-18 heads=0.5 eps=0.0"
    "alpha=0.05 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.1 layers=10-18 heads=0.5 eps=0.0"
    "alpha=0.1 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.15 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.2 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.25 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.3 layers=10-18 heads=0.5 eps=0.1"

    # ========== LOW ALPHA (0.5-1.0) WITH VARIATIONS ==========
    # Best from COCO Round 1
    "alpha=0.5 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.5 layers=12-18 heads=0.5 eps=0.1"
    "alpha=0.5 layers=14-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=12-16 heads=0.5 eps=0.1"
    "alpha=1.0 layers=14-18 heads=0.5 eps=0.1"

    # ========== HEAD SELECTION VARIATIONS ==========
    # Test different head percentages with best alpha
    "alpha=0.5 layers=10-18 heads=0.3 eps=0.1"
    "alpha=0.5 layers=10-18 heads=0.7 eps=0.1"
    "alpha=1.0 layers=10-18 heads=0.3 eps=0.1"
    "alpha=1.0 layers=10-18 heads=0.7 eps=0.1"

    # ========== EPS VARIATIONS ==========
    # Test suppression values
    "alpha=0.5 layers=10-18 heads=0.5 eps=0.0"
    "alpha=0.5 layers=10-18 heads=0.5 eps=0.15"
    "alpha=0.5 layers=10-18 heads=0.5 eps=0.2"
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.0"
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.15"
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.2"

    # ========== NARROW LAYER RANGES ==========
    # Test focused layer selections
    "alpha=0.5 layers=10-12 heads=0.5 eps=0.1"
    "alpha=0.5 layers=12-14 heads=0.5 eps=0.1"
    "alpha=0.5 layers=16-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=10-12 heads=0.5 eps=0.1"
    "alpha=1.0 layers=12-14 heads=0.5 eps=0.1"
    "alpha=1.0 layers=16-18 heads=0.5 eps=0.1"

    # ========== CONSERVATIVE MEDIUM ALPHA ==========
    # Test if higher alpha works differently on GQA
    "alpha=2.0 layers=10-18 heads=0.5 eps=0.1"
    "alpha=2.0 layers=12-18 heads=0.5 eps=0.1"
)

config_id=400
gpu_id=0

echo "Total configurations: ${#configs[@]}"
echo "Config IDs: ${config_id}-$((config_id + ${#configs[@]} - 1))"
echo ""

for config_str in "${configs[@]}"; do
    alpha=$(echo "$config_str" | grep -oP 'alpha=\K[0-9.]+')
    layers=$(echo "$config_str" | grep -oP 'layers=\K[0-9-]+')
    layer_start=$(echo "$layers" | grep -oP '^[^-]+')
    layer_end=$(echo "$layers" | grep -oP '[^-]+$')
    heads=$(echo "$config_str" | grep -oP 'heads=\K[0-9.]+')
    eps=$(echo "$config_str" | grep -oP 'eps=\K[0-9.]+')

    output_dir="${OUTPUT_BASE}/gqa_adversarial_config${config_id}"
    mkdir -p "$output_dir"

    echo "GPU ${gpu_id}: GQA Config ${config_id} - ${config_str}"

    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n ${CONDA_ENV} python -u srf/eval.py \
        --method srf \
        --model "${MODEL}" \
        --datasets pope_vcd \
        --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
        --pope_vcd_name "gqa_adversarial" \
        --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
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

    gpu_id=$(( (gpu_id + 1) % 8 ))
    config_id=$((config_id + 1))
    sleep 2
done

echo ""
echo "============================================================"
echo "Launched ${#configs[@]} GQA experiments"
echo "Results will be in: ${OUTPUT_BASE}/gqa_adversarial_config{400-$((${config_id} - 1))}/"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  bash check_sweep_detailed.sh"
echo "  tail -f ${OUTPUT_BASE}/gqa_adversarial_config*/run.log"
echo ""
echo "Quick analysis when complete:"
python3 << 'EOF'
import json, os
baseline = 75.5
target = baseline + 1.0

print(f"\n{'Config':<8} {'Accuracy':<10} {'vs Baseline':<12} {'Target Met':<12}")
print("-" * 50)

for i in range(400, 450):
    f = f'results/srf_focused_sweep/gqa_adversarial_config{i}/pope_gqa_adversarial.json'
    if os.path.exists(f):
        with open(f) as file:
            data = json.load(file)
            acc = data['method']['0.0']['accuracy'] * 100
            delta = acc - baseline
            met = "✅ YES" if acc >= target else "❌ NO"
            print(f"{i:<8} {acc:<10.2f}% {delta:+<12.2f}% {met:<12}")
EOF
