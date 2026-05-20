#!/bin/bash
# GQA Adversarial Sweep - Top configurations from COCO analysis

set -euo pipefail

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
OUTPUT_BASE="/home/anna2/shruthi/lmms-eval/results/srf_focused_sweep"

echo "============================================================"
echo "GQA ADVERSARIAL SWEEP - 10 Configurations"
echo "============================================================"

configs=(
    # Best from COCO - ultra-low alpha
    "alpha=0.1 layers=10-18 heads=0.5 eps=0.0"
    "alpha=0.2 layers=10-18 heads=0.5 eps=0.1"
    "alpha=0.5 layers=10-18 heads=0.5 eps=0.1"
    
    # Best layer ranges from COCO
    "alpha=1.0 layers=10-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=12-16 heads=0.5 eps=0.1"
    
    # Conservative head selection
    "alpha=2.0 layers=10-18 heads=0.5 eps=0.1"
    "alpha=2.0 layers=12-18 heads=0.5 eps=0.1"
    
    # Narrow ranges
    "alpha=0.5 layers=14-18 heads=0.5 eps=0.1"
    "alpha=1.0 layers=14-15 heads=0.5 eps=0.1"
    
    # Original VAF-like
    "alpha=0.15 layers=10-15 heads=0.5 eps=0.2"
)

config_id=300
gpu_id=0

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

    gpu_id=$(( (gpu_id + 1) % 7 ))
    config_id=$((config_id + 1))
    sleep 2
done

echo ""
echo "Launched 10 GQA experiments (configs 300-309)"
