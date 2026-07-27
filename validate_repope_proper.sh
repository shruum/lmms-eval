#!/bin/bash
# Validation test for SRF improvements on RePOPE - using correct approach
# Based on scripts/experiment_scripts/run_srf_repoe_all.sh

set -e

MODEL="llava-hf/llava-1.5-7b-hf"
REPOPE_DIR="/home/anna2/shruthi/RePOPE/annotations"
IMAGE_DIR="/home/anna2/shruthi/dataset/POPE_images/images/val2014"
OUTPUT_DIR="/home/anna2/shruthi/lmms-eval/results/validation_repope_proper"
mkdir -p $OUTPUT_DIR

SPLIT="adversarial"
N_SAMPLES=100

echo "========================================"
echo "SRF Validation Test on RePOPE"
echo "Environment: mllm"
echo "Model: $MODEL"
echo "Split: $SPLIT, Samples: $N_SAMPLES"
echo "========================================"
echo ""

# Test configurations
CONFIGS=(
    "baseline:0.15:0.0:10:15:0.20"
    "autoresearch_best:4.0:0.2:8:15:0.20"
    "autoresearch_strong:5.0:0.2:8:15:0.20"
    "autoresearch_conservative:3.0:0.15:8:15:0.20"
)

BEST_ACC=0.0
BEST_CONFIG=""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllm

for config_str in "${CONFIGS[@]}"; do
    IFS=':' read -r DESC ALPHA EPS L_START L_END HEAD_PCT <<< "$config_str"
    
    echo "----------------------------------------"
    echo "Testing: $DESC"
    echo "Config: α=$ALPHA, ε=$EPS, layers=$L_START-$L_END, heads=$HEAD_PCT"
    echo "----------------------------------------"
    
    REPOPE_FILE="${REPOPE_DIR}/coco_repope_${SPLIT}.json"
    OUTPUT_SUBDIR="${OUTPUT_DIR}/validation_${DESC}"
    mkdir -p "$OUTPUT_SUBDIR"
    
    python srf/eval.py \
        --method srf \
        --model $MODEL \
        --datasets pope_vcd \
        --pope_vcd_file $REPOPE_FILE \
        --pope_vcd_name validation_${DESC} \
        --pope_image_dir $IMAGE_DIR \
        --do_sample \
        --temperature 0.7 \
        --top_p 0.9 \
        --alpha $ALPHA \
        --layer_start $L_START \
        --layer_end $L_END \
        --head_top_k_pct $HEAD_PCT \
        --eps $EPS \
        --calib_dataset pope \
        --output "$OUTPUT_SUBDIR" \
        2>&1 | tee "$OUTPUT_SUBDIR/run.log"
    
    # Try to extract accuracy from log
    ACC=$(grep -i "accuracy" "$OUTPUT_SUBDIR/run.log" | tail -1 | grep -oP '\d+\.\d+%' || echo "0.0%")
    echo "Result: $ACC"
    
    # Extract numeric accuracy
    ACC_NUM=$(echo $ACC | grep -oP '\d+\.\d+' | head -1 || echo "0.0")
    
    if (( $(echo "$ACC_NUM > 82.0" | bc -l 2>/dev/null || echo "0") )); then
        BEST_ACC=$ACC_NUM
        BEST_CONFIG="$DESC"
        echo "🎉 NEW BEST: $BEST_ACC"
    fi
done

echo ""
echo "========================================"
echo "Validation Complete!"
echo "Best config: $BEST_CONFIG"
echo "Best accuracy: $BEST_ACC"
echo "========================================"
