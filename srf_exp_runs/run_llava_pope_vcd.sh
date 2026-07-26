#!/bin/bash
# =============================================================================
# LLAVA POPE VCD Evaluation - 3 Datasets (COCO, A-OKVQA, GQA) × 3 Splits
# =============================================================================

set -euo pipefail

export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
MODEL="llava-hf/llava-1.5-7b-hf"
POPE_DATA_DIR="/home/anna2/shruthi/dataset/pope_vcd"
OUT_DIR="srf_exp_runs/results/llava_pope_vcd"
GPU_ID=1

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "LLAVA POPE VCD EVALUATION"
echo "Model: ${MODEL}"
echo "Datasets: COCO, A-OKVQA, GQA (3 splits each)"
echo "Method: generation-based (VCD/ClearSight compatible)"
echo "GPU: ${GPU_ID}"
echo "Output: ${OUT_DIR}"
echo "============================================================"

# Datasets and splits
DATASETS=("coco" "aokvqa" "gqa")
SPLITS=("adversarial" "popular" "random")

# Baseline run
echo ""
echo "=========================================="
echo "BASELINE EVALUATION"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        echo ""
        echo "--------------------------------------------------------"
        echo "Baseline: ${DATASET} - ${SPLIT}"
        echo "--------------------------------------------------------"

        DATASET_FILE="${POPE_DATA_DIR}/${DATASET}_${SPLIT}.json"
        OUTPUT_DIR="${OUT_DIR}/baseline/${DATASET}_${SPLIT}"

        mkdir -p "${OUTPUT_DIR}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n ${CONDA_ENV} python -u srf/eval.py \
            --method baseline \
            --model "${MODEL}" \
            --datasets pope_vcd \
            --pope_vcd_file "${DATASET_FILE}" \
            --pope_vcd_name "${DATASET}_${SPLIT}" \
            --eval_method generation \
            --output "${OUTPUT_DIR}/" \
            | tee "${OUTPUT_DIR}/baseline.log"
    done
done

echo ""
echo "=========================================="
echo "SRF EVALUATION"
echo "=========================================="

# SRF run
for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        echo ""
        echo "--------------------------------------------------------"
        echo "SRF: ${DATASET} - ${SPLIT}"
        echo "--------------------------------------------------------"

        DATASET_FILE="${POPE_DATA_DIR}/${DATASET}_${SPLIT}.json"
        OUTPUT_DIR="${OUT_DIR}/srf/${DATASET}_${SPLIT}"

        mkdir -p "${OUTPUT_DIR}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n ${CONDA_ENV} python -u srf/eval.py \
            --method srf \
            --model "${MODEL}" \
            --datasets pope_vcd \
            --pope_vcd_file "${DATASET_FILE}" \
            --pope_vcd_name "${DATASET}_${SPLIT}" \
            --eval_method generation \
            --alpha 4.0 \
            --eps 0.2 \
            --phase both \
            --layer_start 8 \
            --layer_end 15 \
            --head_top_k_pct 0.20 \
            --output "${OUTPUT_DIR}/" \
            | tee "${OUTPUT_DIR}/srf.log"
    done
done

echo ""
echo "============================================================"
echo "EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results structure:"
echo "  ${OUT_DIR}/baseline/{dataset}_{split}/pope.json"
echo "  ${OUT_DIR}/srf/{dataset}_{split}/pope.json"
echo ""
echo "Example:"
echo "  ${OUT_DIR}/baseline/coco_adversarial/pope.json"
echo "============================================================"
