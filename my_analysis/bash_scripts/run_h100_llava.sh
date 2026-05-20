#!/bin/bash
# =============================================================================
# LLaVA-1.5 evaluation — POPE / MMVP / VLM-Bias (all groups)
# Methods: baseline, srf_clip_basic, srf_clip, srf_hssa
#
# Layer defaults follow ClearSight (arXiv 2503.13107) for LLaVA-1.5 / 32-layer LLaMA:
#   vaf_layer_start=9, vaf_layer_end=14
#
# Usage:
#   bash run_h100_llava.sh            # 7B model (default)
#   bash run_h100_llava.sh 13B        # 13B model
#   bash run_h100_llava.sh 7B /custom/output/dir
# =============================================================================
set -e
cd "$(dirname "$0")"

# ---- Model config ----
MODEL_SIZE=${1:-7B}
if [ "$MODEL_SIZE" = "13B" ]; then
    MODEL_ID="llava-hf/llava-1.5-13b-hf"
else
    MODEL_ID="llava-hf/llava-1.5-7b-hf"
fi

# ---- Paths ----
RESULTS=${2:-/volumes2/mllm/lmms-eval/results/h100/llava_${MODEL_SIZE}}
SCRIPT="python run_llava_eval.py"

# ---- Shared args ----
MODEL_ARGS="--model $MODEL_ID"
SWEEP="--sweep 1.5 2.0 4.0 8.0"

# SRF hyperparams — ClearSight-aligned layer range for LLaVA-1.5 (32-layer LLaMA)
SRF_ARGS="--sal_top_k 0.3 --srf_background_eps 0.1 --vaf_beta 0.1 --vaf_layer_start 9 --vaf_layer_end 14"
VHR_ARGS="--vhr_top_k 0.5"

# ---- Dataset sizes ----
# POPE adversarial: 500 total (all 3 categories × ~167)
# MMVP: 150 pairs (full dataset)
# VLM Bias: all groups, cap 500 per group
N_POPE=500
N_MMVP=150
N_BIAS=500

echo "=============================================="
echo "  Model  : $MODEL_ID"
echo "  Results: $RESULTS"
echo "=============================================="

# ==============================================================================
# POPE (adversarial / popular / random — all 3 categories)
# ==============================================================================
echo ""
echo "=== POPE ==="
OUT=$RESULTS/pope

echo "[POPE 1/4] baseline"
$SCRIPT $MODEL_ARGS --dataset pope --method baseline \
    --n_samples $N_POPE --output_dir $OUT

echo "[POPE 2/4] srf_clip_basic"
$SCRIPT $MODEL_ARGS --dataset pope --method srf_clip_basic $SWEEP \
    $VHR_ARGS --sal_top_k 0.3 \
    --n_samples $N_POPE --output_dir $OUT

echo "[POPE 3/4] srf_clip"
$SCRIPT $MODEL_ARGS --dataset pope --method srf_clip $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_POPE --output_dir $OUT

echo "[POPE 4/4] srf_hssa"
$SCRIPT $MODEL_ARGS --dataset pope --method srf_hssa $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_POPE --output_dir $OUT

# ==============================================================================
# MMVP (all 150 pairs)
# ==============================================================================
echo ""
echo "=== MMVP ==="
OUT=$RESULTS/mmvp

echo "[MMVP 1/4] baseline"
$SCRIPT $MODEL_ARGS --dataset mmvp --method baseline \
    --n_samples $N_MMVP --output_dir $OUT

echo "[MMVP 2/4] srf_clip_basic"
$SCRIPT $MODEL_ARGS --dataset mmvp --method srf_clip_basic $SWEEP \
    $VHR_ARGS --sal_top_k 0.3 \
    --n_samples $N_MMVP --output_dir $OUT

echo "[MMVP 3/4] srf_clip"
$SCRIPT $MODEL_ARGS --dataset mmvp --method srf_clip $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_MMVP --output_dir $OUT

echo "[MMVP 4/4] srf_hssa"
$SCRIPT $MODEL_ARGS --dataset mmvp --method srf_hssa $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_MMVP --output_dir $OUT

# ==============================================================================
# VLM Bias (all topic groups, cap N_BIAS per group)
# ==============================================================================
echo ""
echo "=== VLM Bias (all groups) ==="
OUT=$RESULTS/vlm_bias

echo "[BIAS 1/4] baseline"
$SCRIPT $MODEL_ARGS --dataset vlm_bias --method baseline \
    --n_samples $N_BIAS --output_dir $OUT

echo "[BIAS 2/4] srf_clip_basic"
$SCRIPT $MODEL_ARGS --dataset vlm_bias --method srf_clip_basic $SWEEP \
    $VHR_ARGS --sal_top_k 0.3 \
    --n_samples $N_BIAS --output_dir $OUT

echo "[BIAS 3/4] srf_clip"
$SCRIPT $MODEL_ARGS --dataset vlm_bias --method srf_clip $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_BIAS --output_dir $OUT

echo "[BIAS 4/4] srf_hssa"
$SCRIPT $MODEL_ARGS --dataset vlm_bias --method srf_hssa $SWEEP \
    $VHR_ARGS $SRF_ARGS \
    --n_samples $N_BIAS --output_dir $OUT

echo ""
echo "All done. Results in $RESULTS"
