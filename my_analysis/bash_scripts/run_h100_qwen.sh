#!/bin/bash
# =============================================================================
# Qwen2.5-VL evaluation — POPE / MMVP / VLM-Bias (Chess+Logos)
# Methods: baseline, srf_clip_basic, srf_clip, srf_hssa
#
# Usage:
#   bash run_h100_qwen.sh            # 3B model (default)
#   bash run_h100_qwen.sh 7B         # 7B model
#   bash run_h100_qwen.sh 3B /custom/output/dir
# =============================================================================
set -e
cd "$(dirname "$0")"

# ---- Model config ----
MODEL_SIZE=${1:-3B}
if [ "$MODEL_SIZE" = "7B" ]; then
    MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
else
    MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
fi

# ---- Paths ----
RESULTS=${2:-/volumes2/mllm/lmms-eval/results/h100/qwen_${MODEL_SIZE}}
SCRIPT="python run_qwen_eval.py"

# ---- Shared args ----
MODEL_ARGS="--model $MODEL_ID --max_pixels $((512 * 28 * 28))"
SWEEP="--sweep 1.5 2.0 4.0 8.0"
SRF_ARGS="--sal_top_k 0.3 --srf_background_eps 0.1 --vaf_beta 0.1 --vaf_layer_start 8 --vaf_layer_end 15"
VHR_ARGS="--vhr_top_k 0.5"

# ---- Dataset sizes ----
# POPE adversarial: 500 total
# MMVP: 150 pairs (full dataset)
# VLM Bias chess+logos: ~350; use 500 → loader caps at available
N_POPE=500
N_MMVP=150
N_BIAS=500

echo "=============================================="
echo "  Model  : $MODEL_ID"
echo "  Results: $RESULTS"
echo "=============================================="

# ==============================================================================
# POPE (adversarial)
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
# MMVP
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
# VLM Bias (Chess Pieces + Logos only)
# ==============================================================================
echo ""
echo "=== VLM Bias (Chess + Logos) ==="
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
