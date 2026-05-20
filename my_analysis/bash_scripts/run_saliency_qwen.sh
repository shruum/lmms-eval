#!/bin/bash
set -e
cd "$(dirname "$0")"

RESULTS=/volumes2/mllm/lmms-eval/results/new_datasets
#
echo "========================================="
echo " CV-Bench spatial evaluation"
echo "========================================="

#echo "[1] POPE-CLIP"
#python run_qwen_eval.py --dataset pope --method srf_clip --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/pope
#echo "[2] POPE-HSSA"
#python run_qwen_eval.py --dataset pope --method srf_hssa --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/pope
echo "[3] BIAS-CLIP"
python run_qwen_eval.py --dataset vlm_bias --method srf_clip --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/bias
echo "[4] BIAS-HSSA"
python run_qwen_eval.py --dataset vlm_bias --method srf_hssa --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/bias
echo "[5] MMVP-CLIP"
python run_qwen_eval.py --dataset mmvp --method srf_clip --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/mmvp
echo "[6] MMVP-HSSA"
python run_qwen_eval.py --dataset mmvp --method srf_hssa --sweep 1.5 2.0 4.0 8.0 --n_samples 50 --output_dir $RESULTS/mmvp


#
#echo "[1/4] Baseline..."
#python run_qwen_eval.py \
#    --dataset cv_bench \
#    --method baseline \
#    --n_samples 50 \
#    --output_dir $RESULTS/base
#
#echo "[2/4] visboost_heads sweep..."
#python run_qwen_eval.py \
#    --dataset cv_bench \
#    --method vhr_boost \
#    --sweep 2.0 4.0 8.0 \
#    --vhr_top_k 0.5 \
#    --n_samples 50 \
#    --output_dir $RESULTS/visboost_heads
#
#echo "[3/4] vaf sweep..."
#python run_qwen_eval.py \
#    --dataset cv_bench \
#    --method vaf \
#    --sweep 2.0 4.0 8.0 \
#    --vhr_top_k 0.5 \
#    --n_samples 50 \
#    --output_dir $RESULTS/vaf
#
#echo "[4/4] clip_salience sweep..."
#python run_qwen_eval.py \
#    --dataset cv_bench \
#    --method clip_salience \
#    --sweep 2.0 4.0 8.0 \
#    --vhr_top_k 0.5 \
#    --sal_top_k 0.3 \
#    --n_samples 50 \
#    --output_dir $RESULTS/clip_salience

echo ""
echo "All done/"
