#!/bin/bash
# MME Parameter Sweep - Qwen-VL-Chat (GPU 0)
# Based on POPE best parameters: layers 8-14, alpha 1.5-2.5, absence-aware

export CUDA_VISIBLE_DEVICES=0
cd /home/anna2/shruthi/lmms-eval

echo "========================================"
echo "Qwen-VL-Chat MME - PARAMETER SWEEP"
echo "========================================"
echo "GPU: 0"
echo "Total experiments: 8"
echo "========================================"

# Create results directory
mkdir -p results/qwen_mme_sweep

# Experiment 1: Baseline from POPE best (params_best.py)
echo ""
echo "[1/8] SRF - POPE best (layers 8-14, alpha=1.5)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 1.5 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp01_pope_best.log

# Experiment 2: Higher alpha (stronger boost)
echo ""
echo "[2/8] SRF - Stronger boost (alpha=2.0)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp02_alpha_2.0.log

# Experiment 3: Even higher alpha
echo ""
echo "[3/8] SRF - Aggressive boost (alpha=2.5)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.5 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp03_alpha_2.5.log

# Experiment 4: Wider layer range
echo ""
echo "[4/8] SRF - Wider fusion (layers 6-16)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 6 \
    --layer_end 16 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp04_wider_layers.log

# Experiment 5: More focused CLIP (top 15%)
echo ""
echo "[5/8] SRF - Focused CLIP (top_k=15%)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.15 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp05_clip_15pct.log

# Experiment 6: More heads (top 20%)
echo ""
echo "[6/8] SRF - More heads (top_k=20%)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/qwen_mme_sweep/exp06_head_20pct.log

# Experiment 7: SRF-E with moderate contrastive
echo ""
echo "[7/8] SRF-E - Moderate contrastive (beta=1.5)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srfe \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --beta 1.5 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp07_srfe_beta_1.5.log

# Experiment 8: SRF-E with strong contrastive
echo ""
echo "[8/8] SRF-E - Strong contrastive (beta=2.0)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srfe \
    --model Qwen/Qwen-VL-Chat \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --beta 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.15 \
    2>&1 | tee results/qwen_mme_sweep/exp08_srfe_beta_2.0.log

echo ""
echo "========================================"
echo "✓ Qwen-VL-Chat MME sweep complete!"
echo "Results: results/qwen_mme_sweep/"
echo "========================================"
