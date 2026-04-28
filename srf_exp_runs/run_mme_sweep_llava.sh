#!/bin/bash
# MME Parameter Sweep - LLaVA-1.5-7B (GPU 1)
# Based on POPE findings: LLaVA needs different parameters (deeper layers)

export CUDA_VISIBLE_DEVICES=1
cd /home/anna2/shruthi/lmms-eval

echo "========================================"
echo "LLaVA MME - PARAMETER SWEEP"
echo "========================================"
echo "GPU: 1"
echo "Total experiments: 8"
echo "========================================"

# Create results directory
mkdir -p results/llava_mme_sweep

# Experiment 1: Current baseline (from MME full run - NO IMPROVEMENT)
echo ""
echo "[1/8] SRF - Original baseline (layers 10-14, alpha=2.0)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 10 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp01_baseline.log

# Experiment 2: Earlier layers (inspired by Qwen success)
echo ""
echo "[2/8] SRF - Earlier fusion (layers 8-14)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp02_layers_8_14.log

# Experiment 3: Wider range (LLaVA has 32 layers)
echo ""
echo "[3/8] SRF - Wider fusion (layers 8-16)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 8 \
    --layer_end 16 \
    --alpha 2.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp03_layers_8_16.log

# Experiment 4: Late fusion (from POPE idea3)
echo ""
echo "[4/8] SRF - Late fusion (layers 13-16)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 13 \
    --layer_end 16 \
    --alpha 3.0 \
    --clip_top_k_pct 0.25 \
    --head_top_k_pct 0.50 \
    2>&1 | tee results/llava_mme_sweep/exp04_late_fusion.log

# Experiment 5: Stronger boost (alpha=3.0)
echo ""
echo "[5/8] SRF - Stronger boost (alpha=3.0, layers 10-14)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 10 \
    --layer_end 14 \
    --alpha 3.0 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp05_alpha_3.0.log

# Experiment 6: More focused CLIP
echo ""
echo "[6/8] SRF - Focused CLIP (top_k=15%)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 8 \
    --layer_end 14 \
    --alpha 2.0 \
    --clip_top_k_pct 0.15 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp06_clip_15pct.log

# Experiment 7: SRF-E with moderate contrastive
echo ""
echo "[7/8] SRF-E - Moderate contrastive (beta=1.5)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srfe \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 10 \
    --layer_end 14 \
    --alpha 2.0 \
    --beta 1.5 \
    --clip_top_k_pct 0.20 \
    --head_top_k_pct 0.20 \
    2>&1 | tee results/llava_mme_sweep/exp07_srfe_beta_1.5.log

# Experiment 8: SRF-E with strong contrastive (from POPE idea3)
echo ""
echo "[8/8] SRF-E - Strong contrastive (beta=1.5, layers 13-16)"
/home/anna2/miniconda3/envs/mllm/bin/python srf/eval.py \
    --method srfe \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets mme \
    --layer_start 13 \
    --layer_end 16 \
    --alpha 3.0 \
    --beta 1.5 \
    --clip_top_k_pct 0.25 \
    --head_top_k_pct 0.50 \
    2>&1 | tee results/llava_mme_sweep/exp08_srfe_late_fusion.log

echo ""
echo "========================================"
echo "✓ LLaVA MME sweep complete!"
echo "Results: results/llava_mme_sweep/"
echo "========================================"
