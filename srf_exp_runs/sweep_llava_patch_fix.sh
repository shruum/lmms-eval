#!/bin/bash
#SBATCH --job-name=patch_fix
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/patch_fix_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/patch_fix_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/patch_fix_repope"
REPOPE_DIR="data/repope"
# 500 samples, adversarial, RePOPE labels — all 4 runs use identical settings

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "LLaVA patch fix comparison — RePOPE adversarial n=500"
echo "Started: $(date)"
echo "======================================================"

# Baseline — no intervention
echo -e "\n>>> Baseline"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets pope --pope_splits adversarial \
    --n_pope 500 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/baseline"

# Current SRF — multiplicative with neg_absent_alpha=2.0 (the bug)
echo -e "\n>>> Current SRF (multiplicative, neg_absent_alpha=2.0)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets pope --pope_splits adversarial \
    --n_pope 500 \
    --alpha 2.0 --layer_end 20 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/srf_current"

# Fix 1: disable absent suppression (neg_absent_alpha=0)
echo -e "\n>>> Fix 1: neg_absent_alpha=0 (no absent suppression)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets pope --pope_splits adversarial \
    --n_pope 500 \
    --alpha 2.0 --layer_end 20 \
    --neg_absent_alpha 0 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/fix1_no_absent"

# Fix 2: additive pre-softmax logit (same as Qwen)
echo -e "\n>>> Fix 2: additive pre-softmax (llava_boost_mode=additive)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets pope --pope_splits adversarial \
    --n_pope 500 \
    --alpha 2.0 --layer_end 20 \
    --llava_boost_mode additive \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/fix2_additive"

echo -e "\n======================================================"
echo "All done: $(date)"
echo "======================================================"
