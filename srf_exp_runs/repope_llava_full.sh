#!/bin/bash
#SBATCH --job-name=repope_llava
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repope_llava_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repope_llava_%j.err

set -e

source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/repope_llava_full"
REPOPE_DIR="data/repope"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Started: $(date)"
echo "======================================================"

# --- Baseline ---
echo -e "\n>>> BASELINE (RePOPE labels)"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets pope \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/baseline"

# --- SRF (best params: alpha=2.0, layer_end=20) ---
echo -e "\n>>> SRF (RePOPE labels)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets pope \
    --alpha 2.0 \
    --layer_end 20 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/srf"

# --- SRF-E (best params: gamma=0.3) ---
echo -e "\n>>> SRF-E  gamma=0.3 (RePOPE labels)"
python srf/eval.py \
    --method srfe \
    --model "$MODEL" \
    --datasets pope \
    --gamma 0.3 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT/srfe_g03"

echo -e "\n======================================================"
echo "All done: $(date)"
echo "Results in $OUT/"
echo "======================================================"
