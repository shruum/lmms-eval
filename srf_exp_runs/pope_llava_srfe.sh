#!/bin/bash
#SBATCH --job-name=pope_srfe
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/pope_srfe_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/pope_srfe_%j.err

set -e

source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/pope_llava_full"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Started: $(date)"
echo "======================================================"

echo -e "\n>>> SRF-E  gamma=3.0"
python srf/eval.py \
    --method srfe \
    --model "$MODEL" \
    --datasets pope \
    --gamma 3.0 \
    --output "$OUT/srfe_g3"

echo -e "\n======================================================"
echo "All done: $(date)"
echo "======================================================"
