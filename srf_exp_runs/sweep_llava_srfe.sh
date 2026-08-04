#!/bin/bash
#SBATCH --job-name=sweep_srfe
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/sweep_srfe_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/sweep_srfe_%j.err

set -e

source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/srfe"
N=500

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "SRF-E gamma sweep — LLaVA 7B, POPE adversarial n=$N"
echo "Started: $(date)"
echo "======================================================"

# All gammas in one pass (eval.py loops over gammas per sample — efficient)
echo -e "\n>>> gamma sweep: 0.3 0.5 1.0 1.5 2.0 3.0"
python srf/eval.py \
    --method srfe \
    --model "$MODEL" \
    --datasets pope --pope_splits adversarial \
    --n_pope $N \
    --gamma 0.3 0.5 1.0 1.5 2.0 3.0 \
    --output "$OUT/gamma_sweep"

echo -e "\n======================================================"
echo "All done: $(date)"
echo "======================================================"
