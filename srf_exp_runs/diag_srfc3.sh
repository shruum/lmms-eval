#!/bin/bash
#SBATCH --job-name=diag_srfc3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/diag_srfc3_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/diag_srfc3_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "SRF-C v3 diagnostic — 10 MME existence samples"
echo "Compares: Baseline / SRF-E / SRF-C v2 (pixel) / SRF-C v3 (embed-space)"
echo "Started: $(date)"
echo "======================================================"

python srf/diag_srfc3.py --n 10 --gamma 0.3

echo ""
echo "Done: $(date)"
