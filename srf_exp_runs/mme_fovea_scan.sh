#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=mme_fovea_scan
#SBATCH --output=srf_exp_runs/logs/mme_fovea_scan_%j.out
#SBATCH --error=srf_exp_runs/logs/mme_fovea_scan_%j.err
#SBATCH --exclude=gcn58

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "======================================================"

python -u srf/analysis/mme_fovea_scan.py

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Candidates: analysis/mme_candidates/"
echo "======================================================"
