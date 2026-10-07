#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=llava_vtar
#SBATCH --output=srf_exp_runs/logs/llava_vtar_%j.out
#SBATCH --error=srf_exp_runs/logs/llava_vtar_%j.err
#SBATCH --exclude=gcn58

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "VTAR quick check — 20 MMHal samples"
echo "Started: $(date)"
echo "======================================================"

python -u srf/analysis/llava_modality_attention.py --collect --n 20

echo "Done: $(date)"
echo "Figure: analysis/llava_analysis_figure.png"
echo "VTAR heatmap: analysis/llava_attn_data/rho_matrix.npy"
