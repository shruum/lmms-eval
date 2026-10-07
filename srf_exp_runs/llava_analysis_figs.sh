#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --job-name=llava_analysis
#SBATCH --output=srf_exp_runs/logs/llava_analysis_%j.out
#SBATCH --error=srf_exp_runs/logs/llava_analysis_%j.err
#SBATCH --exclude=gcn58

# Step 1: Fig 2 equivalent — collect attention stats from MMHal + generate figure
#   Output: analysis/llava_analysis_figure.png
#
# Step 2: Fig 3 equivalent — scan all MMHal candidates (Image | Attn | SRF)
#   Output: analysis/candidates/*.png  → pick two, then run with --idx_a --idx_b
#
# After step 2, inspect analysis/candidates/, choose two, and run locally:
#   python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011

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

echo ""
echo "=== Step 1: Fig 2 — modality attention analysis (MMHal, 50 samples) ==="
python -u srf/analysis/llava_modality_attention.py --collect --n 50

echo ""
echo "=== Step 2: Fig 3 — scan MMHal candidates (Image | Attn | SRF) ==="
python -u srf/analysis/llava_fig3_routing.py --scan

echo ""
echo "======================================================"
echo "Done: $(date)"
echo ""
echo "Fig 2 output:  analysis/llava_analysis_figure.png"
echo "Fig 3 candidates: analysis/candidates/"
echo ""
echo "Next: inspect candidates, pick two indices, then run:"
echo "  python srf/analysis/llava_fig3_routing.py --idx_a 003 --idx_b 011"
echo "======================================================"
