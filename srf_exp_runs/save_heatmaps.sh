#!/bin/bash
#SBATCH --job-name=heatmaps
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/heatmaps_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/heatmaps_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

# 8 adversarial samples, default v3 thresholds (full_img_thresh=0.20, patch_thresh=0.27)
python srf/saliency/save_llava_heatmaps.py \
    --n 8 \
    --split adversarial \
    --repope_dir data/repope \
    --output results/saliency_heatmaps \
    --full_img_thresh 0.20 \
    --patch_thresh 0.27 \
    --backup none \
    --seed 42

echo "Heatmaps saved."
