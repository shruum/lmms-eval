#!/bin/bash
#SBATCH --job-name=repope_best
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repope_best_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repope_best_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/repope_best"
REPOPE_DIR="data/repope"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Full RePOPE — best additive config: α=0.5 fit=0.25 naa=0"
echo "All 3 splits (adversarial, popular, random), full dataset"
echo "Started: $(date)"
echo "======================================================"

python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets pope \
    --pope_splits adversarial popular random \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 0.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --repope_dir "$REPOPE_DIR" \
    --output "$OUT"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
