#!/bin/bash
#SBATCH --job-name=fovea_test
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --nodelist=gcn36
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/fovea_test_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/fovea_test_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

echo "Node: $SLURMD_NODENAME  Job: $SLURM_JOB_ID  Started: $(date)"

python srf/eval.py \
    --method srffovea \
    --fovea_sigma 20 \
    --model "llava-hf/llava-1.5-7b-hf" \
    --datasets mmhalbench \
    --mmhalbench_n 2 \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "results/fovea_test_2sample"

echo "Done: $(date)"
