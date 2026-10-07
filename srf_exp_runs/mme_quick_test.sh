#!/bin/bash
#SBATCH --job-name=mme_test
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_test_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_test_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MME quick test — existence subtask only (60 Qs)"
echo "Baseline + SRF best config, LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

MODEL="llava-hf/llava-1.5-7b-hf"
MME_DIR="/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"

echo ""
echo ">>> Pass 1: baseline"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence \
    --output "results/mme_quicktest/baseline"

echo ""
echo ">>> Pass 2: SRF (best MMHal config)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "results/mme_quicktest/srf"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
