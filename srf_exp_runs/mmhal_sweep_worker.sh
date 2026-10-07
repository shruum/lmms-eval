#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
# job-name, output, error, and export set by submission script via CLI

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
# Do NOT export OPENAI_API_KEY — scoring is done separately after all jobs finish
unset OPENAI_API_KEY
cd /home/sgowda/workspace/SRF/lmms-eval

OUT="results/mmhal_sweep/${SRF_TAG}"
mkdir -p "$OUT"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Tag: $SRF_TAG"
echo "sigma=$SRF_SIGMA  alpha=$SRF_ALPHA  k=$SRF_K  le=$SRF_LE  phase=$SRF_PHASE"
echo "Started: $(date)"
echo "======================================================"

python -u srf/eval.py \
    --method srffovea \
    --fovea_sigma "$SRF_SIGMA" \
    --phase        "$SRF_PHASE" \
    --model        "llava-hf/llava-1.5-7b-hf" \
    --datasets     mmhalbench \
    --mmhalbench_json "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json" \
    --alpha        "$SRF_ALPHA" \
    --layer_end    "$SRF_LE" \
    --head_top_k_pct "$SRF_K" \
    --neg_absent_alpha 0.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output       "$OUT"
# Scoring deferred — run mmhal_sweep_score.sh after all jobs complete

echo "Done: $(date)"
