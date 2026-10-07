#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --job-name=mmhal_best
#SBATCH --output=srf_exp_runs/logs/mmhal_best_%j.out
#SBATCH --error=srf_exp_runs/logs/mmhal_best_%j.err
#SBATCH --exclude=gcn58

# Optimal combined config from 1-D sweep:
#   α=0.3, k=100% (all heads), σ=30, le=16, phase=both, naa=0.0
# All per-dimension best values tested together for the first time.

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
unset OPENAI_API_KEY
cd /home/sgowda/workspace/SRF/lmms-eval

OUT="results/mmhal_best_combined"
mkdir -p "$OUT"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Config: srffovea  σ=30  α=0.3  k=100%  le=16  phase=both  naa=0.0"
echo "Started: $(date)"
echo "======================================================"

python -u srf/eval.py \
    --method          srffovea \
    --fovea_sigma     30 \
    --phase           both \
    --model           "llava-hf/llava-1.5-7b-hf" \
    --datasets        mmhalbench \
    --mmhalbench_json "/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json" \
    --alpha           0.3 \
    --layer_end       16 \
    --head_top_k_pct  1.00 \
    --neg_absent_alpha 0.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh    0.27 \
    --output          "$OUT"

echo "Done: $(date)"
echo "Response file: $OUT/mmhalbench_responses.json"
echo "Run mmhal_sweep_score.sh (or score manually) to get GPT-4o scores."
