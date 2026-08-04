#!/bin/bash
#SBATCH --job-name=sweep_srf
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/sweep_srf_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/sweep_srf_%j.err

set -e

source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm

export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/srf"
N=500

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "SRF alpha + layer_end sweep — LLaVA 7B, POPE adversarial n=$N"
echo "Started: $(date)"
echo "======================================================"

# ── Alpha sweep (layer_end fixed at 20) ───────────────────────────────────────
for ALPHA in 1.2 1.5 2.0 2.5 3.0; do
    echo -e "\n>>> alpha=$ALPHA  layer_end=20"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets pope --pope_splits adversarial \
        --n_pope $N \
        --alpha $ALPHA \
        --output "$OUT/alpha${ALPHA}_le20"
done

# ── Layer_end sweep (alpha fixed at best from above — update after first sweep) ─
for LE in 12 14 16 18 20; do
    echo -e "\n>>> alpha=2.0  layer_end=$LE"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets pope --pope_splits adversarial \
        --n_pope $N \
        --alpha 2.0 \
        --layer_end $LE \
        --output "$OUT/alpha2.0_le${LE}"
done

echo -e "\n======================================================"
echo "All done: $(date)"
echo "======================================================"
