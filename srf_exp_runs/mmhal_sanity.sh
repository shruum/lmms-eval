#!/bin/bash
#SBATCH --job-name=mmhal_sanity
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_sanity_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_sanity_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_sanity"
N=5   # only 5 samples for sanity check

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench sanity check — n=$N samples"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> Pass 1: baseline (greedy, no intervention)"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_n $N \
    --openai_model gpt-4o \
    --output "$OUT/baseline"

echo ""
echo ">>> Pass 2: SRF additive (best config)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_n $N \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 0.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --openai_model gpt-4o \
    --output "$OUT/srf"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

echo ""
echo "=== Sanity check — side-by-side responses (first $N samples) ==="
python - <<'PYEOF'
import json, pathlib

base_path = pathlib.Path("results/mmhalbench_sanity/baseline/mmhalbench_responses.json")
srf_path  = pathlib.Path("results/mmhalbench_sanity/srf/mmhalbench_responses.json")

base = json.loads(base_path.read_text()) if base_path.exists() else []
srf  = json.loads(srf_path.read_text())  if srf_path.exists()  else []

for i, (b, s) in enumerate(zip(base, srf)):
    print(f"\n{'='*60}")
    print(f"Sample {i+1} | type={b['question_type']} | content={b['image_content']}")
    print(f"Q: {b['question']}")
    print(f"GT: {b['gt_answer']}")
    print(f"Baseline: {b['model_answer']}")
    print(f"SRF:      {s['model_answer']}")
PYEOF
