#!/bin/bash
#SBATCH --job-name=mmhal_fixed
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fixed_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fixed_%j.err

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MMHAL_JSON="/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
OUT="results/mmhal_fixed_patch"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench — fixed patch (phase=both, 6 heads, lambda_bg, lambda_sys)"
echo "Previously reported: baseline score=2.01 hal=67.0%  SRF score=2.40 hal=56.2%"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> Baseline"
python -u srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_json "$MMHAL_JSON" \
    --openai_model gpt-4o \
    --output "$OUT/baseline"

echo ""
echo ">>> SRF fixed (phase=both, alpha=0.5, eps=0.2, sys_beta=0.30, 6 heads, additive)"
python -u srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_json "$MMHAL_JSON" \
    --alpha 0.5 --layer_end 20 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --openai_model gpt-4o \
    --output "$OUT/srf_fixed"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 - <<'PYEOF'
import json, pathlib

def load(d):
    p = pathlib.Path(d) / "mmhalbench_scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

base = load("results/mmhal_fixed_patch/baseline")
srf  = load("results/mmhal_fixed_patch/srf_fixed")

print("\n" + "="*65)
print("MMHal-Bench results — fixed patch (phase=both, 6 heads, lambda_bg+sys)")
print("="*65)
print(f"{'Method':<18} {'Score':>7} {'Hal%':>8}")
print("-"*40)
for label, d in [("Baseline", base), ("SRF fixed", srf)]:
    if d is None:
        print(f"{label:<18}  no results")
        continue
    score = d.get("score", float("nan"))
    hal   = d.get("hallucination_rate", float("nan")) * 100
    print(f"{label:<18} {score:7.2f}  {hal:7.1f}%")
print("-"*40)
print("Previously reported (buggy):  baseline score=2.01 hal=67.0%  SRF score=2.40 hal=56.2%")
print("="*65)
PYEOF
