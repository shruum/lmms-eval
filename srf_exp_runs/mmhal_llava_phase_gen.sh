#!/bin/bash
#SBATCH --job-name=mmhal_gen
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_gen_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_gen_%j.err

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MMHAL_JSON="/home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json"
OUT="results/mmhal_phase_gen"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench — phase=both vs phase=generation"
echo "Fixed patch: 6 heads, layers=8-20, alpha=0.5, eps=0.2, sys_beta=0.30"
echo "Baseline already known: score=2.28  hal=60.4% (n=96)"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> 1/2 SRF phase=both (prefill + generation)"
python -u srf/eval.py \
    --method srf \
    --phase both \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_json "$MMHAL_JSON" \
    --alpha 0.5 --layer_end 20 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --neg_absent_alpha 1.0 \
    --openai_model gpt-4o \
    --output "$OUT/srf_both"

echo ""
echo ">>> 2/2 SRF phase=generation only"
python -u srf/eval.py \
    --method srf \
    --phase generation \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_json "$MMHAL_JSON" \
    --alpha 0.5 --layer_end 20 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --neg_absent_alpha 1.0 \
    --openai_model gpt-4o \
    --output "$OUT/srf_gen"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 - <<'PYEOF'
import json, pathlib

def load(d):
    sf = pathlib.Path(d) / "mmhalbench_scores.json"
    if not sf.exists(): return None, None, None
    data = json.loads(sf.read_text())
    return data.get("score"), data.get("hal_pct"), data.get("n")

rows = [
    ("Baseline (known)",    None),
    ("SRF phase=both",      "results/mmhal_phase_gen/srf_both"),
    ("SRF phase=gen",       "results/mmhal_phase_gen/srf_gen"),
]

print("\n" + "="*65)
print("MMHal-Bench — phase=both vs phase=generation (fixed patch)")
print("Fixed: 6 heads, layers 8-20, alpha=0.5, eps=0.2, sys_beta=0.30")
print("="*65)
print(f"{'Method':<24} {'Score':>6} {'Hal%':>7} {'n':>4}")
print("-"*46)

# Print known baseline
print(f"{'Baseline (known)':<24}  2.28   60.4%   96")

for label, d in rows[1:]:
    s, h, n = load(d)
    if s is None: print(f"{label:<24}  no results"); continue
    delta_s = s - 2.28125
    delta_h = h - 60.4167
    print(f"{label:<24} {s:6.2f} {h:7.1f}%  {n:3d}   (Δscore={delta_s:+.2f}  Δhal={delta_h:+.1f}pp)")

print("-"*46)
print("Paper ref (buggy fovea s20, gen):   SRF=2.40  hal=56.2%")
print("="*65)
PYEOF
