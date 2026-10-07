#!/bin/bash
#SBATCH --job-name=mmhal_calib
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_calib_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_calib_%j.err

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhal_calib_validation"

# Fixed params — NO neg_absent_alpha, additive boost
FIXED="--method srffovea --model $MODEL --datasets mmhalbench \
    --fovea_sigma 20 --layer_start 8 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27 \
    --openai_model gpt-4o"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench calibration validation — 3 configs, n=96"
echo "MME winner: le=20 htk=0.20 | Extended: le=32 htk=0.35"
echo "Started: $(date)"
echo "======================================================"

# Run 1: MME winner + MMHal alpha (clean, no naa)
echo ""; echo ">>> [1] le=20  htk=0.20  alpha=0.3  $(date)"
python -u srf/eval.py $FIXED \
    --alpha 0.3 --layer_end 20 --head_top_k_pct 0.20 \
    --output "$OUT/R1_le20_htk020_a03"

# Run 2: Extended range (Config F) + MMHal alpha
echo ""; echo ">>> [2] le=32  htk=0.35  alpha=0.3  $(date)"
python -u srf/eval.py $FIXED \
    --alpha 0.3 --layer_end 32 --head_top_k_pct 0.35 \
    --output "$OUT/R2_le32_htk035_a03"

# Run 3: Extended range + slightly higher alpha
echo ""; echo ">>> [3] le=32  htk=0.35  alpha=0.4  $(date)"
python -u srf/eval.py $FIXED \
    --alpha 0.4 --layer_end 32 --head_top_k_pct 0.35 \
    --output "$OUT/R3_le32_htk035_a04"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 - <<'PYEOF'
import json, pathlib

base = pathlib.Path("results/mmhal_calib_validation")
configs = [
    ("R1_le20_htk020_a03", "le=20 htk=0.20 α=0.3  [MME winner]"),
    ("R2_le32_htk035_a03", "le=32 htk=0.35 α=0.3  [extended]"),
    ("R3_le32_htk035_a04", "le=32 htk=0.35 α=0.4  [extended+α]"),
]

print("\n" + "="*62)
print("MMHal-Bench calibration validation (n=96, gpt-4o)")
print(f"{'Config':<30}  {'Score':>6}  {'Hal%':>6}")
print("-"*48)
print(f"  {'prev best (a=0.3, le=20, naa=1.0)':<30}  {'2.438':>6}  {'56.3%':>6}  ← ref")
print(f"  {'baseline':<30}  {'2.21':>6}  {'60.4%':>6}  ← ref")
print("-"*48)

results = []
for key, label in configs:
    sf = base / key / "mmhalbench_scores.json"
    if not sf.exists():
        print(f"  {label:<30}  [no scores]")
        continue
    d = json.loads(sf.read_text())
    if "score" not in d:
        d = list(d.values())[0]
    score = d["score"]
    hal   = d.get("hal_pct", float("nan"))
    marker = ""
    print(f"  {label:<30}  {score:>6.3f}  {hal:>5.1f}%{marker}")
    results.append((score, label))

if results:
    best = max(results, key=lambda x: x[0])
    print("-"*48)
    print(f"  BEST: {best[1]}  →  {best[0]:.3f}")
print("="*62)
PYEOF
