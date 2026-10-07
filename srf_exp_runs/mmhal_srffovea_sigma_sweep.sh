#!/bin/bash
#SBATCH --job-name=mmhal_fovea
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fovea_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fovea_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_fovea_sweep"

# σ sweep: 5→30 covers 1.5%→9% of the 336px image width
SIGMAS=(5 10 15 20 30)

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench — SRF-Fovea σ sweep {5,10,15,20,30}, LLaVA-1.5-7B"
echo "Config: α=0.5, naa=1.0, le=20, additive, fit=0.25, pth=0.27"
echo "Started: $(date)"
echo "======================================================"

for SIGMA in "${SIGMAS[@]}"; do
    echo ""
    echo ">>> SRF-Fovea σ=${SIGMA}"
    python srf/eval.py \
        --method srffovea \
        --model "$MODEL" \
        --datasets mmhalbench \
        --fovea_sigma "$SIGMA" \
        --alpha 0.5 --layer_end 20 \
        --neg_absent_alpha 1.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --openai_model gpt-4o \
        --output "$OUT/s${SIGMA}"
    echo "  σ=${SIGMA} done at $(date +%H:%M:%S)"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def load(d):
    for fname in ["mmhalbench_scores.json"]:
        p = pathlib.Path(d) / fname
        if p.exists():
            data = json.loads(p.read_text())
            if "score" in data:
                return data["score"], data.get("hal_pct", float("nan"))
            key = list(data.keys())[0]
            return data[key]["score"], data[key].get("hal_pct", float("nan"))
    return None, None

# Reference: existing baseline and SRF
refs = [
    ("Baseline",   "results/mmhalbench_full/baseline"),
    ("SRF",        "results/mmhalbench_full/srf"),
]
sweep = [(f"SRF-Fovea σ={s}", f"results/mmhalbench_fovea_sweep/s{s}") for s in [5,10,15,20,30]]

print("\n" + "="*58)
print("MMHal-Bench: SRF-Fovea σ Sweep (LLaVA-1.5-7B, n=96)")
print("="*58)
print(f"{'Method':<24} {'Score':>6} {'Hal%':>7} {'Δ Score':>8}")
print("-"*48)

base_score = None
for label, path in refs + sweep:
    score, hal = load(path)
    if score is None:
        print(f"{label:<24} {'N/A':>6} {'N/A':>7} {'N/A':>8}")
        continue
    if base_score is None:
        base_score = score
    delta = f"{score - base_score:+.2f}" if label != "Baseline" else "—"
    print(f"{label:<24} {score:>6.2f} {hal:>6.1f}% {delta:>8}")

print("="*58)
print()
print("ILVAD Table 1 ref (LLaVA-1.5, gpt-4-0314 scoring):")
print("  Baseline: 2.01  VCD: 2.20  AGLA: 2.14  ILVAD: 2.22")
PYEOF
