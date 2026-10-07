#!/bin/bash
#SBATCH --job-name=mmhal_srfc3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_srfc3_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_srfc3_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_srfc3_sweep"
GAMMAS=(0.3 1.0 3.0)

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench — SRF-C v3 γ sweep {0.3,1.0,3.0}, LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

for GAMMA in "${GAMMAS[@]}"; do
    echo ""
    echo ">>> SRF-C v3 γ=${GAMMA}"
    python srf/eval.py \
        --method srfc3 \
        --model "$MODEL" \
        --datasets mmhalbench \
        --gamma "$GAMMA" \
        --alpha 0.5 --layer_end 20 \
        --neg_absent_alpha 1.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --openai_model gpt-4o \
        --output "$OUT/g${GAMMA}"
    echo "  γ=${GAMMA} done at $(date +%H:%M:%S)"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def load(d):
    p = pathlib.Path(d) / "mmhalbench_scores.json"
    if not p.exists(): return None, None
    data = json.loads(p.read_text())
    if "score" in data: return data["score"], data.get("hal_pct", float("nan"))
    key = list(data.keys())[0]
    return data[key]["score"], data[key].get("hal_pct", float("nan"))

refs = [
    ("Baseline",        "results/mmhalbench_full/baseline"),
    ("SRF",             "results/mmhalbench_full/srf"),
    ("SRF-Fovea σ=20",  "results/mmhalbench_fovea_sweep/s20"),
]
sweep = [(f"SRF-C v3 γ={g}", f"results/mmhalbench_srfc3_sweep/g{g}") for g in [0.3, 1.0, 3.0]]

print("\n" + "="*60)
print("MMHal-Bench: SRF-C v3 Sweep (LLaVA-1.5-7B, n=96)")
print("="*60)
print(f"{'Method':<26} {'Score':>6} {'Hal%':>7} {'Δ':>7}")
print("-"*50)
base_score = None
for label, path in refs + sweep:
    score, hal = load(path)
    if score is None:
        print(f"{label:<26} {'N/A':>6}"); continue
    if base_score is None: base_score = score
    delta = f"{score-base_score:+.2f}" if label != "Baseline" else "—"
    print(f"{label:<26} {score:>6.2f} {hal:>6.1f}% {delta:>7}")
print("="*60)
PYEOF
