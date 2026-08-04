#!/bin/bash
#SBATCH --job-name=mmhal_srfe_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_srfe_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_srfe_sweep_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_srfe_sweep"
N=20

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench SRF-E gamma sweep — n=$N samples"
echo "Base config: alpha=0.5 fit=0.25 naa=1.0 pth=0.27 additive le=20"
echo "Started: $(date)"
echo "======================================================"

for gamma in 0.1 0.2 0.3 0.5 1.0 1.5; do
    echo ""
    echo ">>> gamma=$gamma"
    python srf/eval.py \
        --method srfe \
        --gamma $gamma \
        --model "$MODEL" \
        --datasets mmhalbench \
        --mmhalbench_n $N \
        --alpha 0.5 --layer_end 20 \
        --neg_absent_alpha 1.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --openai_model gpt-4o \
        --output "$OUT/g${gamma}"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

root = pathlib.Path("results/mmhalbench_srfe_sweep")

# Load SRF baseline for comparison
srf_sf = pathlib.Path("results/mmhalbench_alpha_sweep/a0.5/mmhalbench_scores.json")
srf_score, srf_hal = None, None
if srf_sf.exists():
    d = json.loads(srf_sf.read_text())
    key = list(d.keys())[0]
    srf_score = d[key]["score"] if isinstance(d[key], dict) else d.get("score")
    srf_hal   = d[key]["hal_pct"] if isinstance(d[key], dict) else d.get("hal_pct")

print(f"\n{'gamma':<8}  {'Score':>6}  {'Hal%':>6}  {'Δ vs SRF':>10}")
print("-" * 38)
if srf_score:
    print(f"  {'SRF':<6}  {srf_score:>6.2f}  {srf_hal:>6.1f}%  {'(base)':>10}")

best_score, best_gamma = -1, ""
for gamma in ["0.1", "0.2", "0.3", "0.5", "1.0", "1.5"]:
    sf = root / f"g{gamma}" / "mmhalbench_scores.json"
    if not sf.exists():
        print(f"  {gamma:<6}  {'—':>6}  {'—':>6}")
        continue
    d = json.loads(sf.read_text())
    # SRF-E stores scores keyed by gamma value
    key = str(float(gamma))
    if key in d:
        score = d[key]["score"]
        hal   = d[key]["hal_pct"]
    else:
        key = list(d.keys())[0]
        score = d[key]["score"] if isinstance(d[key], dict) else d.get("score", 0)
        hal   = d[key]["hal_pct"] if isinstance(d[key], dict) else d.get("hal_pct", 0)
    delta = f"{score - srf_score:+.2f}" if srf_score else "—"
    marker = " ◀" if score > best_score else ""
    if score > best_score:
        best_score = score
        best_gamma = gamma
    print(f"  {gamma:<6}  {score:>6.2f}  {hal:>6.1f}%  {delta:>10}{marker}")

print(f"\nBest SRF-E gamma: {best_gamma}  Score={best_score:.2f}")
PYEOF
