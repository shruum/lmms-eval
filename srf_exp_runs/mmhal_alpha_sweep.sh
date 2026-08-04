#!/bin/bash
#SBATCH --job-name=mmhal_alpha_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_alpha_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_alpha_sweep_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_alpha_sweep"
N=20

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench alpha sweep — n=$N samples, additive patch"
echo "Started: $(date)"
echo "======================================================"

# ── Baseline (greedy, no intervention) ────────────────────────────────────────
echo ""
echo ">>> baseline (greedy)"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mmhalbench \
    --mmhalbench_n $N \
    --openai_model gpt-4o \
    --output "$OUT/baseline"

# ── SRF alpha sweep ────────────────────────────────────────────────────────────
for alpha in 0.5 1.0 2.0 3.0; do
    echo ""
    echo ">>> SRF alpha=$alpha"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets mmhalbench \
        --mmhalbench_n $N \
        --alpha $alpha --layer_end 20 \
        --neg_absent_alpha 0.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --openai_model gpt-4o \
        --output "$OUT/a${alpha}"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

# ── Summary table ─────────────────────────────────────────────────────────────
python - <<'PYEOF'
import json, pathlib

root = pathlib.Path("results/mmhalbench_alpha_sweep")
print(f"\n{'Method':<12}  {'Score':>6}  {'Hal%':>6}  {'Δ Score':>8}  {'Δ Hal%':>8}")
print("-" * 50)

base_score, base_hal = None, None
rows = []

for tag in ["baseline"] + [f"a{a}" for a in ["0.5", "1.0", "2.0", "3.0"]]:
    sf = root / tag / "mmhalbench_scores.json"
    if not sf.exists():
        print(f"  {tag}: no scores file")
        continue
    d = json.loads(sf.read_text())
    # scores dict is keyed by gamma (0.0 for srf/baseline)
    key = list(d.keys())[0] if isinstance(d, dict) and list(d.keys())[0] != "score" else None
    if key is not None and isinstance(d[key], dict):
        score = d[key]["score"]
        hal   = d[key]["hal_pct"]
    else:
        score = d.get("score", 0)
        hal   = d.get("hal_pct", 0)
    rows.append((tag, score, hal))
    if tag == "baseline":
        base_score, base_hal = score, hal

for tag, score, hal in rows:
    ds = f"{score - base_score:+.2f}" if base_score is not None and tag != "baseline" else "—"
    dh = f"{hal - base_hal:+.1f}"     if base_hal  is not None and tag != "baseline" else "—"
    label = "baseline" if tag == "baseline" else f"α={tag[1:]}"
    print(f"  {label:<12}  {score:>6.2f}  {hal:>6.1f}%  {ds:>8}  {dh:>8}")

PYEOF

# ── Saliency heatmaps for MMHal-Bench samples ────────────────────────────────
echo ""
echo ">>> Generating CLIP saliency heatmaps for first 8 MMHal-Bench samples"
python srf/saliency/save_mmhal_heatmaps.py \
    --n 8 \
    --output results/saliency_heatmaps/mmhal \
    --full_img_thresh 0.25 \
    --patch_thresh 0.27
