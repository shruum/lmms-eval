#!/bin/bash
#SBATCH --job-name=mmhal_param_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_param_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_param_sweep_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_param_sweep"
N=20
ALPHA=0.5   # fixed — best from alpha sweep

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench full param sweep — n=$N, alpha=$ALPHA fixed"
echo "Sweeping: clip_fallback_thresh × neg_absent_alpha × clip_patch_thresh"
echo "Started: $(date)"
echo "======================================================"

# ── Phase 1: clip_fallback_thresh × neg_absent_alpha ─────────────────────────
# Most likely to matter: full-image gate controls when boosting fires at all.
echo ""
echo "=== Phase 1: clip_fallback_thresh x neg_absent_alpha ==="
for fit in 0.20 0.25 0.30; do
    for naa in 0.0 0.5 1.0; do
        tag="fit${fit}_naa${naa}"
        echo ""
        echo ">>> fit=$fit  naa=$naa"
        python srf/eval.py \
            --method srf \
            --model "$MODEL" \
            --datasets mmhalbench \
            --mmhalbench_n $N \
            --alpha $ALPHA --layer_end 20 \
            --neg_absent_alpha $naa \
            --llava_boost_mode additive \
            --clip_fallback_thresh $fit \
            --clip_patch_thresh 0.27 \
            --openai_model gpt-4o \
            --output "$OUT/phase1/$tag"
    done
done

# ── Phase 2: clip_patch_thresh (backup gate) ──────────────────────────────────
# Fix best fit from Phase 1 (use 0.25 as starting point from POPE) and sweep patch_thresh.
echo ""
echo "=== Phase 2: clip_patch_thresh sweep (fit=0.25, naa=0.0) ==="
for pth in 0.22 0.25 0.27 0.30 0.33; do
    tag="pth${pth}"
    echo ""
    echo ">>> patch_thresh=$pth"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets mmhalbench \
        --mmhalbench_n $N \
        --alpha $ALPHA --layer_end 20 \
        --neg_absent_alpha 0.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh $pth \
        --openai_model gpt-4o \
        --output "$OUT/phase2/$tag"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

# ── Summary ───────────────────────────────────────────────────────────────────
python - <<'PYEOF'
import json, pathlib

def load_score(d):
    sf = d / "mmhalbench_scores.json"
    if not sf.exists():
        return None, None
    data = json.loads(sf.read_text())
    # scores file is {gamma: {score, hal_pct}} or {score, hal_pct} directly
    if "score" in data:
        return data["score"], data["hal_pct"]
    key = list(data.keys())[0]
    return data[key]["score"], data[key]["hal_pct"]

root = pathlib.Path("results/mmhalbench_param_sweep")

print("\n=== Phase 1: clip_fallback_thresh × neg_absent_alpha (alpha=0.5, patch_thresh=0.27) ===")
print(f"{'fit':>5}  {'naa':>5}  {'Score':>6}  {'Hal%':>6}")
print("-" * 30)
best_p1_score, best_p1_tag = -1, ""
for fit in ["0.20", "0.25", "0.30"]:
    for naa in ["0.0", "0.5", "1.0"]:
        tag = f"fit{fit}_naa{naa}"
        d = root / "phase1" / tag
        score, hal = load_score(d)
        if score is None:
            print(f"  {fit:>5}  {naa:>5}  {'—':>6}  {'—':>6}")
        else:
            marker = " ◀" if score > best_p1_score else ""
            if score > best_p1_score:
                best_p1_score = score
                best_p1_tag = tag
            print(f"  {fit:>5}  {naa:>5}  {score:>6.2f}  {hal:>6.1f}%{marker}")

print(f"\nBest Phase 1: {best_p1_tag}  Score={best_p1_score:.2f}")

print("\n=== Phase 2: clip_patch_thresh (alpha=0.5, fit=0.25, naa=0.0) ===")
print(f"{'patch_thresh':>12}  {'Score':>6}  {'Hal%':>6}")
print("-" * 30)
best_p2_score, best_p2_tag = -1, ""
for pth in ["0.22", "0.25", "0.27", "0.30", "0.33"]:
    tag = f"pth{pth}"
    d = root / "phase2" / tag
    score, hal = load_score(d)
    if score is None:
        print(f"  {pth:>12}  {'—':>6}  {'—':>6}")
    else:
        marker = " ◀" if score > best_p2_score else ""
        if score > best_p2_score:
            best_p2_score = score
            best_p2_tag = tag
        print(f"  {pth:>12}  {score:>6.2f}  {hal:>6.1f}%{marker}")

print(f"\nBest Phase 2: {best_p2_tag}  Score={best_p2_score:.2f}")
PYEOF
