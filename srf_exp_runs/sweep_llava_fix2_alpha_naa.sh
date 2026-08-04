#!/bin/bash
#SBATCH --job-name=fix2_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/fix2_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/fix2_sweep_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/fix2_full"
REPOPE_DIR="data/repope"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Fix 2 (additive) — alpha x neg_absent_alpha x full_img_thresh sweep"
echo "500 RePOPE adversarial  |  layer_end=20  |  boost_mode=additive"
echo "Started: $(date)"
echo "======================================================"

# Phase 1: alpha x neg_absent_alpha (20 runs, clip thresholds at defaults)
# alpha: 1.0 1.5 2.0 2.5 3.0 — find best attention scale
# neg_absent_alpha: 0.0 1.0 2.0 3.0 — how much to suppress absent-object tokens
# clip thresholds fixed: full_img_thresh=0.20, patch_thresh=0.27 (defaults)

echo ""
echo "=== Phase 1: alpha x neg_absent_alpha (full_img_thresh=0.20 fixed) ==="

for alpha in 1.0 1.5 2.0 2.5 3.0; do
  for naa in 0.0 1.0 2.0 3.0; do
    tag="a${alpha}_naa${naa}"
    echo ""
    echo ">>> $tag"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets pope --pope_splits adversarial \
        --n_pope 500 \
        --alpha "$alpha" --layer_end 20 \
        --neg_absent_alpha "$naa" \
        --llava_boost_mode additive \
        --repope_dir "$REPOPE_DIR" \
        --output "$OUT/phase1/$tag"
  done
done

# Phase 2: CLIP threshold sweep
# Best alpha/naa from phase 1 not yet known → use alpha=2.0, naa=2.0 as starting point.
# full_img_thresh (--clip_fallback_thresh): controls primary full-image presence gate.
#   Lower → more objects pass gate → higher recall; higher → more conservative.
# patch_thresh (--clip_patch_thresh): controls backup patch-max gate.
#   Lower → patch gate fires more often → catches borderline absent cases.

echo ""
echo "=== Phase 2: CLIP threshold sweep (alpha=2.0, naa=2.0) ==="

for fit in 0.15 0.18 0.20 0.22 0.25; do
  for pt in 0.20 0.27 0.32; do
    tag="fit${fit}_pt${pt}"
    echo ""
    echo ">>> $tag"
    python srf/eval.py \
        --method srf \
        --model "$MODEL" \
        --datasets pope --pope_splits adversarial \
        --n_pope 500 \
        --alpha 2.0 --layer_end 20 \
        --neg_absent_alpha 2.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh "$fit" \
        --clip_patch_thresh "$pt" \
        --repope_dir "$REPOPE_DIR" \
        --output "$OUT/phase2/$tag"
  done
done

echo ""
echo "======================================================"
echo "All done: $(date)"
echo "======================================================"

# Print compact summary
echo ""
echo "=== PHASE 1 SUMMARY (acc / prec / rec / F1) ==="
python - <<'PYEOF'
import json, pathlib

out_root = pathlib.Path("results/sweep_llava/fix2_full/phase1")
if not out_root.exists():
    print("No phase1 results yet.")
else:
    rows = []
    for d in sorted(out_root.iterdir()):
        jf = d / "pope.json"
        if not jf.exists():
            continue
        data = json.loads(jf.read_text())
        overall = data.get("overall", {})
        g0 = list(overall.values())[0] if overall else {}
        tag = d.name
        acc  = g0.get("acc",  0) * 100
        prec = g0.get("prec", 0) * 100
        rec  = g0.get("rec",  0) * 100
        f1   = g0.get("f1",   0) * 100
        rows.append((f1, tag, acc, prec, rec, f1))
    rows.sort(reverse=True)
    print(f"{'tag':<20}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'F1':>6}")
    for _, tag, acc, prec, rec, f1 in rows:
        print(f"{tag:<20}  {acc:6.2f}  {prec:6.2f}  {rec:6.2f}  {f1:6.2f}")
PYEOF

echo ""
echo "=== PHASE 2 SUMMARY (acc / prec / rec / F1) ==="
python - <<'PYEOF'
import json, pathlib

out_root = pathlib.Path("results/sweep_llava/fix2_full/phase2")
if not out_root.exists():
    print("No phase2 results yet.")
else:
    rows = []
    for d in sorted(out_root.iterdir()):
        jf = d / "pope.json"
        if not jf.exists():
            continue
        data = json.loads(jf.read_text())
        overall = data.get("overall", {})
        g0 = list(overall.values())[0] if overall else {}
        tag = d.name
        acc  = g0.get("acc",  0) * 100
        prec = g0.get("prec", 0) * 100
        rec  = g0.get("rec",  0) * 100
        f1   = g0.get("f1",   0) * 100
        rows.append((f1, tag, acc, prec, rec, f1))
    rows.sort(reverse=True)
    print(f"{'tag':<22}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'F1':>6}")
    for _, tag, acc, prec, rec, f1 in rows:
        print(f"{tag:<22}  {acc:6.2f}  {prec:6.2f}  {rec:6.2f}  {f1:6.2f}")
PYEOF
