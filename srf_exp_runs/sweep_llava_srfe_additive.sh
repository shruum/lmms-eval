#!/bin/bash
#SBATCH --job-name=srfe_add_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/srfe_add_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/srfe_add_sweep_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/sweep_llava/srfe_additive"
REPOPE_DIR="data/repope"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "SRF-E gamma sweep — additive patch, α=0.5, fit=0.25"
echo "500 RePOPE adversarial | layer_end=20"
echo "Started: $(date)"
echo "======================================================"

# Sweep wider than old multiplicative sweep (γ=0.3 was best there).
# Additive α=0.5 produces a smaller per-token steering signal than
# multiplicative α=2.0, so the contrastive gap may be smaller → higher γ needed.
for gamma in 0.1 0.2 0.3 0.5 0.8 1.0 1.5 2.0; do
    echo ""
    echo ">>> gamma=$gamma"
    python srf/eval.py \
        --method srfe \
        --gamma "$gamma" \
        --model "$MODEL" \
        --datasets pope --pope_splits adversarial \
        --n_pope 500 \
        --alpha 0.5 --layer_end 20 \
        --neg_absent_alpha 0.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --repope_dir "$REPOPE_DIR" \
        --output "$OUT/g${gamma}"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

root = pathlib.Path("results/sweep_llava/srfe_additive")
rows = []
for d in sorted(root.iterdir()):
    jf = d / "pope.json"
    if not jf.exists(): continue
    data = json.loads(jf.read_text())
    overall = data.get("overall", {})
    # SRF-E has one entry per gamma in the overall dict
    for gkey, g in overall.items():
        acc  = g.get("acc", 0) * 100
        prec = g.get("precision", 0) * 100
        rec  = g.get("recall", 0) * 100
        f1   = g.get("f1", 0) * 100
        rows.append((f1, d.name, acc, prec, rec, f1))

rows.sort(reverse=True)
print(f"\n{'gamma':<10}  {'acc':>6}  {'prec':>7}  {'rec':>7}  {'F1':>7}")
for _, tag, acc, prec, rec, f1 in rows:
    gamma = tag.replace("g", "")
    print(f"{gamma:<10}  {acc:6.2f}  {prec:7.2f}  {rec:7.2f}  {f1:7.2f}")
PYEOF
