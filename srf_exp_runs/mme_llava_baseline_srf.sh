#!/bin/bash
#SBATCH --job-name=mme_bl_srf
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_bl_srf_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_bl_srf_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MME_DIR="/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
OUT="results/mme_llava_full"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MME full — Baseline + SRF, all 4 subtasks, LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> Pass 1: baseline"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --output "$OUT/baseline"

echo ""
echo ">>> Pass 2: SRF (α=0.5, naa=1.0, le=20, additive)"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "$OUT/srf"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def mme_score(d):
    p = pathlib.Path(d) / "mme.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    key = list(data["method_mme_score"].keys())[0]
    return data["method_mme_score"][key], data["cat_stats"]

def subtask_score(cat_stats, key):
    n_yes = cat_stats[key]["n_yes"] or 1
    n_no  = cat_stats[key]["n_no"]  or 1
    gamma_key = list(cat_stats[key]["srf_yes"].keys())[0]
    return (cat_stats[key]["srf_yes"][gamma_key] / n_yes +
            cat_stats[key]["srf_no"][gamma_key]  / n_no) * 100

base_total, base_cats = mme_score("results/mme_llava_full/baseline")
srf_total,  srf_cats  = mme_score("results/mme_llava_full/srf")

subtasks = ["existence", "count", "position", "color"]
print("\n" + "="*62)
print("MME Full Results (LLaVA-1.5-7B, 4 subtasks × 60 Qs)")
print("="*62)
print(f"{'Method':<12} {'exist':>7} {'count':>7} {'pos':>7} {'color':>7} {'TOTAL':>8}")
print("-"*52)
for label, cats in [("Baseline", base_cats), ("SRF", srf_cats)]:
    scores = [subtask_score(cats, s) for s in subtasks if s in cats]
    total  = sum(scores)
    row    = "  ".join(f"{s:6.2f}" for s in scores)
    print(f"{label:<12} {row}  {total:7.2f}")
print("-"*52)
exist_d = subtask_score(srf_cats,"existence") - subtask_score(base_cats,"existence") if "existence" in srf_cats else 0
count_d = subtask_score(srf_cats,"count")     - subtask_score(base_cats,"count")     if "count" in srf_cats else 0
pos_d   = subtask_score(srf_cats,"position")  - subtask_score(base_cats,"position")  if "position" in srf_cats else 0
color_d = subtask_score(srf_cats,"color")     - subtask_score(base_cats,"color")     if "color" in srf_cats else 0
total_d = (srf_total or 0) - (base_total or 0)
print(f"{'Δ SRF-Base':<12} {exist_d:+6.2f}  {count_d:+6.2f}  {pos_d:+6.2f}  {color_d:+6.2f}  {total_d:+7.2f}")
print("="*62)
print()
print("ILVAD Table 2 reference (LLaVA-1.5-7B):")
print("  Baseline: exist=195.00  count=130.00  pos=143.33  color=173.33  total=641.66")
print("  ILVAD:    exist=198.33  count=143.33  pos=151.67  color=193.33  total=686.67")
PYEOF
