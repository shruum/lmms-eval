#!/bin/bash
#SBATCH --job-name=mme_fixed
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_fixed_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_fixed_%j.err

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MME_DIR="/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
OUT="results/mme_fixed_patch"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MME — fixed patch validation (phase=both, 6 heads, lambda_bg, lambda_sys)"
echo "Previous reported: baseline=631.66, SRF=670.00 (buggy patch, all 32 heads, no eps/sys)"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> Baseline"
python -u srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --output "$OUT/baseline"

echo ""
echo ">>> SRF fixed (phase=both, alpha=0.5, eps=0.2, sys_beta=0.30, 6 heads, additive)"
python -u srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --alpha 0.5 --layer_end 20 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "$OUT/srf_fixed"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 - <<'PYEOF'
import json, pathlib

def load(d):
    p = pathlib.Path(d) / "mme.json"
    if not p.exists():
        return None, None
    data = json.loads(p.read_text())
    key = list(data["method_mme_score"].keys())[0]
    return data["method_mme_score"][key], data.get("cat_stats", {})

def subtask_acc(cats, task):
    if task not in cats:
        return float("nan")
    c = cats[task]
    n_yes = c.get("n_yes") or 1
    n_no  = c.get("n_no")  or 1
    g = list(c["srf_yes"].keys())[0]
    return (c["srf_yes"][g] / n_yes + c["srf_no"][g] / n_no) * 100

subtasks = ["existence", "count", "position", "color"]
base_total, base_cats = load("results/mme_fixed_patch/baseline")
srf_total,  srf_cats  = load("results/mme_fixed_patch/srf_fixed")

print("\n" + "="*65)
print("MME results — fixed patch (phase=both, 6 heads, lambda_bg+sys active)")
print("="*65)
print(f"{'Method':<18} {'exist':>7} {'count':>7} {'pos':>7} {'color':>7} {'TOTAL':>8}")
print("-"*60)
for label, total, cats in [("Baseline", base_total, base_cats), ("SRF fixed", srf_total, srf_cats)]:
    if cats:
        scores = [subtask_acc(cats, t) for t in subtasks]
        print(f"{label:<18} {'  '.join(f'{s:6.2f}' for s in scores)}  {total:7.2f}")
print("-"*60)
print("Previously reported (buggy):  baseline=631.66  SRF=670.00")
print("="*65)
PYEOF
