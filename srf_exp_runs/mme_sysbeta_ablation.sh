#!/bin/bash
#SBATCH --job-name=mme_sysbeta
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_sysbeta_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_sysbeta_%j.err
#SBATCH --exclude=gcn58

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MME_DIR="/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
OUT="results/mme_sysbeta_ablation"

# Best params fixed, only sys_beta varies
FIXED="--method srffovea --model $MODEL --datasets mme \
    --mme_data_dir $MME_DIR --mme_subtasks existence count position color \
    --alpha 0.5 --fovea_sigma 20 --layer_start 8 --layer_end 20 \
    --head_top_k_pct 0.20 --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MME sys_beta ablation — lambda_bg=0 vs lambda_bg=0.30"
echo "Started: $(date)"
echo "======================================================"

# Config A: lambda_bg = 0 (disabled)
echo ""; echo ">>> [A] sys_beta=0.00 (lambda_bg disabled)  $(date)"
python -u srf/eval.py $FIXED \
    --sys_beta 0.0 \
    --output "$OUT/A_beta0"

# Config B: lambda_bg = 0.30 (current default)
echo ""; echo ">>> [B] sys_beta=0.30 (current default)  $(date)"
python -u srf/eval.py $FIXED \
    --sys_beta 0.30 \
    --output "$OUT/B_beta030"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 - <<'PYEOF'
import json, pathlib

def load_score(d):
    p = pathlib.Path(d) / "mme.json"
    if not p.exists():
        return None, {}
    data = json.loads(p.read_text())
    score = list(data["method_mme_score"].values())[0]
    return score, data.get("cat_stats", {})

def subtask_score(cats, task):
    if task not in cats:
        return float("nan")
    c = cats[task]
    n_yes = c.get("n_yes") or 1
    n_no  = c.get("n_no")  or 1
    g = list(c["srf_yes"].keys())[0]
    return (c["srf_yes"][g] / n_yes + c["srf_no"][g] / n_no) * 100

base = pathlib.Path("results/mme_sysbeta_ablation")
configs = [
    ("A_beta0",   "sys_beta=0.00 (disabled)"),
    ("B_beta030", "sys_beta=0.30 (default) "),
]

subtasks = ["existence", "count", "position", "color"]
print("\n" + "="*70)
print("MME sys_beta (lambda_bg) ablation")
print(f"{'Config':<26} {'exist':>6} {'count':>6} {'pos':>6} {'color':>6} {'TOTAL':>7}")
print("-"*65)
print(f"  baseline (no SRF)                                         670.00  ← ref")
print("-"*65)

results = []
for key, label in configs:
    total, cats = load_score(base / key)
    if total is None:
        print(f"  {label:<24}  [missing]")
        continue
    scores = [subtask_score(cats, t) for t in subtasks]
    print(f"  {label:<24}  {'  '.join(f'{s:5.1f}' for s in scores)}  {total:7.2f}")
    results.append((total, label))

if results:
    best = max(results, key=lambda x: x[0])
    print("-"*65)
    print(f"  BEST: {best[1]}  →  {best[0]:.2f}")
print("="*70)
PYEOF
