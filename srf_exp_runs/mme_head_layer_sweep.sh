#!/bin/bash
#SBATCH --job-name=mme_hl_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_hl_sweep_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_hl_sweep_%j.err

set -eo pipefail
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
MME_DIR="/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
OUT="results/mme_hl_sweep"

# Fixed params (best from prior runs, no neg_absent_alpha)
FIXED="--method srffovea --model $MODEL --datasets mme \
    --mme_data_dir $MME_DIR --mme_subtasks existence count position color \
    --alpha 0.5 --fovea_sigma 20 --layer_start 8 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MME head×layer sweep — srffovea, alpha=0.5, sigma=20"
echo "Grid: layer_end in {20,24,32} x head_top_k_pct in {0.20,0.30}"
echo "Started: $(date)"
echo "======================================================"

# Config A: baseline (current defaults)
echo ""; echo ">>> [A] le=20  htk=0.20  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 20 --head_top_k_pct 0.20 \
    --output "$OUT/A_le20_htk020"

# Config B: more heads, same range
echo ""; echo ">>> [B] le=20  htk=0.30  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 20 --head_top_k_pct 0.30 \
    --output "$OUT/B_le20_htk030"

# Config C: wider range, moderate heads
echo ""; echo ">>> [C] le=24  htk=0.20  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 24 --head_top_k_pct 0.20 \
    --output "$OUT/C_le24_htk020"

# Config D: wider range + more heads
echo ""; echo ">>> [D] le=24  htk=0.30  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 24 --head_top_k_pct 0.30 \
    --output "$OUT/D_le24_htk030"

# Config E: full range (includes L31 spike), moderate heads
echo ""; echo ">>> [E] le=32  htk=0.25  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 32 --head_top_k_pct 0.25 \
    --output "$OUT/E_le32_htk025"

# Config F: full range + more heads
echo ""; echo ">>> [F] le=32  htk=0.35  $(date)"
python -u srf/eval.py $FIXED \
    --layer_end 32 --head_top_k_pct 0.35 \
    --output "$OUT/F_le32_htk035"

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

base = pathlib.Path("results/mme_hl_sweep")
configs = [
    ("A_le20_htk020", "le=20 htk=0.20 (current)"),
    ("B_le20_htk030", "le=20 htk=0.30"),
    ("C_le24_htk020", "le=24 htk=0.20"),
    ("D_le24_htk030", "le=24 htk=0.30"),
    ("E_le32_htk025", "le=32 htk=0.25"),
    ("F_le32_htk035", "le=32 htk=0.35"),
]

subtasks = ["existence", "count", "position", "color"]
print("\n" + "="*72)
print("MME head×layer sweep — srffovea, alpha=0.5, sigma=20, no neg_absent")
print(f"{'Config':<24} {'exist':>6} {'count':>6} {'pos':>6} {'color':>6} {'TOTAL':>7}")
print("-"*65)
print(f"  prev best (srf method)                                    686.70  ← ref")
print(f"  baseline                                                  670.00  ← ref")
print("-"*65)

results = []
for key, label in configs:
    total, cats = load_score(base / key)
    if total is None:
        print(f"  {label:<22}  [missing]")
        continue
    scores = [subtask_score(cats, t) for t in subtasks]
    marker = " ◄ best" if total == max(r[0] for r in results + [(total, label)]) else ""
    print(f"  {label:<22}  {'  '.join(f'{s:5.1f}' for s in scores)}  {total:7.2f}{marker}")
    results.append((total, label))

if results:
    best = max(results, key=lambda x: x[0])
    print("-"*65)
    print(f"  BEST: {best[1]}  →  {best[0]:.2f}")
print("="*72)
PYEOF
