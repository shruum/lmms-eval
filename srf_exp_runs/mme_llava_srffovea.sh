#!/bin/bash
#SBATCH --job-name=mme_fovea
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_fovea_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_fovea_%j.err

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
echo "MME full — SRF-Fovea (σ=20), all 4 subtasks, LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> SRF-Fovea (σ=20, α=0.5, naa=1.0, le=20, additive)"
python srf/eval.py \
    --method srffovea \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --fovea_sigma 20 \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "$OUT/srffovea"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def load_mme(d):
    p = pathlib.Path(d) / "mme.json"
    if not p.exists():
        return None, {}
    data = json.loads(p.read_text())
    key = list(data["method_mme_score"].keys())[0]
    return data["method_mme_score"][key], data["cat_stats"]

def subtask_score(cat_stats, task):
    if task not in cat_stats:
        return float("nan")
    s = cat_stats[task]
    n_yes = s["n_yes"] or 1
    n_no  = s["n_no"]  or 1
    gk = list(s["srf_yes"].keys())[0]
    return (s["srf_yes"][gk] / n_yes + s["srf_no"][gk] / n_no) * 100

total, cats = load_mme("results/mme_llava_full/srffovea")
subtasks = ["existence", "count", "position", "color"]
scores = [subtask_score(cats, s) for s in subtasks]

print("\n" + "="*62)
print("SRF-Fovea MME (LLaVA-1.5-7B, σ=20)")
print("="*62)
print(f"{'exist':>8} {'count':>8} {'pos':>8} {'color':>8} {'TOTAL':>8}")
print("  ".join(f"{s:7.2f}" for s in scores), f"  {sum(scores):7.2f}")
print("="*62)
PYEOF
