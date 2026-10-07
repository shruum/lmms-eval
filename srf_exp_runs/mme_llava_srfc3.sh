#!/bin/bash
#SBATCH --job-name=mme_srfc3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_srfc3_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_srfc3_%j.err

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
echo "MME full — SRF-C v3 (embed-space zero, γ=0.3/1.0/3.0), LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

python srf/eval.py \
    --method srfc3 \
    --model "$MODEL" \
    --datasets mme \
    --mme_data_dir "$MME_DIR" \
    --mme_subtasks existence count position color \
    --gamma 0.3 1.0 3.0 \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --output "$OUT/srfc3"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

data = json.loads(pathlib.Path("results/mme_llava_full/srfc3/mme.json").read_text())
subtasks = ["existence", "count", "position", "color"]
gammas = sorted(data["method_mme_score"].keys(), key=float)

def subtask_score(cat_stats, task, gk):
    s = cat_stats.get(task, {})
    n_yes = s.get("n_yes") or 1; n_no = s.get("n_no") or 1
    return (s["srf_yes"][gk] / n_yes + s["srf_no"][gk] / n_no) * 100

print("\n" + "="*72)
print("SRF-C v3 (embed-space zero) MME Gamma Sweep — LLaVA-1.5-7B")
print("="*72)
print(f"{'γ':<6} {'exist':>8} {'count':>8} {'pos':>8} {'color':>8} {'TOTAL':>9}")
print("-"*54)
best_total, best_gamma = -1, None
for gk in gammas:
    scores = [subtask_score(data["cat_stats"], s, gk) for s in subtasks]
    total  = data["method_mme_score"][gk]
    marker = " ← best" if total > best_total else ""
    if total > best_total: best_total, best_gamma = total, gk
    print(f"{float(gk):<6.1f} " + "  ".join(f"{s:7.2f}" for s in scores) + f"  {total:8.2f}{marker}")
print("="*72)
print(f"\nBest: γ={best_gamma} → {best_total:.2f}")
print("\nReference: Baseline=670.00  SRF=670.00  SRF-Fovea=663.33  SRF-Cv2=653.33  SRF-Cv1=633.33")
PYEOF
