#!/bin/bash
#SBATCH --job-name=mme_srfc
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_srfc_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mme_srfc_%j.err

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
echo "MME full — SRF-C (srfe γ=0.3/1.0/3.0), all 4 subtasks, LLaVA-1.5-7B"
echo "Gamma sweep: best LLaVA gamma was 0.3 on POPE; 3.0 is Qwen-tuned"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> SRF-C (α=0.5, naa=1.0, le=20, additive, γ sweep)"
python srf/eval.py \
    --method srfe \
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
    --output "$OUT/srfc"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def load_mme(d):
    p = pathlib.Path(d) / "mme.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

data = load_mme("results/mme_llava_full/srfc")
if data is None:
    print("No results found at results/mme_llava_full/srfc/mme.json")
else:
    subtasks = ["existence", "count", "position", "color"]
    gammas = sorted(data["method_mme_score"].keys(), key=float)

    def subtask_score(cat_stats, task, gk):
        if task not in cat_stats:
            return float("nan")
        s = cat_stats[task]
        n_yes = s["n_yes"] or 1
        n_no  = s["n_no"]  or 1
        return (s["srf_yes"][gk] / n_yes + s["srf_no"][gk] / n_no) * 100

    print("\n" + "="*72)
    print("SRF-C (SRF-E) MME Gamma Sweep — LLaVA-1.5-7B")
    print("="*72)
    print(f"{'γ':<6} {'exist':>8} {'count':>8} {'pos':>8} {'color':>8} {'TOTAL':>9}")
    print("-"*54)
    best_total, best_gamma = -1, None
    for gk in gammas:
        scores = [subtask_score(data["cat_stats"], s, gk) for s in subtasks]
        total  = data["method_mme_score"][gk]
        marker = " ← best" if total > best_total else ""
        if total > best_total:
            best_total, best_gamma = total, gk
        print(f"{float(gk):<6.1f} " + "  ".join(f"{s:7.2f}" for s in scores) + f"  {total:8.2f}{marker}")
    print("="*72)
    print(f"\nBest: γ={best_gamma} → total={best_total:.2f}")
    print()
    print("ILVAD Table 2 reference (LLaVA-1.5-7B):")
    print("  Baseline: 195.00 + 130.00 + 143.33 + 173.33 = 641.66")
    print("  ILVAD:    198.33 + 143.33 + 151.67 + 193.33 = 686.67")
PYEOF
