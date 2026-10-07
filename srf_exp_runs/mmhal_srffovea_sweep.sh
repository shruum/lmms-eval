#!/bin/bash
#SBATCH --job-name=mmhal_fovea
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fovea_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_fovea_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_srffovea_sweep"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench SRF-Fovea σ sweep — n=20, LLaVA-1.5-7B"
echo "Started: $(date)"
echo "======================================================"

for sigma in 10 20 30 50; do
    echo ""
    echo ">>> sigma=${sigma}  $(date)"
    python srf/eval.py \
        --method srffovea \
        --fovea_sigma ${sigma} \
        --model "$MODEL" \
        --datasets mmhalbench \
        --mmhalbench_n 20 \
        --alpha 0.5 --layer_end 20 \
        --neg_absent_alpha 1.0 \
        --llava_boost_mode additive \
        --clip_fallback_thresh 0.25 \
        --clip_patch_thresh 0.27 \
        --openai_model gpt-4o \
        --output "$OUT/s${sigma}"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

base = pathlib.Path("results/mmhalbench_srffovea_sweep")

print("\n" + "="*56)
print("SRF-Fovea σ sweep  (n=20, gpt-4o)  LLaVA-1.5-7B")
print(f"{'Method':<22}  {'Score':>6}  {'Hal%':>6}")
print("-"*40)
print(f"{'SRF (n=96, full)':<22}  {'2.29':>6}  {'59.4%':>6}  ← ref")
print(f"{'Baseline (n=96)':<22}  {'2.21':>6}  {'60.4%':>6}  ← ref")
print("-"*40)

for sigma in [10, 20, 30, 50]:
    sf = base / f"s{sigma}" / "mmhalbench_scores.json"
    if not sf.exists():
        print(f"  srffovea σ={sigma:<3}          [no scores file]")
        continue
    d = json.loads(sf.read_text())
    if "score" not in d:
        d = list(d.values())[0]
    print(f"  srffovea σ={sigma:<3}          {d['score']:>6.2f}  {d['hal_pct']:>5.1f}%")

print("="*56)
PYEOF
