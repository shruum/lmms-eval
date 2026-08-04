#!/bin/bash
#SBATCH --job-name=mmhal_full
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_full_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/mmhal_full_%j.err

set -e
source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate mllm
export HF_HOME=/home/sgowda/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd /home/sgowda/workspace/SRF/lmms-eval

MODEL="llava-hf/llava-1.5-7b-hf"
OUT="results/mmhalbench_full"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "MMHal-Bench full eval — 96 samples, baseline + SRF best config"
echo "Best config: alpha=0.5 fit=0.25 naa=1.0 pth=0.27 additive le=20"
echo "Started: $(date)"
echo "======================================================"

echo ""
echo ">>> Pass 1: baseline (greedy)"
python srf/eval.py \
    --method baseline \
    --model "$MODEL" \
    --datasets mmhalbench \
    --openai_model gpt-4o \
    --output "$OUT/baseline"

echo ""
echo ">>> Pass 2: SRF best config"
python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets mmhalbench \
    --alpha 0.5 --layer_end 20 \
    --neg_absent_alpha 1.0 \
    --llava_boost_mode additive \
    --clip_fallback_thresh 0.25 \
    --clip_patch_thresh 0.27 \
    --openai_model gpt-4o \
    --output "$OUT/srf"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python - <<'PYEOF'
import json, pathlib

def load(d):
    sf = pathlib.Path(d) / "mmhalbench_scores.json"
    if not sf.exists():
        return None, None
    data = json.loads(sf.read_text())
    if "score" in data:
        return data["score"], data["hal_pct"]
    key = list(data.keys())[0]
    return data[key]["score"], data[key]["hal_pct"]

base_score, base_hal = load("results/mmhalbench_full/baseline")
srf_score,  srf_hal  = load("results/mmhalbench_full/srf")

print("\n" + "="*58)
print("MMHal-Bench Full Results (96 samples, gpt-4o scoring)")
print("="*58)
print(f"{'Method':<30}  {'Score':>6}  {'Hal%':>6}")
print("-"*46)
print(f"{'Baseline (greedy)':<30}  {base_score:>6.2f}  {base_hal:>6.1f}%")
print(f"{'SRF (α=0.5,fit=0.25,naa=1.0)':<30}  {srf_score:>6.2f}  {srf_hal:>6.1f}%")
print(f"{'Δ (SRF − Baseline)':<30}  {srf_score-base_score:>+6.2f}  {srf_hal-base_hal:>+6.1f}%")
print("="*58)
print()
print("Reference (ILVAD paper Table 1, gpt-4-0314 scoring):")
print(f"  Greedy        Score=2.01  Hal%=67.0%")
print(f"  VCD           Score=2.20  Hal%=61.8%")
print(f"  AGLA          Score=2.14  Hal%=63.8%")
print(f"  ILVAD (best)  Score=2.22  Hal%=61.5%")
PYEOF
