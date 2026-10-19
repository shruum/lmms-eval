#!/bin/bash
#SBATCH --job-name=repro
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repro_%x_%j.out
#SBATCH --error=/home/sgowda/workspace/SRF/lmms-eval/srf_exp_runs/logs/repro_%x_%j.err
#SBATCH --exclude=gcn58
#
# Reproduction check: run the reference configs of one model against a given
# code checkout, so a reference branch and srf-final can be compared on
# identical commands.
#
#   sbatch -J qwen_final  srf_exp_runs/repro_check.sh <code_dir> qwen  <tag>
#   sbatch -J llava_final srf_exp_runs/repro_check.sh <code_dir> llava <tag>
#   conda env: --export=ALL,ENV=srf (default mllm); MMHal GPT scoring: SCORE=1
#   short suites qwen_core (MMVP base+SRF) / llava_core (MME base+SRF); DTYPE=float32 etc.
#
# Expected (reference numbers):
#   qwen : MMVP baseline 40.00 pair | MMVP SRF (ratio_topk, k=0.2, L6-31) 45.33 pair
#          POPE n=500 — compared per-sample between checkouts (--save_records)
#   llava: MME baseline 670.0 | MME SRFovea a=0.5 693.33 | MMHal SRFovea a=0.3 2.458

set -eo pipefail
CODE="${1:?code dir}"; SUITE="${2:?qwen|llava}"; TAG="${3:?tag}"

source /home/sgowda/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV:-mllm}"
export HF_HOME=/home/sgowda/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="${OPENAI_API_KEY}"

OUT=/home/sgowda/workspace/SRF/lmms-eval/results/repro/$TAG
# --dtype only exists on srf-final; leave DTYPE unset for the reference checkouts.
DT=""; [ -n "${DTYPE:-}" ] && DT="--dtype $DTYPE"
mkdir -p "$OUT"
cd "$CODE"

echo "======================================================"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Suite: $SUITE  Tag: $TAG"
echo "Code: $CODE @ $(git rev-parse --short HEAD)  Env: ${ENV:-mllm}  Dtype: ${DTYPE:-bfloat16}"
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
echo "Started: $(date)"
echo "======================================================"

if [ "$SUITE" = "qwen" ]; then
    MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
    SRF="--method srffovea --model $MODEL --head_mode ratio_topk --head_top_k_pct 0.2 --layer_start 6 --layer_end 31"

    echo ">>> [Q1] MMVP baseline  $(date)"
    python -u srf/eval.py --method baseline --model $MODEL --datasets mmvp \
        --save_records 1 --output "$OUT/mmvp_baseline"

    echo ">>> [Q2] MMVP SRF  $(date)"
    python -u srf/eval.py $SRF --datasets mmvp \
        --save_records 1 --output "$OUT/mmvp_srf"

    echo ">>> [Q3] POPE SRF n=500  $(date)"
    python -u srf/eval.py $SRF --datasets pope --n_pope 500 \
        --save_records "$OUT/pope_srf/records.json" --output "$OUT/pope_srf"

elif [ "$SUITE" = "qwen_core" ]; then
    MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
    SRF="--method srffovea --model $MODEL --head_mode ratio_topk --head_top_k_pct 0.2 --layer_start 6 --layer_end 31"
    echo ">>> [Q1] MMVP baseline  $(date)"
    python -u srf/eval.py --method baseline --model $MODEL $DT --datasets mmvp --save_records 1 --output "$OUT/mmvp_baseline"
    echo ">>> [Q2] MMVP SRF  $(date)"
    python -u srf/eval.py $SRF $DT --datasets mmvp --save_records 1 --output "$OUT/mmvp_srf"

elif [ "$SUITE" = "llava_core" ]; then
    MODEL="llava-hf/llava-1.5-7b-hf"
    MME="--datasets mme --mme_data_dir /home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark --mme_subtasks existence count position color"
    COMMON="--method srffovea --model $MODEL --fovea_sigma 20 --layer_start 8 --layer_end 20 --head_top_k_pct 0.20 --llava_boost_mode additive --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27"
    echo ">>> [L1] MME baseline  $(date)"
    python -u srf/eval.py --method baseline --model $MODEL $DT $MME --output "$OUT/mme_baseline"
    echo ">>> [L2] MME SRFovea alpha=0.5  $(date)"
    python -u srf/eval.py $COMMON $DT --alpha 0.5 --sys_beta 0.30 $MME --output "$OUT/mme_srf"

elif [ "$SUITE" = "llava" ]; then
    MODEL="llava-hf/llava-1.5-7b-hf"
    MME="--datasets mme --mme_data_dir /home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark --mme_subtasks existence count position color"
    COMMON="--method srffovea --model $MODEL --fovea_sigma 20 --layer_start 8 --layer_end 20 --head_top_k_pct 0.20 --llava_boost_mode additive --clip_fallback_thresh 0.25 --clip_patch_thresh 0.27"

    echo ">>> [L1] MME baseline  $(date)"
    python -u srf/eval.py --method baseline --model $MODEL $MME --output "$OUT/mme_baseline"

    echo ">>> [L2] MME SRFovea alpha=0.5  $(date)"
    python -u srf/eval.py $COMMON --alpha 0.5 --sys_beta 0.30 $MME --output "$OUT/mme_srf"

    echo ">>> [L3] MMHal SRFovea alpha=0.3  $(date)"
    # Responses are compared across checkouts; GPT scoring only when SCORE=1.
    if [ "${SCORE:-0}" != "1" ]; then unset OPENAI_API_KEY; fi
    python -u srf/eval.py $COMMON --alpha 0.3 --datasets mmhalbench \
        --mmhalbench_json /home/sgowda/workspace/ILVAD/data/MMHal-Bench/response_template.json \
        --openai_model gpt-4o --output "$OUT/mmhal_srf"
else
    echo "unknown suite $SUITE"; exit 1
fi

echo "======================================================"
echo "Done: $(date)"
echo "======================================================"
