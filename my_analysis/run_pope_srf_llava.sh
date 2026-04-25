#!/usr/bin/env bash
# =============================================================================
# run_pope_srf_llava.sh — POPE SRF evaluation, LLaVA-1.5-7B only, GPU 1
#
# Derived from run_pope_srf.sh (autoresearch-validated settings).
# Run this in parallel with run_pope_srf_qwen.sh on a separate GPU.
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_EXPERIMENTAL_HTTP_STREAM=1
export XET_CACHE_DIR="${HF_HOME}/xet"
export XET_LOG_FILE="${HF_HOME}/xet.log"
export HF_HUB_CACHE="${HF_HOME}/hub"

CONDA_ENV="mllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/pope_srf_eval.py"
PYTHON_PATH="$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"
OUT_DIR="${SCRIPT_DIR}/pope_full_results"
N_SAMPLES=500
MODE="both"

mkdir -p "${OUT_DIR}"

echo "[$(date '+%H:%M:%S')] LLaVA-1.5-7B — POPE SRF (GPU ${CUDA_VISIBLE_DEVICES})"
echo "HF_HOME=${HF_HOME}"
echo "Output: ${OUT_DIR}/llava_7b.json"
echo "============================================================"

"${PYTHON_PATH}" "${EVAL_SCRIPT}" \
    --arch    llava \
    --model   llava-hf/llava-1.5-7b-hf \
    --n_samples "${N_SAMPLES}" \
    --mode    "${MODE}" \
    --output  "${OUT_DIR}/llava_7b.json"

echo "[$(date '+%H:%M:%S')] Done — ${OUT_DIR}/llava_7b.json"
