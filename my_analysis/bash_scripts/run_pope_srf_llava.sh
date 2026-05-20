#!/usr/bin/env bash
# =============================================================================
# run_pope_srf_llava.sh — POPE SRF evaluation, LLaVA-1.5-7B only, GPU 1
#
# Derived from run_pope_srf.sh (autoresearch-validated settings).
# Run this in parallel with run_pope_srf_qwen.sh on a separate GPU.
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="${HF_HOME:-$HOME/shruthi/hf_cache}"

CONDA_ENV="mllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/pope_srf_eval.py"
OUT_DIR="${SCRIPT_DIR}/pope_full_results"
N_SAMPLES=500
MODE="both"

mkdir -p "${OUT_DIR}"

echo "[$(date '+%H:%M:%S')] LLaVA-1.5-7B — POPE SRF (GPU ${CUDA_VISIBLE_DEVICES})"
echo "HF_HOME=${HF_HOME}"
echo "Output: ${OUT_DIR}/llava_7b.json"
echo "============================================================"

conda run -n "${CONDA_ENV}" --no-capture-output \
    python "${EVAL_SCRIPT}" \
        --arch    llava \
        --model   llava-hf/llava-1.5-7b-hf \
        --n_samples "${N_SAMPLES}" \
        --mode    "${MODE}" \
        --output  "${OUT_DIR}/llava_7b.json"

echo "[$(date '+%H:%M:%S')] Done — ${OUT_DIR}/llava_7b.json"
