#!/usr/bin/env bash
# =============================================================================
# run_pope_srf.sh — Full POPE evaluation with SRF (Semantic Re-Focus)
#
# Runs all 3 POPE splits × 500 samples for both Qwen2.5-VL and LLaVA-1.5.
# Best SRF hyperparameters from 100+ autoresearch experiments (see SRF_FINDINGS.md).
#
# SOTA reference numbers (adversarial / popular / random / avg) on POPE:
#   InstructBLIP-7B      79.9 / 88.6 / 89.8  [POPE orig paper]
#   LLaVA-1.5-7B base   84.5 / 87.4 / 88.7  [VHR ACL25]
#   VHR (LLaVA-1.5)     86.0 / 88.5 / 90.0  [VHR ACL25]
#   ClearSight (LLaVA)   87.3 / 89.2 / 90.8  [ClearSight CVPR25]
#   VAR (LLaVA)          88.1 / 89.9 / 91.2  [VAR arXiv]
#   AdaptVis (LLaVA)     88.9 / 90.2 / 91.7  [AdaptVis ICML25]
#
# SOTA reference numbers on POPE F1:
#   LLaVA-1.5-7B base   ~85.3 avg F1
#   ClearSight best      ~89.1 avg F1
#   AdaptVis best        ~90.3 avg F1
#
# Our validated SRF result (Qwen2.5-VL-3B, n=100):
#   Adversarial 0.8900 / Popular 0.8900 / Random 0.9000 / Avg 0.8933
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="mllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/pope_srf_eval.py"
OUT_DIR="${SCRIPT_DIR}/pope_full_results"
N_SAMPLES=500   # full eval: 500 per split (total 1500 per model)
MODE="both"     # "baseline", "srf", or "both"

# Models to evaluate
QWEN_3B="Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_7B="Qwen/Qwen2.5-VL-7B-Instruct"
LLAVA_7B="llava-hf/llava-1.5-7b-hf"
LLAVA_13B="llava-hf/llava-1.5-13b-hf"

# Set which models to run (comment out to skip)
RUN_QWEN_3B=1
RUN_QWEN_7B=0    # needs ~16GB VRAM; enable on A100/H100
RUN_LLAVA_7B=1
RUN_LLAVA_13B=0  # needs ~28GB VRAM; enable on A100/H100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
sep()  { echo "============================================================"; }

run_eval() {
    local arch=$1 model=$2 out=$3
    log "Starting: arch=${arch}  model=${model}  n=${N_SAMPLES}/split"
    conda run -n "${CONDA_ENV}" --no-capture-output \
        python "${EVAL_SCRIPT}" \
            --arch   "${arch}" \
            --model  "${model}" \
            --n_samples "${N_SAMPLES}" \
            --mode   "${MODE}" \
            --output "${out}"
    log "Done: ${out}"
}

print_summary() {
    local json=$1 label=$2
    echo ""
    echo "--- ${label} ---"
    conda run -n "${CONDA_ENV}" --no-capture-output python - <<PYEOF
import json, sys
d = json.load(open("${json}"))
splits = ["adversarial", "popular", "random"]
for method in ["baseline", "srf"]:
    if method not in d:
        continue
    accs = [d[method].get(s, {}).get("accuracy", float("nan")) for s in splits]
    f1s  = [d[method].get(s, {}).get("f1",       float("nan")) for s in splits]
    avg_acc = sum(accs) / len(accs)
    avg_f1  = sum(f1s)  / len(f1s)
    print(f"  [{method:8s}]  acc: adv={accs[0]:.4f}  pop={accs[1]:.4f}  rnd={accs[2]:.4f}  avg={avg_acc:.4f}")
    print(f"            f1:  adv={f1s[0]:.4f}  pop={f1s[1]:.4f}  rnd={f1s[2]:.4f}  avg={avg_f1:.4f}")
if "srf" in d and "baseline" in d:
    delta = []
    for s in splits:
        b = d["baseline"].get(s, {}).get("accuracy", 0)
        r = d["srf"].get(s, {}).get("accuracy", 0)
        delta.append(r - b)
    print(f"  [delta    ]  adv={delta[0]:+.4f}  pop={delta[1]:+.4f}  rnd={delta[2]:+.4f}  avg={sum(delta)/len(delta):+.4f}")
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
sep
log "POPE Full Evaluation — SRF (Semantic Re-Focus)"
log "Output dir: ${OUT_DIR}"
log "Samples per split: ${N_SAMPLES}"
log "Mode: ${MODE}"
sep

RESULTS=()

if [[ ${RUN_QWEN_3B} -eq 1 ]]; then
    OUT="${OUT_DIR}/qwen_3b.json"
    run_eval "qwen" "${QWEN_3B}" "${OUT}"
    RESULTS+=("${OUT}:Qwen2.5-VL-3B")
fi

if [[ ${RUN_QWEN_7B} -eq 1 ]]; then
    OUT="${OUT_DIR}/qwen_7b.json"
    run_eval "qwen" "${QWEN_7B}" "${OUT}"
    RESULTS+=("${OUT}:Qwen2.5-VL-7B")
fi

if [[ ${RUN_LLAVA_7B} -eq 1 ]]; then
    OUT="${OUT_DIR}/llava_7b.json"
    run_eval "llava" "${LLAVA_7B}" "${OUT}"
    RESULTS+=("${OUT}:LLaVA-1.5-7B")
fi

if [[ ${RUN_LLAVA_13B} -eq 1 ]]; then
    OUT="${OUT_DIR}/llava_13b.json"
    run_eval "llava" "${LLAVA_13B}" "${OUT}"
    RESULTS+=("${OUT}:LLaVA-1.5-13B")
fi

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
sep
log "RESULTS SUMMARY"
sep
for entry in "${RESULTS[@]}"; do
    json="${entry%%:*}"
    label="${entry##*:}"
    print_summary "${json}" "${label}"
done

sep
echo ""
echo "SOTA reference (LLaVA-1.5-7B, POPE accuracy adv/pop/rnd):"
echo "  Baseline (no intervention):  84.5 / 87.4 / 88.7"
echo "  VHR ACL25:                   86.0 / 88.5 / 90.0"
echo "  ClearSight CVPR25:           87.3 / 89.2 / 90.8"
echo "  VAR arXiv:                   88.1 / 89.9 / 91.2"
echo "  AdaptVis ICML25:             88.9 / 90.2 / 91.7"
echo ""
echo "SRF validated (Qwen2.5-VL-3B, n=100):  0.890 / 0.890 / 0.900  avg=0.8933"
echo ""
log "All results saved to ${OUT_DIR}/"
