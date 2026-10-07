#!/bin/bash
# Submit 15 parallel MMHal sweep jobs (srffovea, naa=0.0 fixed).
# Groups: sigma(6) | alpha(3) | k/heads(3) | layer_end(2) | phase(1)
#
# Usage: bash srf_exp_runs/mmhal_sweep_submit.sh

set -e
cd /home/sgowda/workspace/SRF/lmms-eval
mkdir -p srf_exp_runs/logs

WORKER="srf_exp_runs/mmhal_sweep_worker.sh"
JOBS=()

submit() {
    local TAG=$1 SIGMA=$2 ALPHA=$3 K=$4 LE=$5 PHASE=$6
    local JID
    JID=$(sbatch \
        --job-name="mhs_${TAG}" \
        --output="srf_exp_runs/logs/mmhal_sw_${TAG}_%j.out" \
        --error="srf_exp_runs/logs/mmhal_sw_${TAG}_%j.err" \
        --export=ALL,SRF_TAG=${TAG},SRF_SIGMA=${SIGMA},SRF_ALPHA=${ALPHA},SRF_K=${K},SRF_LE=${LE},SRF_PHASE=${PHASE} \
        "$WORKER" | awk '{print $NF}')
    JOBS+=("$JID:$TAG")
    echo "  Submitted $TAG → job $JID"
}

echo "======================================================"
echo "MMHal sweep — 15 jobs (srffovea, naa=0.0)"
echo "Baseline ref: score=2.28  hal=60.4%  (n=96)"
echo "======================================================"

echo ""
echo "Group A — sigma sweep (alpha=0.5, k=0.20, le=20, phase=both)"
submit sig5    5   0.5  0.20  20  both
submit sig10   10  0.5  0.20  20  both
submit sig15   15  0.5  0.20  20  both
submit sig20   20  0.5  0.20  20  both   # reference
submit sig25   25  0.5  0.20  20  both
submit sig30   30  0.5  0.20  20  both

echo ""
echo "Group B — alpha sweep (sigma=20, k=0.20, le=20, phase=both)"
submit a0p3    20  0.3  0.20  20  both
submit a1p0    20  1.0  0.20  20  both
submit a2p0    20  2.0  0.20  20  both

echo ""
echo "Group C — head k sweep (sigma=20, alpha=0.5, le=20, phase=both)"
submit k10     20  0.5  0.10  20  both
submit k50     20  0.5  0.50  20  both
submit k100    20  0.5  1.00  20  both   # all heads (= pre-fix behaviour)

echo ""
echo "Group D — layer_end sweep (sigma=20, alpha=0.5, k=0.20, phase=both)"
submit le16    20  0.5  0.20  16  both
submit le24    20  0.5  0.20  24  both

echo ""
echo "Group E — phase comparison (sigma=20, alpha=0.5, k=0.20, le=20)"
submit ph_gen  20  0.5  0.20  20  generation

echo ""
echo "======================================================"
echo "All 15 jobs submitted: ${JOBS[*]}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  python3 srf_exp_runs/mmhal_sweep_results.py"
echo "======================================================"
