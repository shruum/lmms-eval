#!/bin/bash
# Score all 15 saved MMHal sweep responses with GPT-4o.
# Run this once after all sweep jobs finish (or when credits are available).
# Responses are already saved at results/mmhal_sweep/<tag>/mmhalbench_responses.json
#
# Usage: bash srf_exp_runs/mmhal_sweep_score.sh

set -e
cd /home/sgowda/workspace/SRF/lmms-eval

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set. Export it first."
    exit 1
fi

TAGS="sig5 sig10 sig15 sig20 sig25 sig30 a0p3 a1p0 a2p0 k10 k50 k100 le16 le24 ph_gen"

echo "======================================================"
echo "Scoring 15 MMHal sweep runs with GPT-4o"
echo "Started: $(date)"
echo "======================================================"

for tag in $TAGS; do
    resp="results/mmhal_sweep/${tag}/mmhalbench_responses.json"
    out="results/mmhal_sweep/${tag}/mmhalbench_scores.json"
    if [ ! -f "$resp" ]; then
        echo "SKIP $tag — no responses file"
        continue
    fi
    if [ -f "$out" ]; then
        echo "SKIP $tag — already scored"
        continue
    fi
    echo ""
    echo ">>> Scoring $tag …"
    python3 srf/score_mmhalbench.py \
        --response "$resp" \
        --scores   "$out" \
        --api-key  "$OPENAI_API_KEY"
done

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "======================================================"

python3 srf_exp_runs/mmhal_sweep_results.py
