#!/bin/bash
# Full evaluation with best layer-specific config

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT="results/vlmbias_layer_specific_full/"

echo "=========================================="
echo "🚀 FULL VLM BIAS EVALUATION"
echo "=========================================="
echo "Config: late_16-27 + textBeta=0.6"
echo "Dataset: All 2784 samples"
echo "Expected: >21% (Δ=+2-3%)"
echo "=========================================="

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 conda run -n mllm --no-capture-output python srf/eval.py \
    --method srf \
    --model "$MODEL" \
    --datasets vlmbias \
    --output "$OUTPUT" \
    --alpha 8.0 \
    --layer_start 16 \
    --layer_end 27 \
    --text_beta 0.6 \
    --phase generation

echo ""
echo "=========================================="
echo "✅ EVALUATION COMPLETE"
echo "=========================================="

# Show results
if [ -f "${OUTPUT}vlmbias.json" ]; then
    python3 << 'EOF'
import json

with open('results/vlmbias_layer_specific_full/vlmbias.json') as f:
    data = json.load(f)

baseline = data.get('baseline', 0) * 100
srf = list(data.get('method', {}).values())[0] * 100
delta = srf - baseline

print(f"Baseline: {baseline:.2f}%")
print(f"SRF:      {srf:.2f}%")
print(f"Delta:    {delta:+.2f}%")

if delta >= 2.0:
    print("\n✅ TARGET ACHIEVED!")
elif delta >= 1.0:
    print("\n⚠️  Promising (1-2%)")
else:
    print("\n❌ Target not achieved")

EOF
fi

echo ""
echo "=========================================="
echo "📁 Results: $OUTPUT"
echo "=========================================="
