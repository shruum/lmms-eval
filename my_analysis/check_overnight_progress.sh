#!/usr/bin/env bash
# =============================================================================
# Check overnight experiment progress
# =============================================================================

echo "🌙 OVERNIGHT EXPERIMENT STATUS"
echo "================================"
date
echo ""

# Check GPU 0 (Qwen-7B)
echo "📊 GPU 0: Qwen-7B (late focus)"
if [ -f "overnight_late_Qwen2.5-VL-7B-Instruct/results_partial.json" ]; then
    python -c "
import json, sys
d = json.load(open('overnight_late_Qwen2.5-VL-7B-Instruct/results_partial.json'))
print(f\"  Progress: {d['progress']}\")
print(f\"  Best improvement: {d['best_improvement']:+.4f}\")
if d['best_config']:
    print(f\"  Best config: Layers {d['best_config']['layer_start']}-{d['best_config']['layer_end']}\")
print(f\"  Experiments completed: {len(d['results'])}\")
successful = sum(1 for r in d['results'] if r.get('status') == 'success')
failed = sum(1 for r in d['results'] if r.get('status') in ['failed', 'error', 'timeout'])
print(f\"  Success: {successful}, Failed: {failed}\")
"
else
    echo "  No results yet"
fi
echo ""

# Check GPU 1 (Qwen-3B)
echo "📊 GPU 1: Qwen-3B (mid focus)"
if [ -f "overnight_mid_Qwen2.5-VL-3B-Instruct/results_partial.json" ]; then
    python -c "
import json, sys
d = json.load(open('overnight_mid_Qwen2.5-VL-3B-Instruct/results_partial.json'))
print(f\"  Progress: {d['progress']}\")
print(f\"  Best improvement: {d['best_improvement']:+.4f}\")
if d['best_config']:
    print(f\"  Best config: Layers {d['best_config']['layer_start']}-{d['best_config']['layer_end']}\")
print(f\"  Experiments completed: {len(d['results'])}\")
successful = sum(1 for r in d['results'] if r.get('status') == 'success')
failed = sum(1 for r in d['results'] if r.get('status') in ['failed', 'error', 'timeout'])
print(f\"  Success: {successful}, Failed: {failed}\")
"
else
    echo "  No results yet"
fi
echo ""

# Check processes
echo "🔄 Running processes:"
ps aux | grep overnight_experiments | grep -v grep | wc -l | xargs echo "  Processes running:"
echo ""

echo "💡 Monitor logs with:"
echo "  tail -f overnight_gpu0.log"
echo "  tail -f overnight_gpu1.log"
echo ""
echo "================================"
