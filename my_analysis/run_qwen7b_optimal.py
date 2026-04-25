#!/usr/bin/env python3
"""
Run full POPE evaluation for Qwen-7B with optimal SRF configuration

Optimal config found from overnight experiments:
- Layers 15-21
- Boost 2.0/Suppress 5.0
- Head 25%
- Improvement: +2.67%
"""

import os
import subprocess
import sys
from pathlib import Path

# Optimal configuration from overnight experiments
OPTIMAL_CONFIG = {
    "layer_start": 15,
    "layer_end": 21,
    "boost_alpha": 2.0,
    "suppress_alpha": 5.0,
    "head_top_k_pct": 0.25,  # 25% heads (better than 20% for 7B)
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.30,
    "clip_use_soft": True,
    "saliency_method": "clip",
}

# Model config
MODEL_CONFIG = {
    "arch": "qwen",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "n_samples": 500,  # Full dataset
    "mode": "both",
}

# Output
OUTPUT_DIR = Path("pope_full_results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "qwen7b_optimal.json"

# Environment
python_path = "/home/anna2/miniconda3/envs/mllm/bin/python"
env_vars = {
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HOME": "/home/anna2/.cache/huggingface",
    "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
    "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
    "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
}

print("🚀 QWEN-7B FULL POPE EVALUATION")
print("=" * 70)
print(f"Model: {MODEL_CONFIG['model_id']}")
print(f"Samples per split: {MODEL_CONFIG['n_samples']} (full dataset)")
print(f"GPU: {env_vars['CUDA_VISIBLE_DEVICES']}")
print()
print("Optimal SRF Configuration:")
print(f"  Layers: {OPTIMAL_CONFIG['layer_start']}-{OPTIMAL_CONFIG['layer_end']}")
print(f"  Boost/Suppress: {OPTIMAL_CONFIG['boost_alpha']}/{OPTIMAL_CONFIG['suppress_alpha']}")
print(f"  Head selection: top {int(OPTIMAL_CONFIG['head_top_k_pct']*100)}%")
print(f"  Expected improvement: +2.67%")
print()
print(f"Output: {OUTPUT_FILE}")
print("=" * 70)
print()
print("⏱️  Estimated time: ~30-40 minutes")
print()

# Build experiment script
exp_script = f'''
import os, sys, json
os.environ.update({env_vars})
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")

from pope_srf_eval import main, SRF, ARCH_DEFAULTS

# Update SRF config
SRF.update({OPTIMAL_CONFIG})

# Update layer config in ARCH_DEFAULTS (layers are controlled here)
ARCH_DEFAULTS["qwen"]["layer_start"] = {OPTIMAL_CONFIG['layer_start']}
ARCH_DEFAULTS["qwen"]["layer_end"] = {OPTIMAL_CONFIG['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "{MODEL_CONFIG['arch']}",
    "--model", "{MODEL_CONFIG['model_id']}",
    "--n_samples", "{MODEL_CONFIG['n_samples']}",
    "--mode", "{MODEL_CONFIG['mode']}",
    "--output", r"{OUTPUT_FILE}",
]

print("Starting evaluation...")
main()
'''

print("🔬 Starting evaluation...")
print()

try:
    current_env = os.environ.copy()
    current_env.update(env_vars)

    result = subprocess.run(
        [python_path, "-c", exp_script],
        env=current_env,
        timeout=3600,  # 60 minutes
    )

    print()
    print("=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_FILE}")
    print()

    # Parse and display results
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            data = json.load(f)

        if "metrics" in data:
            metrics = data["metrics"]

            print("📊 RESULTS BY SPLIT:")
            print("-" * 70)

            for split_name, split_data in metrics.items():
                baseline = split_data["baseline"]["accuracy"]
                srf = split_data["srf"]["accuracy"]
                improvement = srf - baseline

                print(f"\n{split_name.upper():12s}: Baseline {baseline:.4f} → SRF {srf:.4f} ({improvement:+.4f})")

            # Compute average
            baseline_accs = [split["baseline"]["accuracy"] for split in metrics.values()]
            srf_accs = [split["srf"]["accuracy"] for split in metrics.values()]

            avg_baseline = sum(baseline_accs) / len(baseline_accs)
            avg_srf = sum(srf_accs) / len(srf_accs)
            avg_improvement = avg_srf - avg_baseline

            print()
            print("-" * 70)
            print(f"{'AVERAGE':12s}: Baseline {avg_baseline:.4f} → SRF {avg_srf:.4f} ({avg_improvement:+.4f})")
            print(f"              Improvement: {avg_improvement*100:.2f}%")
            print()

            # Compare to expected
            expected_improvement = 0.0267
            if avg_improvement >= expected_improvement:
                print(f"✅ Met or exceeded expected improvement ({expected_improvement:.4f})")
            else:
                print(f"⚠️  Below expected improvement ({expected_improvement:.4f})")
                print(f"   Difference: {avg_improvement - expected_improvement:.4f}")

except subprocess.TimeoutExpired:
    print()
    print("❌ Timed out after 60 minutes")
except Exception as e:
    print()
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
