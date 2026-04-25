#!/usr/bin/env python3
"""
Sanity check for Qwen-7B SRF investigation - single experiment

This tests that the setup works before running the full investigation.
"""

import os
import subprocess
import sys
from pathlib import Path

# Test configuration - just one simple experiment
test_exp = {
    "name": "Sanity Check",
    "layer_start": 10,
    "layer_end": 16,
    "boost_alpha": 2.0,
    "suppress_alpha": 5.0,
    "head_top_k_pct": 0.20,
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.30,
    "clip_use_soft": True,
    "saliency_method": "clip",
}

print("🧪 SANITY CHECK FOR QWEN-7B INVESTIGATION")
print("=" * 60)
print(f"Testing single experiment:")
print(f"  Layers: {test_exp['layer_start']}-{test_exp['layer_end']}")
print(f"  Boost/Suppress: {test_exp['boost_alpha']}/{test_exp['suppress_alpha']}")
print(f"  GPU: 0")
print("=" * 60)

output_dir = Path("qwen7b_sanity_check")
output_dir.mkdir(exist_ok=True)

python_path = "/home/anna2/miniconda3/envs/mllm/bin/python"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
n_samples = 50  # Small sample for quick test

log_file = output_dir / "sanity_check.log"
result_file = output_dir / "result.json"

env_vars = {
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HOME": "/home/anna2/.cache/huggingface",
    "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
    "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
    "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
}

exp_script = f'''
import os, sys, json
os.environ.update({env_vars})
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")

from pope_srf_eval import main, SRF, ARCH_DEFAULTS

# Update SRF config
SRF.update({test_exp})
ARCH_DEFAULTS["qwen"]["layer_start"] = {test_exp['layer_start']}
ARCH_DEFAULTS["qwen"]["layer_end"] = {test_exp['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "qwen",
    "--model", "{model_id}",
    "--n_samples", "{n_samples}",
    "--mode", "both",
    "--output", r"{result_file}",
]

main()
'''

print("\n🔬 Running sanity check experiment...")
print("(This should take ~3-5 minutes with 50 samples)")

try:
    current_env = os.environ.copy()
    current_env.update(env_vars)

    with open(log_file, 'w') as f:
        result = subprocess.run(
            [python_path, "-c", exp_script],
            env=current_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=600,  # 10 minutes
        )

    print(f"\n✅ Experiment completed!")
    print(f"Log saved to: {log_file}")

    # Check if result file exists
    if result_file.exists():
        print(f"✅ Result file created: {result_file}")

        # Try to read results
        with open(result_file) as f:
            data = f.read()

        # Parse for accuracy
        import re
        srf_matches = re.findall(r'SRF.*?average\s+([\d.]+)', data, re.DOTALL)
        baseline_matches = re.findall(r'BASELINE.*?average\s+([\d.]+)', data, re.DOTALL)

        if srf_matches and baseline_matches:
            srf_acc = float(srf_matches[0])
            baseline_acc = float(baseline_matches[0])
            improvement = srf_acc - baseline_acc

            print(f"\n📊 RESULTS:")
            print(f"   Baseline: {baseline_acc:.4f}")
            print(f"   SRF: {srf_acc:.4f}")
            print(f"   Improvement: {improvement:+.4f}")

            if improvement > 0:
                print(f"\n✅ SANITY CHECK PASSED - SRF works for 7B!")
            else:
                print(f"\n⚠️  SANITY CHECK WARNING - SRF doesn't improve 7B with these params")
        else:
            print(f"\n⚠️  Could not parse accuracy from results")
            print(f"   Raw output length: {len(data)} chars")

        print(f"\n✅ Setup is working! Ready for full investigation.")
        print(f"   Run: python investigate_qwen7b_simple.py --gpu 0 --max_experiments 4")

    else:
        print(f"\n❌ Result file not created. Check log: {log_file}")
        print(f"   Last few lines:")
        with open(log_file) as f:
            lines = f.readlines()
            print("".join(lines[-20:]))

except subprocess.TimeoutExpired:
    print(f"\n❌ Timed out after 10 minutes")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
