#!/usr/bin/env python3
"""
Simple Qwen-7B Investigation - Using working pope_srf_eval.py

Tests why Qwen-3B parameters don't work for Qwen-7B.

Usage:
    python investigate_qwen7b_simple.py --gpu 0 --max_experiments 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

# =============================================================================
# FOCUSED INVESTIGATION
# =============================================================================

# Model config
MODEL_CONFIG = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "n_samples": 100,  # Faster than 500, but still meaningful
    "mode": "both",
    "splits": ["adversarial", "popular"],
}

# Investigation experiments
INVESTIGATION_EXPERIMENTS = [
    {
        "name": "Layers 6-12 (earlier than 3B)",
        "layer_start": 6,
        "layer_end": 12,
        "boost_alpha": 2.0,
        "suppress_alpha": 5.0,
        "head_top_k_pct": 0.20,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
        "clip_use_soft": True,
        "saliency_method": "clip",
    },
    {
        "name": "Layers 10-16 (same as 3B baseline)",
        "layer_start": 10,
        "layer_end": 16,
        "boost_alpha": 2.0,
        "suppress_alpha": 5.0,
        "head_top_k_pct": 0.20,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
        "clip_use_soft": True,
        "saliency_method": "clip",
    },
    {
        "name": "Layers 14-20 (later, more reasoning)",
        "layer_start": 14,
        "layer_end": 20,
        "boost_alpha": 2.0,
        "suppress_alpha": 5.0,
        "head_top_k_pct": 0.20,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
        "clip_use_soft": True,
        "saliency_method": "clip",
    },
    {
        "name": "Gentler intervention (1.0/3.0)",
        "layer_start": 8,
        "layer_end": 14,
        "boost_alpha": 1.0,
        "suppress_alpha": 3.0,
        "head_top_k_pct": 0.20,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
        "clip_use_soft": True,
        "saliency_method": "clip",
    },
]

# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(exp: Dict, output_dir: Path, gpu_id: int):
    """Run investigation experiment."""

    python_path = "/home/anna2/miniconda3/envs/mllm/bin/python"

    print(f"\n🔬 {exp['name']}")
    print(f"   Layers {exp['layer_start']}-{exp['layer_end']}")
    print(f"   Boost {exp['boost_alpha']}/Suppress {exp['suppress_alpha']}")

    log_file = output_dir / f"exp_{exp['name'].replace(' ', '_').replace('/', '_')}.log"
    result_file = output_dir / f"result_{exp['name'].replace(' ', '_').replace('/', '_')}.json"

    # Environment
    env_vars = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "HF_HOME": "/home/anna2/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
        "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
    }

    # Build experiment script
    exp_script = f'''
import os, sys, json
os.environ.update({env_vars})
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")

from pope_srf_eval import main, SRF, ARCH_DEFAULTS

# Update SRF config
SRF.update({exp})
ARCH_DEFAULTS["qwen"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["qwen"]["layer_end"] = {exp['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "qwen",
    "--model", "{MODEL_CONFIG['model_id']}",
    "--n_samples", "{MODEL_CONFIG['n_samples']}",
    "--mode", "{MODEL_CONFIG['mode']}",
    "--output", r"{result_file}",
]

main()
'''

    try:
        current_env = os.environ.copy()
        current_env.update(env_vars)

        with open(log_file, 'w') as f:
            result = subprocess.run(
                [python_path, "-c", exp_script],
                env=current_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=900,  # 15 minutes
            )

        # Parse results
        if result_file.exists():
            with open(result_file) as f:
                data = f.read()

            # Try parsing JSON format first
            try:
                results = json.loads(data)
                if "metrics" in results:
                    # Parse from JSON structure
                    metrics = results["metrics"]

                    # Compute average across splits
                    baseline_accs = []
                    srf_accs = []

                    for split_name, split_data in metrics.items():
                        if "baseline" in split_data and "srf" in split_data:
                            baseline_accs.append(split_data["baseline"]["accuracy"])
                            srf_accs.append(split_data["srf"]["accuracy"])

                    if baseline_accs and srf_accs:
                        baseline_acc = sum(baseline_accs) / len(baseline_accs)
                        srf_acc = sum(srf_accs) / len(srf_accs)
                        improvement = srf_acc - baseline_acc

                        exp["status"] = "success"
                        exp["srf_acc"] = srf_acc
                        exp["baseline_acc"] = baseline_acc
                        exp["improvement"] = improvement
                        exp["detailed_metrics"] = metrics

                        print(f"   ✓ Baseline: {baseline_acc:.4f} → SRF: {srf_acc:.4f} ({improvement:+.4f})")
                        return exp
            except json.JSONDecodeError:
                pass

            # Fallback: try parsing from log text format
            import re
            srf_matches = re.findall(r'SRF.*?average\s+([\d.]+)', data, re.DOTALL)
            baseline_matches = re.findall(r'BASELINE.*?average\s+([\d.]+)', data, re.DOTALL)

            if srf_matches:
                srf_acc = float(srf_matches[0])
                if baseline_matches:
                    baseline_acc = float(baseline_matches[0])
                    improvement = srf_acc - baseline_acc

                    exp["status"] = "success"
                    exp["srf_acc"] = srf_acc
                    exp["baseline_acc"] = baseline_acc
                    exp["improvement"] = improvement

                    print(f"   ✓ Baseline: {baseline_acc:.4f} → SRF: {srf_acc:.4f} ({improvement:+.4f})")
                    return exp

            print(f"   ~ Completed (check log for details)")
            exp["status"] = "completed"
            return exp

        else:
            exp["status"] = "failed"
            print(f"   ✗ No result file")
            return exp

    except subprocess.TimeoutExpired:
        exp["status"] = "timeout"
        print(f"   ✗ Timed out")
        return exp
    except Exception as e:
        exp["status"] = "error"
        print(f"   ✗ Error: {e}")
        return exp


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple Qwen-7B investigation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--max_experiments", type=int, default=4, help="Max experiments")
    parser.add_argument("--output_dir", type=str, default="qwen7b_investigation", help="Output dir")

    args = parser.parse_args()

    print("🔍 QWEN-7B SRF INVESTIGATION")
    print("=" * 60)
    print(f"Model: {MODEL_CONFIG['model_id']}")
    print(f"GPU: {args.gpu}")
    print(f"Testing: Layer ranges + Intervention strength")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    experiments = INVESTIGATION_EXPERIMENTS[:args.max_experiments]

    print(f"\nRunning {len(experiments)} experiments...\n")

    # Run experiments
    results = []
    start_time = time.time()

    for i, exp in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] ", end="")
        result = run_experiment(exp, output_dir, args.gpu)
        results.append(result)

        # Save intermediate
        with open(output_dir / "results_partial.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    successful = [r for r in results if r.get("status") == "success"]

    print("\n" + "=" * 60)
    print("🎯 INVESTIGATION COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(successful)}/{len(experiments)}")

    if successful:
        print(f"\n🏆 RESULTS:")
        for r in sorted(successful, key=lambda x: x.get("improvement", 0), reverse=True):
            print(f"   {r['name']}: {r['improvement']:+.4f} (Baseline {r['baseline_acc']:.4f} → SRF {r['srf_acc']:.4f})")

        best = max(successful, key=lambda x: x.get("improvement", -999))
        print(f"\n💡 BEST CONFIG FOR QWEN-7B:")
        print(f"   Layers {best['layer_start']}-{best['layer_end']}")
        print(f"   Boost {best['boost_alpha']}/Suppress {best['suppress_alpha']}")

        if best['improvement'] > 0:
            print(f"\n   ✅ SRF WORKS for 7B! Use these parameters for full evaluation.")
        else:
            print(f"\n   ⚠️  SRF doesn't improve 7B. May need fundamental changes.")


if __name__ == "__main__":
    main()
