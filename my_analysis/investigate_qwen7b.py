#!/usr/bin/env python3
"""
Qwen-7B SRF Investigation - Why doesn't the 3B configuration work?

Systematic investigation of model scaling effects on SRF performance.

Key Hypotheses to Test:
1. Layer shift: Object reasoning moves to different layers in 7B
2. Intervention strength: 7B needs gentler or stronger intervention?
3. Head selection: Top-K% needs adjustment for larger models
4. Model capacity: 7B might need different CLIP integration

Usage:
    python investigate_qwen7b.py --gpu 3 --max_experiments 12
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# FOCUSED INVESTIGATION FOR QWEN-7B
# =============================================================================

INVESTIGATION_SPACE = {
    # HYPOTHESIS 1: Layer shift (7B might use different layers)
    "layer_ranges": [
        (6, 12),   # Earlier than 3B (more visual, less reasoning)
        (10, 16),  # Same as 3B (current baseline)
        (14, 20),  # Later than 3B (more reasoning)
        (8, 16),   # Wider range (covering both)
    ],

    # HYPOTHESIS 2: Intervention strength scaling
    "boost_suppress_pairs": [
        (1.0, 3.0),   # Much gentler (7B might be more sensitive)
        (2.0, 5.0),   # Same as 3B (current baseline)
        (3.0, 7.0),   # Stronger (7B might need more intervention)
        (1.5, 4.0),   # Balanced middle ground
    ],

    # HYPOTHESIS 3: Head selection adjustment
    "head_top_k_pct": [
        0.10,  # More selective (7B has more heads, pick the best)
        0.20,  # Same as 3B (current baseline)
        0.30,  # More inclusive (7B might need more heads)
    ],

    # HYPOTHESIS 4: CLIP saliency adjustments
    "clip_configs": [
        {
            "clip_coarse_grid": 7,
            "clip_top_k_pct": 0.30,
            "clip_use_soft": True,
            "method": "clip",
        },
        {
            "clip_coarse_grid": 5,  # Finer grid (7B processes higher-res features)
            "clip_top_k_pct": 0.25,
            "clip_use_soft": True,
            "method": "clip",
        },
    ],
}

# Model config
MODEL_CONFIG = {
    "arch": "qwen",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "n_samples": 200,  # Balance speed vs statistical significance
    "mode": "both",
    "splits": ["adversarial", "popular"],  # Focus on hardest splits first
}

# =============================================================================
# Experiment Management
# =============================================================================

def generate_investigation_experiments(investigation_space: Dict) -> List[Dict]:
    """Generate focused investigation experiments."""

    experiments = []
    exp_id = 0

    # We'll test hypotheses one at a time, not full factorial
    # This is faster and more interpretable

    # SET 1: Layer ranges (keep other params constant)
    for layers in investigation_space["layer_ranges"]:
        experiments.append({
            "exp_id": exp_id,
            "hypothesis": f"layer_shift_{layers[0]}_{layers[1]}",
            "layer_start": layers[0],
            "layer_end": layers[1],
            "boost_alpha": 2.0,  # Same as 3B baseline
            "suppress_alpha": 5.0,
            "head_top_k_pct": 0.20,
            "saliency_config": investigation_space["clip_configs"][0],
            "model_config": MODEL_CONFIG.copy(),
        })
        exp_id += 1

    # SET 2: Intervention strength
    for boost, suppress in investigation_space["boost_suppress_pairs"]:
        experiments.append({
            "exp_id": exp_id,
            "hypothesis": f"strength_{boost}_{suppress}",
            "layer_start": 8,
            "layer_end": 14,
            "boost_alpha": boost,
            "suppress_alpha": suppress,
            "head_top_k_pct": 0.20,
            "saliency_config": investigation_space["clip_configs"][0],
            "model_config": MODEL_CONFIG.copy(),
        })
        exp_id += 1

    # SET 3: Head selection
    for head_k in investigation_space["head_top_k_pct"]:
        experiments.append({
            "exp_id": exp_id,
            "hypothesis": f"head_selection_{int(head_k*100)}",
            "layer_start": 8,
            "layer_end": 14,
            "boost_alpha": 2.0,
            "suppress_alpha": 5.0,
            "head_top_k_pct": head_k,
            "saliency_config": investigation_space["clip_configs"][0],
            "model_config": MODEL_CONFIG.copy(),
        })
        exp_id += 1

    print(f"Generated {len(experiments)} investigation experiments")
    return experiments


def run_experiment(exp: Dict, output_dir: Path, gpu_id: int, conda_env: str = "mllm"):
    """Run investigation experiment."""

    python_path = f"/home/anna2/miniconda3/envs/{conda_env}/bin/python"

    print(f"\n🔬 [{exp['hypothesis']}] Exp {exp['exp_id']}")
    print(f"   Layers {exp['layer_start']}-{exp['layer_end']}, "
          f"Boost {exp['boost_alpha']}/{exp['suppress_alpha']}, "
          f"Head-K {exp['head_top_k_pct']*100:.0f}%")

    log_file = output_dir / f"exp_{exp['exp_id']:04d}_{exp['hypothesis']}.log"
    result_file = output_dir / f"result_{exp['exp_id']:04d}.json"

    # Environment
    env_vars = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "HF_HOME": "/home/anna2/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
        "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
    }

    # SRF config
    srf_config = {
        "clip_coarse_grid": exp["saliency_config"]["clip_coarse_grid"],
        "clip_top_k_pct": exp["saliency_config"]["clip_top_k_pct"],
        "clip_use_soft": exp["saliency_config"]["clip_use_soft"],
        "boost_alpha": exp["boost_alpha"],
        "suppress_alpha": exp["suppress_alpha"],
        "head_top_k_pct": exp["head_top_k_pct"],
        "sys_beta": 0.10,
        "background_eps": 0.0,
        "n_calib_samples": 20,
        "calib_seed": 0,
        "layer_start": exp["layer_start"],
        "layer_end": exp["layer_end"],
        "saliency_method": exp["saliency_config"]["method"],
    }

    # Experiment script
    exp_script = f'''
import os, sys, json
os.environ.update({env_vars})
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")

# Override MODEL_ID in autoresearch/pope_eval.py
import subprocess
import tempfile

# Create a temporary modified pope_eval.py
pope_eval_content = open("/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch/pope_eval.py").read()

# Replace model ID and constants
modified_eval = pope_eval_content.replace(
    'MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"',
    f'MODEL_ID       = "{exp["model_config"]["model_id"]}"'
)

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(modified_eval)
    temp_eval_path = f.name

# Import and run
import importlib.util
spec = importlib.util.spec_from_file_location("pope_eval_temp", temp_eval_path)
pope_eval_module = importlib.util.module_from_spec(spec)

# Override srf config
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch")

from srf import SRF, ARCH_DEFAULTS

SRF.update({srf_config})
ARCH_DEFAULTS["qwen"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["qwen"]["layer_end"] = {exp['layer_end']}

# Run the evaluation
pope_eval_module.run()

# Clean up
import os
os.unlink(temp_eval_path)
'''

    try:
        import subprocess
        current_env = os.environ.copy()
        current_env.update(env_vars)

        with open(log_file, 'w') as f:
            result = subprocess.run(
                [python_path, "-c", exp_script],
                env=current_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1200,  # 20 minutes
            )

        if result_file.exists():
            with open(result_file) as f:
                data = f.read()
            try:
                results = json.loads(data)
            except:
                # Try to parse from text format
                import re
                acc_match = re.search(r'POPE accuracy:\s+([\d.]+)', data)
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    baseline_match = re.search(r'BASELINE.*?([\d.]+)', data, re.DOTALL)
                    baseline = float(baseline_match.group(1)) if baseline_match else None

                    results = {
                        "accuracy": accuracy,
                        "baseline_accuracy": baseline,
                        "improvement": accuracy - baseline if baseline else 0.0,
                    }

            exp["status"] = "success"
            exp["results"] = results
            exp["result_file"] = str(result_file)

            improvement = results.get("improvement", 0.0)
            baseline = results.get("baseline_accuracy", results.get("accuracy", 0.0))
            srf_acc = results.get("accuracy", 0.0)

            print(f"   ✓ Result: Baseline {baseline:.4f} → SRF {srf_acc:.4f} ({improvement:+.4f})")
            return True
        else:
            exp["status"] = "failed"
            print(f"   ✗ Failed: No result file")
            return False

    except subprocess.TimeoutExpired:
        exp["status"] = "timeout"
        print(f"   ✗ Timed out")
        return False
    except Exception as e:
        exp["status"] = "error"
        print(f"   ✗ Error: {e}")
        return False


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(results: List[Dict]):
    """Analyze investigation results and provide recommendations."""

    print("\n" + "=" * 80)
    print("🔍 INVESTIGATION RESULTS ANALYSIS")
    print("=" * 80)

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        print("❌ No successful experiments to analyze")
        return

    # Find best result
    best = max(successful, key=lambda x: x["results"].get("improvement", -999))

    print(f"\n🏆 BEST RESULT:")
    print(f"   Hypothesis: {best['hypothesis']}")
    print(f"   Improvement: {best['results'].get('improvement', 0):+4f}")
    print(f"   Layers: {best['layer_start']}-{best['layer_end']}")
    print(f"   Boost/Suppress: {best['boost_alpha']}/{best['suppress_alpha']}")
    print(f"   Head-K: {best['head_top_k_pct']*100:.0f}%")

    # Analyze by hypothesis type
    print(f"\n📊 BY HYPOTHESIS:")

    # Layer ranges
    layer_results = {r['hypothesis']: r for r in successful if r['hypothesis'].startswith('layer_shift_')}
    if layer_results:
        print(f"\n  Layer Ranges:")
        for hyp, res in sorted(layer_results.items()):
            improvement = res["results"].get("improvement", 0.0)
            status = "✓" if improvement > 0 else "✗"
            print(f"    {status} {hyp}: {improvement:+.4f}")

    # Intervention strength
    strength_results = {r['hypothesis']: r for r in successful if r['hypothesis'].startswith('strength_')}
    if strength_results:
        print(f"\n  Intervention Strength:")
        for hyp, res in sorted(strength_results.items()):
            improvement = res["results"].get("improvement", 0.0)
            status = "✓" if improvement > 0 else "✗"
            print(f"    {status} {hyp}: {improvement:+.4f}")

    # Head selection
    head_results = {r['hypothesis']: r for r in successful if r['hypothesis'].startswith('head_selection_')}
    if head_results:
        print(f"\n  Head Selection:")
        for hyp, res in sorted(head_results.items()):
            improvement = res["results"].get("improvement", 0.0)
            status = "✓" if improvement > 0 else "✗"
            print(f"    {status} {hyp}: {improvement:+.4f}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR QWEN-7B:")

    if any(r["results"].get("improvement", 0) > 0 for r in successful):
        print(f"   ✅ SRF CAN work for Qwen-7B! Use these parameters:")
        print(f"      • Layer range: {best['layer_start']}-{best['layer_end']}")
        print(f"      • Boost/Suppress: {best['boost_alpha']}/{best['suppress_alpha']}")
        print(f"      • Head selection: top {best['head_top_k_pct']*100:.0f}%")
    else:
        print(f"   ❌ SRF doesn't help Qwen-7B with current approach.")
        print(f"      Possible issues:")
        print(f"      • 7B might have different architecture (check layers)")
        print(f"      • Intervention strategy might need fundamental changes")
        print(f"      • Consider model-specific tuning")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen-7B SRF Investigation")
    parser.add_argument("--gpu", type=int, default=3, help="GPU ID")
    parser.add_argument("--max_experiments", type=int, default=12, help="Max experiments")
    parser.add_argument("--output_dir", type=str, default="qwen7b_investigation", help="Output dir")

    args = parser.parse_args()

    print("🔍 QWEN-7B SRF INVESTIGATION")
    print("=" * 60)
    print(f"Model: {MODEL_CONFIG['model_id']}")
    print(f"GPU: {args.gpu}")
    print(f"Max experiments: {args.max_experiments}")
    print()
    print("Investigating why Qwen-3B configuration doesn't work for Qwen-7B")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate experiments
    experiments = generate_investigation_experiments(INVESTIGATION_SPACE)

    # Limit experiments
    experiments = experiments[:args.max_experiments]

    print(f"\nRunning {len(experiments)} experiments...\n")

    # Run experiments
    results = []
    start_time = time.time()

    for i, exp in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] ", end="")
        success = run_experiment(exp, output_dir, args.gpu)
        results.append(exp)

        # Save intermediate
        with open(output_dir / f"results_partial.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    elapsed = time.time() - start_time
    with open(output_dir / "results_final.json", 'w') as f:
        json.dump({
            "elapsed_time": elapsed,
            "experiments": results,
            "successful": sum(1 for r in results if r.get("status") == "success"),
        }, f, indent=2)

    # Analyze
    analyze_results(results)

    print(f"\n✅ Investigation complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
