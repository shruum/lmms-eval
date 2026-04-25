#!/usr/bin/env python3
"""
Overnight SRF Experiments - Comprehensive Search Around Best Configuration

Based on investigation findings, explores around the best config to find even better results.
Model-agnostic design - works for both Qwen-3B, Qwen-7B, and LLaVA.

Usage:
    # Qwen-7B (GPU 0) - explore around layers 14-20
    python overnight_experiments.py --gpu 0 --model Qwen/Qwen2.5-VL-7B-Instruct --arch qwen --focus late

    # Qwen-3B (GPU 1) - explore around layers 8-14
    python overnight_experiments.py --gpu 1 --model Qwen/Qwen2.5-VL-3B-Instruct --arch qwen --focus mid

    # LLaVA (GPU 2) - explore around layers 10-16
    python overnight_experiments.py --gpu 2 --model llava-hf/llava-1.5-7b-hf --arch llava --focus standard
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# =============================================================================
# Experiment Grid Design
# =============================================================================

def get_experiments_for_focus(focus: str, arch: str) -> List[Dict]:
    """
    Generate experiments based on focus area and architecture.

    Focus areas:
    - 'late': Layers 14-20 (optimal for 7B)
    - 'mid': Layers 8-14 (optimal for 3B)
    - 'standard': Layers 10-16 (LLaVA baseline)
    - 'wide': Broad exploration
    """

    if focus == "late":
        # Explore around 14-20 (7B optimal)
        layer_ranges = [
            (14, 20),  # Current best
            (13, 19),  # Shift earlier
            (15, 21),  # Shift later
            (14, 18),  # Narrower
            (16, 20),  # Narrower, later
            (12, 18),  # Wide overlap
            (15, 19),  # Narrow core
        ]
    elif focus == "mid":
        # Explore around 8-14 (3B optimal)
        layer_ranges = [
            (8, 14),   # Current best for 3B
            (7, 13),   # Shift earlier
            (9, 15),   # Shift later
            (8, 12),   # Narrower
            (10, 14),  # Narrower, later
            (6, 12),   # Wide overlap
        ]
    elif focus == "standard":
        # LLaVA baseline area
        layer_ranges = [
            (10, 16),  # LLaVA baseline
            (9, 15),   # Shift earlier
            (11, 17),  # Shift later
            (10, 14),  # Narrower
            (12, 16),  # Narrower, later
        ]
    else:  # wide
        # Broad exploration
        layer_ranges = [
            (6, 12),
            (8, 14),
            (10, 16),
            (12, 18),
            (14, 20),
            (16, 22),
        ]

    # Intervention strengths to test
    strengths = [
        (1.5, 4.0),  # Gentler
        (2.0, 5.0),  # Current best
        (2.5, 6.0),  # Stronger
        (1.0, 3.0),  # Very gentle
    ]

    # Head selection to test
    head_selections = [0.15, 0.20, 0.25]

    # CLIP configurations
    clip_configs = [
        {"clip_coarse_grid": 7, "clip_top_k_pct": 0.30, "clip_use_soft": True},
        {"clip_coarse_grid": 5, "clip_top_k_pct": 0.25, "clip_use_soft": True},
    ]

    experiments = []
    exp_id = 0

    # Priority 1: Layer ranges with best strength (2.0/5.0)
    for layers in layer_ranges:
        for clip_cfg in clip_configs[:1]:  # Just main CLIP config
            experiments.append({
                "exp_id": exp_id,
                "priority": 1,
                "layer_start": layers[0],
                "layer_end": layers[1],
                "boost_alpha": 2.0,
                "suppress_alpha": 5.0,
                "head_top_k_pct": 0.20,
                "clip_coarse_grid": clip_cfg["clip_coarse_grid"],
                "clip_top_k_pct": clip_cfg["clip_top_k_pct"],
                "clip_use_soft": clip_cfg["clip_use_soft"],
                "saliency_method": "clip",
                "description": f"Layers {layers[0]}-{layers[1]}",
            })
            exp_id += 1

    # Priority 2: Intervention strengths around best layer range
    best_layers = layer_ranges[0]  # Use first (current best)
    for boost, suppress in strengths:
        for head_k in head_selections:
            experiments.append({
                "exp_id": exp_id,
                "priority": 2,
                "layer_start": best_layers[0],
                "layer_end": best_layers[1],
                "boost_alpha": boost,
                "suppress_alpha": suppress,
                "head_top_k_pct": head_k,
                "clip_coarse_grid": 7,
                "clip_top_k_pct": 0.30,
                "clip_use_soft": True,
                "saliency_method": "clip",
                "description": f"Strength {boost}/{suppress}, head {int(head_k*100)}%",
            })
            exp_id += 1

    # Priority 3: Alternative CLIP configs for top 2 layer ranges
    for layers in layer_ranges[:2]:
        experiments.append({
            "exp_id": exp_id,
            "priority": 3,
            "layer_start": layers[0],
            "layer_end": layers[1],
            "boost_alpha": 2.0,
            "suppress_alpha": 5.0,
            "head_top_k_pct": 0.20,
            "clip_coarse_grid": 5,  # Finer grid
            "clip_top_k_pct": 0.25,
            "clip_use_soft": True,
            "saliency_method": "clip",
            "description": f"Layers {layers[0]}-{layers[1]}, fine CLIP",
        })
        exp_id += 1

    # Sort by priority
    experiments.sort(key=lambda x: x["priority"])

    print(f"Generated {len(experiments)} experiments")
    print(f"  Priority 1 (layer ranges): {sum(1 for e in experiments if e['priority'] == 1)}")
    print(f"  Priority 2 (strengths): {sum(1 for e in experiments if e['priority'] == 2)}")
    print(f"  Priority 3 (CLIP configs): {sum(1 for e in experiments if e['priority'] == 3)}")

    return experiments

# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(exp: Dict, model_config: Dict, output_dir: Path, gpu_id: int) -> Dict:
    """Run a single experiment."""

    print(f"\n🔬 [{exp['exp_id']}] {exp['description']}")
    print(f"   Layers {exp['layer_start']}-{exp['layer_end']}, "
          f"Boost {exp['boost_alpha']}/{exp['suppress_alpha']}, "
          f"Head {int(exp['head_top_k_pct']*100)}%")

    log_file = output_dir / f"exp_{exp['exp_id']:04d}_{exp['description'].replace(' ', '_').replace('/', '_').replace(',', '_')}.log"
    result_file = output_dir / f"result_{exp['exp_id']:04d}.json"

    python_path = "/home/anna2/miniconda3/envs/mllm/bin/python"

    env_vars = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
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

# Update both SRF and ARCH_DEFAULTS (layers are in ARCH_DEFAULTS)
SRF.update({exp})
ARCH_DEFAULTS["{model_config['arch']}"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["{model_config['arch']}"]["layer_end"] = {exp['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "{model_config['arch']}",
    "--model", "{model_config['model_id']}",
    "--n_samples", "{model_config['n_samples']}",
    "--mode", "{model_config['mode']}",
    "--output", r"{result_file}",
]

main()
'''

    try:
        current_env = os.environ.copy()
        current_env.update(env_vars)

        start = time.time()
        with open(log_file, 'w') as f:
            result = subprocess.run(
                [python_path, "-c", exp_script],
                env=current_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=900,  # 15 minutes per experiment
            )
        elapsed = time.time() - start

        # Parse results
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)

            if "metrics" in data:
                metrics = data["metrics"]
                baseline_accs = [split["baseline"]["accuracy"] for split in metrics.values()]
                srf_accs = [split["srf"]["accuracy"] for split in metrics.values()]

                baseline_acc = sum(baseline_accs) / len(baseline_accs)
                srf_acc = sum(srf_accs) / len(srf_accs)
                improvement = srf_acc - baseline_acc

                exp["status"] = "success"
                exp["srf_acc"] = srf_acc
                exp["baseline_acc"] = baseline_acc
                exp["improvement"] = improvement
                exp["elapsed_time"] = elapsed

                print(f"   ✓ Baseline: {baseline_acc:.4f} → SRF: {srf_acc:.4f} ({improvement:+.4f}) [{elapsed/60:.1f}min]")
                return exp

        exp["status"] = "failed"
        exp["elapsed_time"] = elapsed
        print(f"   ✗ Failed (no results) [{elapsed/60:.1f}min]")
        return exp

    except subprocess.TimeoutExpired:
        exp["status"] = "timeout"
        print(f"   ✗ Timed out")
        return exp
    except Exception as e:
        exp["status"] = "error"
        exp["error"] = str(e)
        print(f"   ✗ Error: {e}")
        return exp

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight SRF experiments")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID")
    parser.add_argument("--model", type=str, required=True, help="Model ID")
    parser.add_argument("--arch", type=str, required=True, choices=["qwen", "llava"], help="Architecture")
    parser.add_argument("--focus", type=str, default="late", choices=["late", "mid", "standard", "wide"],
                       help="Focus area (late=7B, mid=3B, standard=LLaVA, wide=broad)")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per split")
    parser.add_argument("--max_experiments", type=int, default=100, help="Max experiments to run")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        model_name = args.model.split('/')[-1]
        args.output_dir = f"overnight_{args.focus}_{model_name}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🌙 OVERNIGHT SRF EXPERIMENTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Architecture: {args.arch}")
    print(f"Focus: {args.focus}")
    print(f"GPU: {args.gpu}")
    print(f"Samples per split: {args.n_samples}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Generate experiments
    experiments = get_experiments_for_focus(args.focus, args.arch)
    experiments = experiments[:args.max_experiments]

    print(f"\nRunning {len(experiments)} experiments...")
    print(f"Estimated time: {len(experiments) * 10 / 60:.1f} hours")
    print("\nPress Ctrl+C to stop early (results are saved automatically)\n")

    # Model config
    model_config = {
        "arch": args.arch,
        "model_id": args.model,
        "n_samples": args.n_samples,
        "mode": "both",
    }

    # Run experiments
    results = []
    start_time = time.time()
    best_improvement = -999
    best_config = None

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] ", end="")

        result = run_experiment(exp, model_config, output_dir, args.gpu)
        results.append(result)

        # Track best
        if result.get("status") == "success" and result.get("improvement", 0) > best_improvement:
            best_improvement = result["improvement"]
            best_config = result.copy()
            print(f"   🏆 NEW BEST!")

        # Save intermediate results
        with open(output_dir / "results_partial.json", 'w') as f:
            json.dump({
                "progress": f"{i+1}/{len(experiments)}",
                "elapsed_time": time.time() - start_time,
                "best_improvement": best_improvement,
                "best_config": best_config,
                "results": results,
            }, f, indent=2)

        # Small pause between experiments
        time.sleep(2)

    # Final summary
    total_elapsed = time.time() - start_time
    successful = [r for r in results if r.get("status") == "success"]

    print("\n" + "=" * 60)
    print("🎉 OVERNIGHT EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"Successful: {len(successful)}/{len(experiments)}")

    if successful:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Improvement: {best_config['improvement']:+.4f}")
        print(f"   Baseline: {best_config['baseline_acc']:.4f} → SRF: {best_config['srf_acc']:.4f}")
        print(f"   Layers: {best_config['layer_start']}-{best_config['layer_end']}")
        print(f"   Boost/Suppress: {best_config['boost_alpha']}/{best_config['suppress_alpha']}")
        print(f"   Head selection: top {int(best_config['head_top_k_pct']*100)}%")

        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, r in enumerate(sorted(successful, key=lambda x: x.get("improvement", 0), reverse=True)[:5]):
            print(f"   {i+1}. {r['description']}: {r['improvement']:+.4f} "
                  f"(Layers {r['layer_start']}-{r['layer_end']}, "
                  f"{r['boost_alpha']}/{r['suppress_alpha']})")

    # Save final results
    with open(output_dir / "results_final.json", 'w') as f:
        json.dump({
            "model_config": model_config,
            "elapsed_time": total_elapsed,
            "best_improvement": best_improvement,
            "best_config": best_config,
            "results": results,
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
