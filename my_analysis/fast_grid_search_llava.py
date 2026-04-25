#!/usr/bin/env python3
"""
FAST Grid Search for LLaVA SRF Parameters - Staged Approach

Stage 1: Coarse search (18 experiments, ~1.5-3 hours)
- Tests most important parameters first
- Find promising areas, then do focused search

Usage:
    python fast_grid_search_llava.py --gpu 0 --physical_gpu 2 --stage 1
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
# STAGE 1: COARSE SEARCH (18 experiments)
# =============================================================================

COARSE_SEARCH_SPACE = {
    # Layer ranges - WHERE to apply SRF (most important!)
    "layer_ranges": [
        (8, 12),   # Early-mid: more visual features
        (10, 16),  # Mid: ClearSight finding, likely optimal
        (12, 18),  # Mid-late: more reasoning, less visual
    ],

    # Intervention strength - HOW MUCH to modify (critical!)
    "boost_suppress_pairs": [
        (1.5, 3.0),   # Gentle: less intervention (current default hurts LLaVA)
        (2.0, 4.0),   # Moderate: balanced approach
        (2.5, 5.0),   # Strong: more aggressive intervention
    ],

    # Saliency methods - WHICH patches to boost (important comparison)
    "saliency_configs": [
        {
            "method": "clip",
            "clip_coarse_grid": 7,
            "clip_top_k_pct": 0.30,
            "clip_use_soft": True,
        },
        {
            "method": "dino",
            "dino_model": "facebook/dino-vitb16",
            "dino_grid_size": 8,
            "dino_top_k_pct": 0.30,
            "dino_use_soft": True,
        },
    ],

    # Keep constant for now (can tune in Stage 2)
    "head_top_k_pct": [0.20],
}

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIG = {
    "arch": "llava",
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "n_samples": 100,  # Good balance: fast but meaningful results
    "mode": "both",
    "splits": ["adversarial", "popular"],  # Skip "random" for speed
}

# =============================================================================
# Experiment Management
# =============================================================================

def generate_experiments(search_space: Dict) -> List[Dict[str, Any]]:
    """Generate experiment combinations."""

    experiments = []
    layer_ranges = search_space["layer_ranges"]
    boost_suppress_pairs = search_space["boost_suppress_pairs"]
    saliency_configs = search_space["saliency_configs"]
    head_top_k_pcts = search_space["head_top_k_pct"]

    exp_id = 0
    for layer_start, layer_end in layer_ranges:
        for boost, suppress in boost_suppress_pairs:
            for saliency_config in saliency_configs:
                for head_top_k_pct in head_top_k_pcts:
                    exp = {
                        "exp_id": exp_id,
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "boost_alpha": boost,
                        "suppress_alpha": suppress,
                        "head_top_k_pct": head_top_k_pct,
                        "saliency_config": saliency_config,
                        "model_config": MODEL_CONFIG.copy(),
                    }
                    experiments.append(exp)
                    exp_id += 1

    print(f"Generated {len(experiments)} experiments")
    return experiments


def run_experiment(exp: Dict, output_dir: Path, gpu_id: int, physical_gpu: int = None, conda_env: str = "mllm"):
    """Run a single experiment."""

    python_path = f"/home/anna2/miniconda3/envs/{conda_env}/bin/python"

    print(f"[GPU {physical_gpu if physical_gpu else gpu_id}] Exp {exp['exp_id']}: "
          f"Layers {exp['layer_start']}-{exp['layer_end']}, "
          f"Boost {exp['boost_alpha']}/{exp['suppress_alpha']}, "
          f"Saliency {exp['saliency_config']['method']}")

    log_file = output_dir / f"exp_{exp['exp_id']:04d}.log"
    result_file = output_dir / f"result_{exp['exp_id']:04d}.json"

    # Environment
    env_vars = {
        "CUDA_VISIBLE_DEVICES": str(physical_gpu if physical_gpu is not None else gpu_id),
        "HF_HOME": "/home/anna2/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
        "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
    }

    # SRF config
    srf_config = {
        "clip_coarse_grid": exp["saliency_config"].get("clip_coarse_grid", 7),
        "clip_top_k_pct": exp["saliency_config"].get("clip_top_k_pct", 0.30),
        "clip_use_soft": exp["saliency_config"].get("clip_use_soft", True),
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

    # Add DINO config
    if exp["saliency_config"]["method"] == "dino":
        srf_config["dino_model"] = exp["saliency_config"]["dino_model"]
        srf_config["dino_grid_size"] = exp["saliency_config"]["dino_grid_size"]
        srf_config["dino_top_k_pct"] = exp["saliency_config"]["dino_top_k_pct"]
        srf_config["dino_use_soft"] = exp["saliency_config"]["dino_use_soft"]

    # Experiment script
    exp_script = f'''
import os, sys, json
os.environ.update({env_vars})
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch")

from pope_srf_eval import main, SRF, ARCH_DEFAULTS

SRF.update({srf_config})
ARCH_DEFAULTS["llava"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["llava"]["layer_end"] = {exp['layer_end']}

sys.argv = ["pope_srf_eval.py", "--arch", "llava", "--model", "{exp['model_config']['model_id']}",
    "--n_samples", "{exp['model_config']['n_samples']}", "--mode", "{exp['model_config']['mode']}",
    "--output", r"{result_file}"]

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
                timeout=1800,
            )

        # Check results
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)

            exp["status"] = "success"
            exp["results"] = results

            # Print summary
            if "srf" in results:
                srf_results = results["srf"]
                baseline_acc = srf_results.get('baseline_acc', 0)
                srf_acc = srf_results.get('srf_acc', 0)
                improvement = srf_acc - baseline_acc
                print(f"  ✓ Baseline: {baseline_acc:.4f}, SRF: {srf_acc:.4f} ({improvement:+.4f})")
            return True
        else:
            exp["status"] = "failed"
            return False

    except subprocess.TimeoutExpired:
        exp["status"] = "timeout"
        print(f"  ✗ Timed out")
        return False
    except Exception as e:
        exp["status"] = "error"
        print(f"  ✗ Failed: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fast grid search for LLaVA SRF")
    parser.add_argument("--gpu", type=int, default=0, help="Logical GPU ID")
    parser.add_argument("--physical_gpu", type=int, default=None, help="Physical GPU ID")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs")
    parser.add_argument("--max_experiments", type=int, default=20, help="Max experiments")
    parser.add_argument("--output_dir", type=str, default="fast_grid_search", help="Output dir")
    parser.add_argument("--stage", type=int, default=1, help="Stage (1=coarse, 2=focused)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Choose search space based on stage
    if args.stage == 1:
        search_space = COARSE_SEARCH_SPACE
        print("🔍 STAGE 1: Coarse Search (18 experiments)")
        print("   Testing: 3 layer ranges × 3 boost/suppress × 2 saliency methods")
    else:
        print("Stage 2 not implemented yet - run Stage 1 first!")
        return

    # Generate experiments
    experiments = generate_experiments(search_space)

    # Distribute across GPUs
    my_experiments = [exp for exp in experiments if exp['exp_id'] % args.total_gpus == args.gpu]
    my_experiments = my_experiments[:args.max_experiments]

    print(f"GPU {args.gpu} (physical {args.physical_gpu}): {len(my_experiments)} experiments")

    # Run experiments
    results = []
    start_time = time.time()

    for i, exp in enumerate(my_experiments):
        print(f"\n[{i+1}/{len(my_experiments)}] Experiment {exp['exp_id']}")
        success = run_experiment(exp, output_dir, args.gpu, args.physical_gpu)
        exp["success"] = success
        results.append(exp)

        # Save intermediate results
        with open(output_dir / f"results_gpu{args.gpu}_partial.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    elapsed = time.time() - start_time
    with open(output_dir / f"results_gpu{args.gpu}_final.json", 'w') as f:
        json.dump({
            "stage": args.stage,
            "gpu_id": args.gpu,
            "physical_gpu": args.physical_gpu,
            "elapsed_time": elapsed,
            "experiments": results,
            "successful": sum(1 for r in results if r.get("status") == "success"),
        }, f, indent=2)

    print(f"\n✅ GPU {args.gpu} completed in {elapsed/60:.1f} minutes")
    print(f"   Successful: {sum(1 for r in results if r.get('status') == 'success')}/{len(my_experiments)}")

    # Show quick summary
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        print(f"\n🏆 Top 3 configurations:")
        for r in sorted(successful, key=lambda x: x['results']['srf']['srf_acc'], reverse=True)[:3]:
            srf = r['results']['srf']
            print(f"   Layers {r['layer_start']}-{r['layer_end']}, "
                  f"Boost {r['boost_alpha']}/{r['suppress_alpha']}, "
                  f"{r['saliency_config']['method']}: {srf['srf_acc']:.4f}")


if __name__ == "__main__":
    main()
