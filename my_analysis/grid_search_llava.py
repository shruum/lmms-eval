#!/usr/bin/env python3
"""
Grid Search for LLaVA SRF hyperparameters

Tests different combinations of:
- Layer ranges (where to apply SRF)
- Boost/suppress ratios (intervention strength)
- Saliency methods (CLIP vs DINO)
- CLIP/DINO parameters (grid size, top-k, etc.)

Usage:
    python grid_search_llava.py --gpu 0 --start_idx 0 --total_gpus 2
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# Hyperparameter Search Space
# =============================================================================

SEARCH_SPACE = {
    # Layer ranges to test (based on LLaVA architecture analysis)
    "layer_ranges": [
        (8, 12),   # Early-mid layers (more visual)
        (10, 16),  # Mid layers (ClearSight finding)
        (12, 18),  # Mid-late layers (more reasoning)
        (8, 16),   # Wide range
    ],

    # Intervention strength combinations
    "boost_suppress_pairs": [
        (1.5, 3.0),   # Gentle intervention
        (2.0, 5.0),   # Current default (hurts LLaVA)
        (1.0, 5.0),   # Suppress only (no boost)
        (3.0, 7.0),   # Aggressive intervention
        (2.5, 4.0),   # Moderate-strong
    ],

    # Saliency method parameters
    "saliency_configs": [
        {
            "method": "clip",
            "clip_coarse_grid": 7,
            "clip_top_k_pct": 0.30,
            "clip_use_soft": True,
        },
        {
            "method": "clip",
            "clip_coarse_grid": 5,  # Finer grid
            "clip_top_k_pct": 0.20,  # More selective
            "clip_use_soft": True,
        },
        {
            "method": "clip",
            "clip_coarse_grid": 6,
            "clip_top_k_pct": 0.40,  # More inclusive
            "clip_use_soft": False,  # Hard mask
        },
        {
            "method": "dino",
            "dino_model": "facebook/dino-vitb16",  # DINO ViT-B/16
            "dino_grid_size": 8,
            "dino_top_k_pct": 0.30,
            "dino_use_soft": True,
        },
        {
            "method": "dino",
            "dino_model": "facebook/dino-vitb16",
            "dino_grid_size": 6,
            "dino_top_k_pct": 0.20,
            "dino_use_soft": True,
        },
    ],

    # Head selection parameters
    "head_top_k_pct": [0.15, 0.20, 0.25],
}

# =============================================================================
# Experiment Configuration
# =============================================================================

MODEL_CONFIG = {
    "arch": "llava",
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "n_samples": 100,  # Use smaller sample for faster search
    "mode": "both",    # Test both baseline and SRF
    "splits": ["adversarial", "popular"],  # Test 2 splits first (can add random later)
}

# =============================================================================
# Utilities
# =============================================================================

def generate_experiments(search_space: Dict) -> List[Dict[str, Any]]:
    """Generate all experiment combinations from search space."""

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


def create_experiment_script(exp: Dict, output_dir: Path, gpu_id: int, physical_gpu: int = None) -> Path:
    """Create a temporary Python script to run a single experiment."""

    script_path = output_dir / f"exp_{exp['exp_id']:04d}_gpu{gpu_id}.py"
    result_file = output_dir / f"result_exp_{exp['exp_id']:04d}.json"

    # Build the SRF config
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

    # Add DINO-specific config if needed
    if exp["saliency_config"]["method"] == "dino":
        srf_config["dino_model"] = exp["saliency_config"]["dino_model"]
        srf_config["dino_grid_size"] = exp["saliency_config"]["dino_grid_size"]
        srf_config["dino_top_k_pct"] = exp["saliency_config"]["dino_top_k_pct"]
        srf_config["dino_use_soft"] = exp["saliency_config"]["dino_use_soft"]

    # Create experiment script
    script_content = f'''#!/usr/bin/env python3
"""
Experiment {exp['exp_id']} - GPU {gpu_id}
Layer range: {exp['layer_start']}-{exp['layer_end']}
Boost/Suppress: {exp['boost_alpha']}/{exp['suppress_alpha']}
Saliency: {exp['saliency_config']['method']}
"""

import sys
import os
from pathlib import Path

# Set environment - use physical GPU if specified
physical_gpu = {physical_gpu if physical_gpu is not None else 'None'}
gpu_id = {physical_gpu if physical_gpu is not None else gpu_id}
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Override the hardcoded /volumes2 path in pope_srf_eval.py
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/anna2/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/anna2/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/home/anna2/.cache/huggingface/hub"

# Add paths - use explicit path to my_analysis directory
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch")

# Import after setting environment
import torch
from pope_srf_eval import main, load_llava, SRF, ARCH_DEFAULTS

# Override SRF config
SRF.update({srf_config})

# Override architecture defaults
ARCH_DEFAULTS["llava"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["llava"]["layer_end"] = {exp['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "llava",
    "--model", "{exp['model_config']['model_id']}",
    "--n_samples", "{exp['model_config']['n_samples']}",
    "--mode", "{exp['model_config']['mode']}",
    "--output", r"{result_file}",
]

if __name__ == "__main__":
    main()
'''

    script_path.write_text(script_content)
    return script_path


def run_experiment(exp: Dict, output_dir: Path, gpu_id: int, physical_gpu: int = None, conda_env: str = "mllm"):
    """Run a single experiment using the validated test script approach."""

    python_path = f"/home/anna2/miniconda3/envs/{conda_env}/bin/python"

    print(f"[GPU {physical_gpu if physical_gpu else gpu_id}] Running experiment {exp['exp_id']}: "
          f"Layers {exp['layer_start']}-{exp['layer_end']}, "
          f"Boost {exp['boost_alpha']}, Suppress {exp['suppress_alpha']}, "
          f"Saliency {exp['saliency_config']['method']}")

    log_file = output_dir / f"exp_{exp['exp_id']:04d}_gpu{gpu_id}.log"
    result_file = output_dir / f"result_exp_{exp['exp_id']:04d}.json"

    # Build environment variables
    env_vars = {
        "CUDA_VISIBLE_DEVICES": str(physical_gpu if physical_gpu is not None else gpu_id),
        "HF_HOME": "/home/anna2/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/anna2/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/home/anna2/.cache/huggingface/datasets",
        "HF_HUB_CACHE": "/home/anna2/.cache/huggingface/hub",
    }

    # Build SRF config
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

    # Add DINO config if needed
    if exp["saliency_config"]["method"] == "dino":
        srf_config["dino_model"] = exp["saliency_config"]["dino_model"]
        srf_config["dino_grid_size"] = exp["saliency_config"]["dino_grid_size"]
        srf_config["dino_top_k_pct"] = exp["saliency_config"]["dino_top_k_pct"]
        srf_config["dino_use_soft"] = exp["saliency_config"]["dino_use_soft"]

    try:
        import subprocess
        current_env = os.environ.copy()
        current_env.update(env_vars)

        # Create experiment script (inline, using the working test approach)
        exp_script = f'''
import os
import sys
import json

# Set environment first
os.environ["CUDA_VISIBLE_DEVICES"] = "{physical_gpu if physical_gpu is not None else gpu_id}"
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/anna2/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/home/anna2/.cache/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = "/home/anna2/.cache/huggingface/hub"

# Add paths
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis")
sys.path.insert(0, "/home/anna2/shruthi/lmms-eval/my_analysis/autoresearch")

# Import after setting environment
from pope_srf_eval import main, SRF, ARCH_DEFAULTS

# Update SRF config
SRF.update({srf_config})
ARCH_DEFAULTS["llava"]["layer_start"] = {exp['layer_start']}
ARCH_DEFAULTS["llava"]["layer_end"] = {exp['layer_end']}

# Set model config
sys.argv = [
    "pope_srf_eval.py",
    "--arch", "llava",
    "--model", "{exp['model_config']['model_id']}",
    "--n_samples", "{exp['model_config']['n_samples']}",
    "--mode", "{exp['model_config']['mode']}",
    "--output", r"{result_file}",
]

main()
'''

        with open(log_file, 'w') as f:
            result = subprocess.run(
                [python_path, "-c", exp_script],
                env=current_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1800,  # 30 minutes max per experiment
            )

        # Load results if successful
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
            exp["error"] = "Result file not found"
            return False

    except subprocess.TimeoutExpired:
        exp["status"] = "timeout"
        exp["error"] = "Experiment timed out after 30 minutes"
        print(f"  ✗ Experiment {exp['exp_id']} timed out")
        return False
    except Exception as e:
        exp["status"] = "error"
        exp["error"] = str(e)
        print(f"  ✗ Experiment {exp['exp_id']} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Grid search for LLaVA SRF parameters")
    parser.add_argument("--gpu", type=int, default=0, help="Logical GPU ID (0, 1, 2, ...)")
    parser.add_argument("--physical_gpu", type=int, default=None, help="Physical GPU ID (overrides CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs in parallel run")
    parser.add_argument("--max_experiments", type=int, default=50, help="Max experiments to run")
    parser.add_argument("--output_dir", type=str, default="grid_search_results", help="Output directory")
    parser.add_argument("--conda_env", type=str, default="mllm", help="Conda environment")
    parser.add_argument("--quick_test", action="store_true", help="Quick test with 3 experiments")

    args = parser.parse_args()

    # Validate arguments
    if args.gpu >= args.total_gpus:
        print(f"Error: --gpu {args.gpu} must be less than --total_gpus {args.total_gpus}")
        print(f"Example: For 3 GPUs, use --gpu 0 --total_gpus 3, --gpu 1 --total_gpus 3, etc.")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate experiments
    experiments = generate_experiments(SEARCH_SPACE)

    # Filter experiments for this GPU (round-robin distribution)
    my_experiments = [exp for exp in experiments if exp['exp_id'] % args.total_gpus == args.gpu]

    # Limit number of experiments
    if args.quick_test:
        my_experiments = my_experiments[:3]
        print("Quick test mode: running 3 experiments")
    else:
        my_experiments = my_experiments[:args.max_experiments]

    print(f"GPU {args.gpu}: Running {len(my_experiments)} experiments")

    # Save experiment config
    config_file = output_dir / f"experiments_gpu{args.gpu}.json"
    with open(config_file, 'w') as f:
        json.dump({
            "gpu_id": args.gpu,
            "total_gpus": args.total_gpus,
            "experiments": my_experiments
        }, f, indent=2)

    # Run experiments
    results = []
    start_time = time.time()

    for i, exp in enumerate(my_experiments):
        print(f"\n[{i+1}/{len(my_experiments)}] Experiment {exp['exp_id']}")
        success = run_experiment(exp, output_dir, args.gpu, args.physical_gpu, args.conda_env)
        results.append(exp)

        # Save intermediate results
        with open(output_dir / f"results_gpu{args.gpu}_partial.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Save final results
    elapsed = time.time() - start_time
    with open(output_dir / f"results_gpu{args.gpu}_final.json", 'w') as f:
        json.dump({
            "gpu_id": args.gpu,
            "elapsed_time": elapsed,
            "total_experiments": len(my_experiments),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "experiments": results
        }, f, indent=2)

    print(f"\nGPU {args.gpu} completed in {elapsed/60:.1f} minutes")
    print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}/{len(my_experiments)}")


if __name__ == "__main__":
    main()
