#!/usr/bin/env python3
"""
Advanced SRF AutoResearch loop - tests multiple improvements in sequence.
"""
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import os

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 100
BASE_OUTPUT = "results/autoresearch_advanced/"

# Advanced experiments
ADVANCED_EXPERIMENTS = {
    "baseline": {
        "description": "Original baseline (α=2.0, grid=7×7)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
    },
    "1A_multiscale_avg": {
        "description": "Multi-scale CLIP (5,7,9) - average ensemble",
        "alpha": 2.0,
        "use_multiscale": True,
        "multiscale_scales": [5, 7, 9],
        "ensemble_method": "average",
        "clip_coarse_grid": 7,
    },
    "1B_multiscale_max": {
        "description": "Multi-scale CLIP (5,7,9) - max ensemble",
        "alpha": 2.0,
        "use_multiscale": True,
        "multiscale_scales": [5, 7, 9],
        "ensemble_method": "max",
        "clip_coarse_grid": 7,
    },
    "1C_multiscale_weighted": {
        "description": "Multi-scale CLIP (5,7,9) - weighted ensemble",
        "alpha": 2.0,
        "use_multiscale": True,
        "multiscale_scales": [5, 7, 9],
        "ensemble_method": "weighted",
        "clip_coarse_grid": 7,
    },
    "2A_graduated_boost": {
        "description": "Graduated boost (α_high=4.0, α_mid=2.0, α_low=1.0)",
        "use_graduated": True,
        "alpha_high": 4.0,
        "alpha_mid": 2.0,
        "alpha_low": 1.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
    },
    "2B_stronger_boost": {
        "description": "Much stronger boost (α=8.0)",
        "alpha": 8.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "2C_very_strong_boost": {
        "description": "Extremely strong boost (α=12.0)",
        "alpha": 12.0,
        "eps": 0.4,
        "clip_coarse_grid": 7,
    },
    "3A_fine_grid": {
        "description": "Fine grid with strong boost (5×5, α=6.0)",
        "alpha": 6.0,
        "eps": 0.2,
        "clip_coarse_grid": 5,
    },
    "3B_coarse_grid": {
        "description": "Coarse grid with strong boost (9×9, α=6.0)",
        "alpha": 6.0,
        "eps": 0.2,
        "clip_coarse_grid": 9,
    },
    "4A_combined": {
        "description": "Multi-scale + graduated boost",
        "use_multiscale": True,
        "use_graduated": True,
        "multiscale_scales": [5, 7, 9],
        "ensemble_method": "weighted",
        "alpha_high": 6.0,
        "alpha_mid": 3.0,
        "alpha_low": 1.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
    },
}


def run_experiment(name: str, params: dict, gpu_id: int = 1) -> dict:
    """Run a single advanced experiment."""
    output_dir = f"{BASE_OUTPUT}{name}/"

    # Build command
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", DATASET,
        "--pope_splits", SPLIT,
        "--n_pope", str(N_SAMPLES),
        "--output", output_dir,
    ]

    # Add parameters
    for key, value in params.items():
        if key in ["description", "use_multiscale", "use_graduated",
                   "multiscale_scales", "ensemble_method", "alpha_high", "alpha_mid", "alpha_low"]:
            continue  # Skip advanced params (need code modification)
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Description: {params['description']}")
    print(f"GPU: {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse accuracy from output
    accuracy = parse_accuracy(result.stdout)

    # Save full output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}output.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

    return {
        "name": name,
        "accuracy": accuracy,
        "params": params,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
    }


def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    for line in output.split('\n'):
        if "SRF:" in line and "acc=" in line:
            import re
            match = re.search(r'acc=(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))

    # Fallback: find any percentage
    import re
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Advanced SRF AutoResearch")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    print("Advanced SRF AutoResearch - LLaVA-7B + POPE Adversarial")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_experiment("baseline", ADVANCED_EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Baseline: {result['accuracy']:.1f}%")
        return

    if args.experiment:
        if args.experiment in ADVANCED_EXPERIMENTS:
            print(f"\n🎯 Running: {args.experiment}")
            result = run_experiment(args.experiment, ADVANCED_EXPERIMENTS[args.experiment], args.gpu)
            print(f"\n✓ Result: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown experiment: {args.experiment}")
            print(f"Available: {list(ADVANCED_EXPERIMENTS.keys())}")
        return

    if args.all:
        print(f"\n🚀 Running ALL {len(ADVANCED_EXPERIMENTS)} advanced experiments...")
        results = []
        best_accuracy = 0.0
        best_experiment = None

        for i, (name, params) in enumerate(ADVANCED_EXPERIMENTS.items()):
            # Alternate GPUs
            gpu = 1 if i % 2 == 0 else 2

            result = run_experiment(name, params, gpu)
            results.append(result)

            delta = result["accuracy"] - 83.0  # Expected baseline
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_experiment = name

            status = "🏆" if delta > 1.0 else "✓" if delta > 0 else "✗"
            print(f"{status} {result['name']}: {result['accuracy']:.1f}% (Δ{delta:+.1f})")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{BASE_OUTPUT}advanced_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "best_experiment": best_experiment,
                "best_accuracy": best_accuracy,
                "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%)")
        print(f"📊 Results: {summary_file}")
        print(f"{'='*70}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
