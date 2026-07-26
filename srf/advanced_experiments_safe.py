#!/usr/bin/env python3
"""
Safe advanced SRF experiments - runs different parameter configurations
without modifying base code.
"""
import subprocess
import os
from datetime import datetime

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 100
BASE_OUTPUT = "results/autoresearch_advanced/"

# Advanced parameter experiments (safe - just CLI params)
SAFE_EXPERIMENTS = {
    "baseline": {
        "description": "Original baseline (α=2.0, grid=7×7)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
    },
    "1_strong_boost_4": {
        "description": "Stronger boost (α=4.0, eps=0.2)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
    },
    "2_strong_boost_6": {
        "description": "Stronger boost (α=6.0, eps=0.3)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "3_very_strong_boost_8": {
        "description": "Very strong boost (α=8.0, eps=0.4)",
        "alpha": 8.0,
        "eps": 0.4,
        "clip_coarse_grid": 7,
    },
    "4_extreme_boost_12": {
        "description": "Extreme boost (α=12.0, eps=0.5)",
        "alpha": 12.0,
        "eps": 0.5,
        "clip_coarse_grid": 7,
    },
    "5_fine_grid_boost": {
        "description": "Fine grid with strong boost (5×5, α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 5,
    },
    "6_coarse_grid_boost": {
        "description": "Coarse grid with strong boost (9×9, α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 9,
    },
    "7_no_absence_aware": {
        "description": "No absence-aware (α=6.0, thresh=0.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    "8_higher_topk": {
        "description": "Higher top-k percentage (50% instead of 30%)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.5,
    },
}


def run_experiment(name: str, params: dict, gpu_id: int = 1) -> dict:
    """Run a single experiment."""
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
        if key == "description":
            continue
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
    from pathlib import Path
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
        "params": params["description"],
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
    import argparse
    parser = argparse.ArgumentParser(description="Safe Advanced SRF Experiments")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    print("Safe Advanced SRF Experiments - LLaVA-7B + POPE Adversarial")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_experiment("baseline", SAFE_EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Baseline: {result['accuracy']:.1f}%")
        return

    if args.experiment:
        if args.experiment in SAFE_EXPERIMENTS:
            print(f"\n🎯 Running: {args.experiment}")
            result = run_experiment(args.experiment, SAFE_EXPERIMENTS[args.experiment], args.gpu)
            print(f"\n✓ Result: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown experiment: {args.experiment}")
            print(f"Available: {list(SAFE_EXPERIMENTS.keys())}")
        return

    if args.all:
        print(f"\n🚀 Running ALL {len(SAFE_EXPERIMENTS)} safe experiments...")
        results = []
        best_accuracy = 0.0
        best_experiment = None
        baseline_accuracy = 0.0

        for i, (name, params) in enumerate(SAFE_EXPERIMENTS.items()):
            # Alternate GPUs
            gpu = 1 if i % 2 == 0 else 2

            result = run_experiment(name, params, gpu)
            results.append(result)

            if name == "baseline":
                baseline_accuracy = result["accuracy"]

            delta = result["accuracy"] - baseline_accuracy
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_experiment = name

            status = "🏆" if delta > 1.0 else "✓" if delta > 0 else "✗"
            print(f"{status} {result['name']}: {result['accuracy']:.1f}% (Δ{delta:+.1f})")

        # Save results
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{BASE_OUTPUT}safe_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "baseline": baseline_accuracy,
                "best_experiment": best_experiment,
                "best_accuracy": best_accuracy,
                "best_delta": best_accuracy - baseline_accuracy,
                "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🏆 BASELINE: {baseline_accuracy:.1f}%")
        print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%, Δ{best_accuracy - baseline_accuracy:+.1f})")
        print(f"📊 Results: {summary_file}")
        print(f"{'='*70}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
