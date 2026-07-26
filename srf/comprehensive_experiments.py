#!/usr/bin/env python3
"""
Comprehensive advanced SRF experiments - tests all improvements systematically.
Includes: bigger CLIP, hidden layers, advanced attention strategies.
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 50  # Use 50 samples for faster iteration
BASE_OUTPUT = "results/autoresearch_comprehensive/"

# Comprehensive experiments
COMPREHENSIVE_EXPERIMENTS = {
    # === BASELINE ===
    "baseline": {
        "description": "Original baseline (α=2.0, ViT-B/32)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
    },

    # === PARAMETER VARIATIONS ===
    "strong_boost_4": {
        "description": "Strong boost (α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
    },
    "strong_boost_6": {
        "description": "Stronger boost (α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "strong_boost_8": {
        "description": "Very strong boost (α=8.0)",
        "alpha": 8.0,
        "eps": 0.4,
        "clip_coarse_grid": 7,
    },

    # === GRID VARIATIONS ===
    "fine_grid": {
        "description": "Fine grid (5×5, α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 5,
    },
    "coarse_grid": {
        "description": "Coarse grid (9×9, α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 9,
    },

    # === ABSENCE-AWARE VARIATIONS ===
    "no_absence_aware": {
        "description": "No absence-aware (thresh=0.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    "strict_absence": {
        "description": "Strict absence-aware (thresh=0.3)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.3,
    },

    # === TOP-K VARIATIONS ===
    "higher_topk": {
        "description": "Higher top-k (50%)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.5,
    },
    "lower_topk": {
        "description": "Lower top-k (20%)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.2,
    },

    # === COMBINATIONS ===
    "combo_fine_strong": {
        "description": "Fine grid + strong boost (5×5, α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 5,
    },
    "combo_coarse_strong": {
        "description": "Coarse grid + strong boost (9×9, α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 9,
    },
    "combo_high_topk_strong": {
        "description": "High top-k + strong boost (50%, α=6.0)",
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
    parser = argparse.ArgumentParser(description="Comprehensive Advanced SRF Experiments")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick subset (6 experiments)")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    print("Comprehensive Advanced SRF Experiments")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")
    print(f"Total experiments: {len(COMPREHENSIVE_EXPERIMENTS)}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_experiment("baseline", COMPREHENSIVE_EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Baseline: {result['accuracy']:.1f}%")
        return

    if args.experiment:
        if args.experiment in COMPREHENSIVE_EXPERIMENTS:
            print(f"\n🎯 Running: {args.experiment}")
            result = run_experiment(args.experiment, COMPREHENSIVE_EXPERIMENTS[args.experiment], args.gpu)
            print(f"\n✓ Result: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown experiment: {args.experiment}")
            print(f"Available: {list(COMPREHENSIVE_EXPERIMENTS.keys())}")
        return

    # Select experiments to run
    if args.quick:
        # Quick subset: baseline + 5 key variations
        experiments_to_run = {
            "baseline": COMPREHENSIVE_EXPERIMENTS["baseline"],
            "strong_boost_4": COMPREHENSIVE_EXPERIMENTS["strong_boost_4"],
            "strong_boost_6": COMPREHENSIVE_EXPERIMENTS["strong_boost_6"],
            "fine_grid": COMPREHENSIVE_EXPERIMENTS["fine_grid"],
            "no_absence_aware": COMPREHENSIVE_EXPERIMENTS["no_absence_aware"],
            "higher_topk": COMPREHENSIVE_EXPERIMENTS["higher_topk"],
        }
        print(f"\n🚀 Running QUICK subset ({len(experiments_to_run)} experiments)...")
    else:
        experiments_to_run = COMPREHENSIVE_EXPERIMENTS
        print(f"\n🚀 Running ALL {len(experiments_to_run)} experiments...")

    results = []
    best_accuracy = 0.0
    best_experiment = None
    baseline_accuracy = 0.0

    for i, (name, params) in enumerate(experiments_to_run.items()):
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
    summary_file = f"{BASE_OUTPUT}comprehensive_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "baseline": baseline_accuracy,
            "best_experiment": best_experiment,
            "best_accuracy": best_accuracy,
            "best_delta": best_accuracy - baseline_accuracy,
            "n_samples": N_SAMPLES,
            "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏆 BASELINE: {baseline_accuracy:.1f}%")
    print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%, Δ{best_accuracy - baseline_accuracy:+.1f})")
    print(f"📊 Results: {summary_file}")
    print(f"{'='*70}")

    # Print recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if best_accuracy - baseline_accuracy > 1.0:
        print(f"  ✅ Found improvement! Use: {best_experiment}")
        print(f"  → {COMPREHENSIVE_EXPERIMENTS[best_experiment]['description']}")
    else:
        print(f"  ⚠️  No significant improvement found.")
        print(f"  → Consider: (1) Larger CLIP model, (2) Hidden layer features, (3) Different dataset")

    return


if __name__ == "__main__":
    main()
