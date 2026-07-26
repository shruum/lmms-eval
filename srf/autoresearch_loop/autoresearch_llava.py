#!/usr/bin/env python3
"""
AutoResearch loop for SRF improvements on LLaVA-7B + POPE.
Tests each variation, logs results, keeps winners.

Usage:
    python srf/autoresearch_llava.py --baseline-only    # Run baseline first
    python srf/autoresearch_llava.py --option 1A       # Run specific experiment
    python srf/autoresearch_llava.py --all             # Run all experiments
"""
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 100  # Quick tests for autoresearch
BASE_OUTPUT = "results/autoresearch_llava/"

# Experimental variations
EXPERIMENTS = {
    "baseline": {
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
    },
    # Option 1: Better Salience
    "1A_multiscale_5x5": {
        "alpha": 2.0,
        "clip_coarse_grid": 5,
        "clip_top_k_pct": 0.30,
    },
    "1B_multiscale_9x9": {
        "alpha": 2.0,
        "clip_coarse_grid": 9,
        "clip_top_k_pct": 0.30,
    },
    # Option 2: Graduated Boost (will need code changes)
    "2A_stronger_boost": {
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
    },
    "2B_gentle_boost": {
        "alpha": 1.5,
        "eps": 0.1,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
    },
}

def run_experiment(name: str, params: dict, gpu_id: int = 1) -> dict:
    """Run single experiment, return results."""
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

    # Add params
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"GPU: {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    # Set GPU
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse accuracy from output
    accuracy = parse_accuracy(result.stdout)

    # Save full output
    output_file = f"{output_dir}output.txt"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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
        if "accuracy" in line.lower():
            import re
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))

    # Try to find "XX.XX%" pattern anywhere
    import re
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])  # Last occurrence is likely final

    return 0.0

def main():
    parser = argparse.ArgumentParser(description="SRF AutoResearch Loop")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline")
    parser.add_argument("--option", type=str, help="Run specific experiment (e.g., 1A)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID to use")
    args = parser.parse_args()

    print("SRF AutoResearch - LLaVA-7B + POPE Adversarial")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_experiment("baseline", EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Baseline: {result['accuracy']:.1f}%")
        return

    if args.option:
        if args.option in EXPERIMENTS:
            print(f"\n🎯 Running experiment: {args.option}")
            result = run_experiment(args.option, EXPERIMENTS[args.option], args.gpu)
            print(f"\n✓ Result: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown experiment: {args.option}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
        return

    if args.all:
        print(f"\n🚀 Running ALL {len(EXPERIMENTS)} experiments...")
        results = []
        best_accuracy = 0.0
        best_experiment = None

        for name, params in EXPERIMENTS.items():
            result = run_experiment(name, params, args.gpu)
            results.append(result)

            # Track best
            delta = result["accuracy"] - 84.0  # Expected baseline
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_experiment = name

            status = "🏆" if delta > 1.0 else "✓" if delta > 0 else "✗"
            print(f"{status} {result['name']}: {result['accuracy']:.1f}% (Δ{delta:+.1f})")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{BASE_OUTPUT}summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "best_experiment": best_experiment,
                "best_accuracy": best_accuracy,
                "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%)")
        print(f"📊 Results: {summary_file}")
        print(f"{'='*60}")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
