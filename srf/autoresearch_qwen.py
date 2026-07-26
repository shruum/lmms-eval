#!/usr/bin/env python3
"""
Qwen-VL-Chat AutoResearch loop - tests SRF improvements on Qwen-VL.
Runs on GPU:0 while LLaVA experiments run on GPUs 1,2.
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration for Qwen-VL (original model from arxiv paper)
MODEL = "Qwen/Qwen-VL-Chat"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 100
BASE_OUTPUT = "results/autoresearch_qwen/"

# Qwen-VL experiments (same configs that worked for LLaVA)
QWEN_EXPERIMENTS = {
    "baseline": {
        "description": "Qwen-VL baseline (α=2.0)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
    },
    "no_absence_aware": {
        "description": "No absence-aware (+1% on LLaVA)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    "fine_grid": {
        "description": "Fine grid (+1% on LLaVA)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 5,
    },
    "strong_boost": {
        "description": "Strong boost (α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "coarse_grid": {
        "description": "Coarse grid (+1% on LLaVA)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 9,
    },
}


def run_qwen_experiment(name: str, params: dict, gpu_id: int = 0) -> dict:
    """Run Qwen-VL experiment."""
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
    print(f"🤖 Running Qwen-VL: {name}")
    print(f"Description: {params['description']}")
    print(f"GPU: {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse accuracy
    accuracy = parse_accuracy(result.stdout)

    # Save output
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

    import re
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen-VL AutoResearch")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print("🤖 Qwen-VL-Chat AutoResearch - POPE Adversarial")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")
    print(f"Total experiments: {len(QWEN_EXPERIMENTS)}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_qwen_experiment("baseline", QWEN_EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Qwen-VL Baseline: {result['accuracy']:.1f}%")
        return

    if args.all:
        print(f"\n🚀 Running ALL {len(QWEN_EXPERIMENTS)} Qwen-VL experiments...")
        results = []
        best_accuracy = 0.0
        best_experiment = None
        baseline_accuracy = 0.0

        for name, params in QWEN_EXPERIMENTS.items():
            result = run_qwen_experiment(name, params, args.gpu)
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
        summary_file = f"{BASE_OUTPUT}qwen_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "model": MODEL,
                "baseline": baseline_accuracy,
                "best_experiment": best_experiment,
                "best_accuracy": best_accuracy,
                "best_delta": best_accuracy - baseline_accuracy,
                "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🤖 Qwen-VL Results:")
        print(f"🏆 BASELINE: {baseline_accuracy:.1f}%")
        print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%, Δ{best_accuracy - baseline_accuracy:+.1f})")
        print(f"📊 Results: {summary_file}")
        print(f"{'='*70}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
