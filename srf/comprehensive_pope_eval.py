#!/usr/bin/env python3
"""
Comprehensive POPE evaluation - tests winning configs on all 3 splits.
Tests on random, popular, and adversarial POPE splits.
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
N_SAMPLES = 500  # Larger sample for robust evaluation
BASE_OUTPUT = "results/comprehensive_pope/"

# Winning configurations from our experiments
WINNING_CONFIGS = {
    "no_absence_aware": {
        "description": "No absence-aware (84% on adversarial)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    "fine_grid": {
        "description": "Fine grid (84% on adversarial)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 5,
    },
    "strong_boost": {
        "description": "Strong boost (84% on adversarial)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "coarse_grid": {
        "description": "Coarse grid (84% on adversarial)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 9,
    },
}

POPE_SPLITS = ["random", "popular", "adversarial"]


def run_comprehensive_eval(
    config_name: str,
    config_params: dict,
    split: str,
    gpu_id: int = 1
) -> dict:
    """Run evaluation on specific POPE split."""
    output_dir = f"{BASE_OUTPUT}{config_name}/{split}/"

    # Build command
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "pope",
        "--pope_splits", split,
        "--n_pope", str(N_SAMPLES),
        "--output", output_dir,
    ]

    # Add parameters
    for key, value in config_params.items():
        if key == "description":
            continue
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*70}")
    print(f"🧪 Running: {config_name} on {split.upper()} split")
    print(f"GPU: {gpu_id} | Samples: {N_SAMPLES}")
    print(f"Config: {config_params['description']}")
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

    print(f"✅ Result: {accuracy:.1f}%")

    return {
        "config": config_name,
        "split": split,
        "accuracy": accuracy,
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
    parser = argparse.ArgumentParser(description="Comprehensive POPE Evaluation")
    parser.add_argument("--config", type=str, help="Run specific config only")
    parser.add_argument("--split", type=str, help="Run specific split only")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    print("🧪 Comprehensive POPE Evaluation - All 3 Splits")
    print(f"Model: {MODEL}")
    print(f"Samples per split: {N_SAMPLES}")
    print(f"GPU: {args.gpu}")
    print(f"Configs: {len(WINNING_CONFIGS)}")
    print(f"Splits: {', '.join(POPE_SPLITS).upper()}")

    if args.config and args.split:
        # Run specific config on specific split
        if args.config in WINNING_CONFIGS:
            result = run_comprehensive_eval(args.config, WINNING_CONFIGS[args.config], args.split, args.gpu)
            print(f"\n✅ {args.config} on {args.split}: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown config: {args.config}")
        return

    if args.config:
        # Run specific config on all splits
        if args.config not in WINNING_CONFIGS:
            print(f"❌ Unknown config: {args.config}")
            return

        print(f"\n🚀 Running {args.config} on ALL splits...")
        results = []
        for split in POPE_SPLITS:
            result = run_comprehensive_eval(args.config, WINNING_CONFIGS[args.config], split, args.gpu)
            results.append(result)

        # Average across splits
        avg_acc = sum(r["accuracy"] for r in results) / len(results)
        print(f"\n{'='*70}")
        print(f"📊 Results for {args.config}:")
        for r in results:
            print(f"   {r['split'].upper()}: {r['accuracy']:.1f}%")
        print(f"   AVERAGE: {avg_acc:.1f}%")
        print(f"{'='*70}")
        return

    # Run all configs on all splits
    print(f"\n🚀 Running ALL configs on ALL splits ({len(WINNING_CONFIGS)} × 3 = {len(WINNING_CONFIGS) * 3} experiments)...")

    all_results = {}
    for config_name in WINNING_CONFIGS:
        print(f"\n{'='*70}")
        print(f"🧪 Testing config: {config_name}")
        print(f"{'='*70}")

        config_results = []
        for split in POPE_SPLITS:
            result = run_comprehensive_eval(config_name, WINNING_CONFIGS[config_name], split, args.gpu)
            config_results.append(result)

        all_results[config_name] = config_results

        # Average for this config
        avg_acc = sum(r["accuracy"] for r in config_results) / len(config_results)
        print(f"📊 {config_name} average: {avg_acc:.1f}%")

    # Find best overall config
    best_overall = None
    best_avg = 0.0

    for config_name, results_list in all_results.items():
        avg_acc = sum(r["accuracy"] for r in results_list) / len(results_list)
        if avg_acc > best_avg:
            best_avg = avg_acc
            best_overall = config_name

    # Save comprehensive results
    import json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"{BASE_OUTPUT}comprehensive_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": MODEL,
            "n_samples_per_split": N_SAMPLES,
            "best_overall": best_overall,
            "best_average": best_avg,
            "all_results": all_results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏆 COMPREHENSIVE RESULTS:")
    print(f"{'='*70}")
    print(f"\nBest overall config: {best_overall}")
    print(f"Best average accuracy: {best_avg:.1f}%")

    print(f"\n📊 Detailed breakdown:")
    for config_name, results_list in all_results.items():
        avg_acc = sum(r["accuracy"] for r in results_list) / len(results_list)
        print(f"\n{config_name}:")
        for r in results_list:
            print(f"  {r['split'].upper()}: {r['accuracy']:.1f}%")
        print(f"  AVERAGE: {avg_acc:.1f}%")

    print(f"\n{'='*70}")
    print(f"📊 Results saved: {summary_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()