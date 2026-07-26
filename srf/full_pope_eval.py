#!/usr/bin/env python3
"""
Full POPE dataset evaluation - LLaVA-7B on complete POPE.
Tests baseline and best SRF configs on all 3 splits with full dataset.
"""
import subprocess
import os
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/full_pope_evaluation/"

# Best configs from our experiments
BEST_CONFIGS = {
    "baseline": {
        "description": "Original baseline (no SRF intervention)",
        "use_srf": False,
    },
    "higher_topk": {
        "description": "Higher top-k 50% (86% on 50 samples)",
        "use_srf": True,
        "clip_top_k_pct": 0.5,
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
    },
    "strong_boost": {
        "description": "Strong boost α=4.0 (86% on 50 samples)",
        "use_srf": True,
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
    },
    "no_absence_aware": {
        "description": "No absence-aware (84% on 100 samples)",
        "use_srf": True,
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
}

POPE_SPLITS = ["random", "popular", "adversarial"]


def run_full_pope_eval(
    config_name: str,
    config_params: dict,
    split: str,
    gpu_id: int = 1
) -> dict:
    """Run evaluation on full POPE split."""
    output_dir = f"{BASE_OUTPUT}{config_name}/{split}/"

    # Build command
    if config_params["use_srf"]:
        cmd = [
            "python", "srf/eval.py",
            "--method", "srf",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", split,
            "--output", output_dir,
        ]

        # Add SRF parameters
        for key, value in config_params.items():
            if key in ["description", "use_srf"]:
                continue
            cmd.extend([f"--{key}", str(value)])
    else:
        cmd = [
            "python", "srf/eval.py",
            "--method", "baseline",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", split,
            "--output", output_dir,
        ]

    print(f"\n{'='*70}")
    print(f"🧪 Running: {config_name} on {split.upper()} (FULL DATASET)")
    print(f"Description: {config_params['description']}")
    print(f"GPU: {gpu_id}")
    print(f"Output: {output_dir}")
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
        "samples": "FULL DATASET",  # Marker that we used full dataset
    }


def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    for line in output.split('\n'):
        if "accuracy" in line.lower() and "baseline" in line.lower():
            import re
            match = re.search(r'acc(?:uracy)?[:\s]*(\d+\.?\d*)', line)
            if match:
                return float(match.group(1))

    # Try POPE-specific format
    for line in output.split('\n'):
        if "POPE baseline:" in line or "POPE SRF:" in line:
            import re
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))

    # General percentage search
    import re
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full POPE Dataset Evaluation")
    parser.add_argument("--config", type=str, help="Run specific config")
    parser.add_argument("--split", type=str, help="Run specific split")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all configs on all splits")
    args = parser.parse_args()

    print("🧪 Full POPE Dataset Evaluation - LLaVA-7B")
    print(f"Model: {MODEL}")
    print(f"Dataset: POPE (FULL - all samples per split)")
    print(f"Splits: {', '.join(POPE_SPLITS).upper()}")
    print(f"Configs: {len(BEST_CONFIGS)}")

    if args.config and args.split:
        # Run specific config on specific split
        if args.config in BEST_CONFIGS:
            result = run_full_pope_eval(args.config, BEST_CONFIGS[args.config], args.split, args.gpu)
            print(f"\n✅ {args.config} on {args.split}: {result['accuracy']:.1f}%")
        return

    if args.config:
        # Run specific config on all splits
        if args.config not in BEST_CONFIGS:
            print(f"❌ Unknown config: {args.config}")
            return

        print(f"\n🚀 Running {args.config} on ALL splits (FULL DATASET)...")
        results = []
        for split in POPE_SPLITS:
            result = run_full_pope_eval(args.config, BEST_CONFIGS[args.config], split, args.gpu)
            results.append(result)

        avg_acc = sum(r["accuracy"] for r in results) / len(results)
        print(f"\n{'='*70}")
        print(f"📊 Results for {args.config}:")
        for r in results:
            print(f"   {r['split'].upper()}: {r['accuracy']:.1f}%")
        print(f"   AVERAGE: {avg_acc:.1f}%")
        print(f"{'='*70}")
        return

    if args.all:
        print(f"\n🚀 Running ALL configs on ALL splits (FULL DATASET)...")
        print(f"Total experiments: {len(BEST_CONFIGS)} × 3 splits = {len(BEST_CONFIGS) * 3}")
        print(f"Estimated time: ~2-3 hours (full POPE is large)")

        all_results = {}
        for config_name in BEST_CONFIGS:
            print(f"\n{'='*70}")
            print(f"🧪 Testing config: {config_name}")
            print(f"Description: {BEST_CONFIGS[config_name]['description']}")
            print(f"{'='*70}")

            config_results = []
            for split in POPE_SPLITS:
                result = run_full_pove_eval(config_name, BEST_CONFIGS[config_name], split, args.gpu)
                config_results.append(result)

            all_results[config_name] = config_results

            # Average for this config
            avg_acc = sum(r["accuracy"] for r in config_results) / len(config_results)
            print(f"📊 {config_name} average: {avg_acc:.1f}%")

        # Find best overall
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
        summary_file = f"{BASE_OUTPUT}full_pope_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "model": MODEL,
                "dataset": "POPE_FULL",
                "best_overall": best_overall,
                "best_average": best_avg,
                "all_results": all_results,
            }, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🏆 FULL POPE RESULTS:")
        print(f"{'='*70}")
        print(f"\nBest overall config: {best_overall}")
        print(f"Best average accuracy: {best_avg:.1f}%")

        print(f"\n📊 Detailed breakdown:")
        for config_name, results_list in all_results.items():
            avg_acc = sum(r["accuracy"] for r in results_list) / len(results_list)
            print(f"\n{config_name} ({BEST_CONFIGS[config_name]['description']}):")
            for r in results_list:
                print(f"  {r['split'].upper()}: {r['accuracy']:.1f}%")
            print(f"  AVERAGE: {avg_acc:.1f}%")

        print(f"\n{'='*70}")
        print(f"📊 Results saved: {summary_file}")
        print(f"{'='*70}")

        # Compare baseline vs best SRF
        baseline_results = all_results.get("baseline", [])
        if baseline_results:
            baseline_avg = sum(r["accuracy"] for r in baseline_results) / len(baseline_results)
            improvement = best_avg - baseline_avg

            print(f"\n🎯 FINAL COMPARISON:")
            print(f"   Baseline average: {baseline_avg:.1f}%")
            print(f"   Best SRF average: {best_avg:.1f}%")
            print(f"   Improvement: {improvement:+.1f}%")

            if improvement > 1.0:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT!")
            elif improvement > 0:
                print(f"   ✓ Modest improvement")
            else:
                print(f"   ⚠️  No improvement - SRF may not help on full dataset")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
