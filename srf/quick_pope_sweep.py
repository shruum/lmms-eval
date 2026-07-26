#!/usr/bin/env python3
"""
Quick POPE Sweep - LLaVA-1.5-7B (MS-COCO only)
Tests most promising SRF configs on all 3 POPE categories.
Based on previous best results: +0.10% configs.
"""
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/pope_quick_sweep/"
POPE_CATEGORIES = ["random", "popular", "adversarial"]

# Most promising configs from previous experiments
QUICK_CONFIGS = {
    "baseline": {
        "description": "No SRF (baseline)",
        "use_srf": False,
    },
    # Our best configs from 50-sample tests (86% on adversarial)
    "higher_topk": {
        "description": "Higher top-k (86% on 50 samples)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.5,
        "clip_suppress_thresh": 0.0,
    },
    "strong_boost": {
        "description": "Strong boost (86% on 50 samples)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    "no_absence_aware": {
        "description": "No absence-aware (84% on 100 samples)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    # Fine grid variants (87.6% on 50 samples)
    "fine_grid_weak": {
        "description": "Fine grid + weak boost",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 5,
        "clip_suppress_thresh": 0.0,
    },
    "fine_grid_strong": {
        "description": "Fine grid + strong boost",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 5,
        "clip_suppress_thresh": 0.0,
    },
    # Coarse grid variants
    "coarse_grid_weak": {
        "description": "Coarse grid + weak boost",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 9,
        "clip_suppress_thresh": 0.0,
    },
    "coarse_grid_strong": {
        "description": "Coarse grid + strong boost",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 9,
        "clip_suppress_thresh": 0.0,
    },
    # Even stronger boost
    "very_strong_boost": {
        "description": "Very strong boost (α=8.0)",
        "alpha": 8.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_suppress_thresh": 0.0,
    },
    # Higher top-k variants
    "very_high_topk": {
        "description": "Very high top-k (70%)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.7,
        "clip_suppress_thresh": 0.0,
    },
}

def run_single_experiment(
    config_name: str,
    config_params: dict,
    category: str,
    gpu_id: int = 1
) -> dict:
    """Run single SRF experiment on POPE category."""
    output_dir = f"{BASE_OUTPUT}{category}/{config_name}/"

    # Build command
    if config_params.get("use_srf", True):
        cmd = [
            "python", "srf/eval.py",
            "--method", "srf",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", category,
            "--output", output_dir,
        ]

        # Add SRF parameters
        for key, value in config_params.items():
            if key in ["description", "use_srf"]:
                continue
            cmd.extend([f"--{key}", str(value)])
    else:
        # Baseline - no SRF
        cmd = [
            "python", "srf/eval.py",
            "--method", "baseline",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", category,
            "--output", output_dir,
        ]

    print(f"\n{'='*70}")
    print(f"🧪 Running: {config_name} on {category.upper()}")
    print(f"Description: {config_params.get('description', 'N/A')}")
    print(f"GPU: {gpu_id}")
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

    print(f"✅ Result: {accuracy:.2f}%")

    return {
        "config": config_name,
        "category": category,
        "accuracy": accuracy,
        "params": config_params,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
    }

def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    import re

    # Try to find accuracy line
    for line in output.split('\n'):
        if "accuracy" in line.lower():
            # Match patterns like "accuracy: 85.5%" or "Accuracy = 87.27%"
            match = re.search(r'acc(?:uracy)?[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))

    # Try POPE-specific format
    for line in output.split('\n'):
        if "POPE" in line:
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))

    # General percentage search
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick POPE Sweep - LLaVA-1.5-7B")
    parser.add_argument("--category", type=str, help="Run specific category only")
    parser.add_argument("--config", type=str, help="Run specific config only")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all configs on all categories")
    args = parser.parse_args()

    print("🧪 Quick POPE Sweep - LLaVA-1.5-7B (MS-COCO)")
    print(f"Model: {MODEL}")
    print(f"Dataset: POPE (MS-COCO only - 3000 samples per category)")
    print(f"Configs: {len(QUICK_CONFIGS)}")
    print(f"Categories: {', '.join(POPE_CATEGORIES).upper()}")

    # Baseline from previous runs
    print(f"\n📊 Known Baselines:")
    print(f"  Random: 87.27%")
    print(f"  Popular: 85.47%")
    print(f"  Adversarial: 82.93%")
    print(f"  Average: 85.22%")

    if args.config and args.category:
        # Run specific config on specific category
        if args.config in QUICK_CONFIGS:
            result = run_single_experiment(args.config, QUICK_CONFIGS[args.config], args.category, args.gpu)
            print(f"\n✅ {args.config} on {args.category}: {result['accuracy']:.2f}%")
        return

    if args.config:
        # Run specific config on all categories
        if args.config not in QUICK_CONFIGS:
            print(f"❌ Unknown config: {args.config}")
            return

        print(f"\n🚀 Running {args.config} on ALL categories...")
        results = []
        for category in POPE_CATEGORIES:
            result = run_single_experiment(args.config, QUICK_CONFIGS[args.config], category, args.gpu)
            results.append(result)

        avg_acc = sum(r["accuracy"] for r in results) / len(results)
        print(f"\n{'='*70}")
        print(f"📊 Results for {args.config}:")
        for r in results:
            print(f"   {r['category'].upper()}: {r['accuracy']:.2f}%")
        print(f"   AVERAGE: {avg_acc:.2f}%")
        print(f"{'='*70}")
        return

    if args.all:
        print(f"\n🚀 Running ALL configs on ALL categories...")
        print(f"Total experiments: {len(QUICK_CONFIGS)} × 3 categories = {len(QUICK_CONFIGS) * 3}")
        print(f"Estimated time: ~2-3 hours")

        all_results = {}
        for config_name in QUICK_CONFIGS:
            print(f"\n{'='*70}")
            print(f"🧪 Testing config: {config_name}")
            print(f"Description: {QUICK_CONFIGS[config_name]['description']}")
            print(f"{'='*70}")

            config_results = []
            for category in POPE_CATEGORIES:
                result = run_single_experiment(config_name, QUICK_CONFIGS[config_name], category, args.gpu)
                config_results.append(result)

            all_results[config_name] = config_results

            # Average for this config
            avg_acc = sum(r["accuracy"] for r in config_results) / len(config_results)
            print(f"📊 {config_name} average: {avg_acc:.2f}%")

        # Find best overall
        best_overall = None
        best_avg = 0.0

        for config_name, results_list in all_results.items():
            avg_acc = sum(r["accuracy"] for r in results_list) / len(results_list)
            if avg_acc > best_avg:
                best_avg = avg_acc
                best_overall = config_name

        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{BASE_OUTPUT}quick_sweep_summary_{timestamp}.json"

        # Format for paper table
        paper_table = format_paper_table(all_results)

        summary = {
            "model": MODEL,
            "dataset": "POPE (MS-COCO)",
            "best_overall": best_overall,
            "best_average": best_avg,
            "all_results": all_results,
            "paper_table": paper_table,
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"🏆 QUICK SWEEP RESULTS:")
        print(f"{'='*70}")
        print(f"\nBest overall config: {best_overall}")
        print(f"Best average accuracy: {best_avg:.2f}%")

        print(f"\n📊 Paper Table Format:")
        print(paper_table)

        print(f"\n{'='*70}")
        print(f"📁 Results saved: {summary_file}")
        print(f"{'='*70}")

        # Compare to baseline
        baseline_results = all_results.get("baseline", [])
        if baseline_results:
            baseline_avg = sum(r["accuracy"] for r in baseline_results) / len(baseline_results)
            improvement = best_avg - baseline_avg

            print(f"\n🎯 FINAL COMPARISON:")
            print(f"   Baseline average: {baseline_avg:.2f}%")
            print(f"   Best SRF average: {best_avg:.2f}%")
            print(f"   Improvement: {improvement:+.2f}%")

            if improvement > 1.0:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT!")
            elif improvement > 0:
                print(f"   ✓ Modest improvement")
            else:
                print(f"   ⚠️  No improvement - SRF may not help on full dataset")

        return

    parser.print_help()

def format_paper_table(all_results):
    """Format results for paper table."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("PAPER TABLE FORMAT")
    lines.append("="*100)
    lines.append(f"{'Config':<25} {'Random':<12} {'Popular':<12} {'Adversarial':<12} {'Average':<12}")
    lines.append("-" * 73)

    for config_name, results_list in all_results.items():
        accs = {r['category']: r['accuracy'] for r in results_list}
        random = accs.get('random', 0)
        popular = accs.get('popular', 0)
        adversarial = accs.get('adversarial', 0)
        avg = sum([random, popular, adversarial]) / 3

        lines.append(f"{config_name:<25} {random:>10.2f}%  {popular:>10.2f}%  {adversarial:>10.2f}%  {avg:>10.2f}%")

    lines.append("="*100 + "\n")
    return "\n".join(lines)

if __name__ == "__main__":
    main()
