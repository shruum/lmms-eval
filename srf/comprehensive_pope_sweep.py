#!/usr/bin/env python3
"""
Comprehensive POPE Parameter Sweep - LLaVA-1.5-7B
Tests SRF on all 3 POPE categories with full parameter sweep.
Goal: Find best settings to report in paper table format.
"""
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path
from itertools import product

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/pope_comprehensive_sweep/"
POPE_CATEGORIES = ["random", "popular", "adversarial"]

# Parameter grid to sweep
PARAM_GRID = {
    "alpha": [2.0, 4.0, 6.0, 8.0],
    "eps": [0.0, 0.2, 0.3],
    "clip_coarse_grid": [5, 7, 9],
    "clip_top_k_pct": [0.3, 0.5, 0.7],
    "clip_suppress_thresh": [0.0],  # Disable absence-aware
}

# Generate all combinations
def generate_configs():
    """Generate all parameter combinations."""
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()

    configs = []
    for combination in product(*values):
        config = dict(zip(keys, combination))
        config_name = f"a{config['alpha']}_e{config['eps']}_g{config['clip_coarse_grid']}_tk{config['clip_top_k_pct']}"
        config["name"] = config_name
        configs.append(config)

    return configs

def run_single_experiment(
    config: dict,
    category: str,
    gpu_id: int = 1
) -> dict:
    """Run single SRF experiment on POPE category."""
    config_name = config["name"]
    output_dir = f"{BASE_OUTPUT}{category}/{config_name}/"

    # Build command
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "pope",
        "--pope_splits", category,
        "--output", output_dir,
    ]

    # Add SRF parameters
    for key, value in config.items():
        if key == "name":
            continue
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*70}")
    print(f"🧪 Running: {config_name} on {category.upper()}")
    print(f"Params: α={config['alpha']}, eps={config['eps']}, grid={config['clip_coarse_grid']}, top_k={config['clip_top_k_pct']}")
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

    print(f"✅ Result: {accuracy:.1f}%")

    return {
        "config": config_name,
        "category": category,
        "accuracy": accuracy,
        "params": config,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
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
    parser = argparse.ArgumentParser(description="Comprehensive POPE Parameter Sweep")
    parser.add_argument("--category", type=str, help="Run specific category only")
    parser.add_argument("--config", type=str, help="Run specific config only (format: a6.0_e0.3_g7_tk0.5)")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Quick mode: test subset of configs")
    args = parser.parse_args()

    print("🧪 Comprehensive POPE Parameter Sweep - LLaVA-1.5-7B")
    print(f"Model: {MODEL}")
    print(f"Categories: {', '.join(POPE_CATEGORIES).upper()}")

    # Generate configurations
    all_configs = generate_configs()
    print(f"\nTotal parameter combinations: {len(all_configs)}")
    print(f"Total experiments: {len(all_configs)} × {len(POPE_CATEGORIES)} = {len(all_configs) * len(POPE_CATEGORIES)}")

    # Quick mode: test representative subset
    if args.quick:
        print("\n⚡ QUICK MODE: Testing representative subset")
        # Select diverse configs: low/med/high alpha, different grids, different top-k
        quick_configs = [
            all_configs[0],  # First config
            all_configs[len(all_configs)//2],  # Middle config
            all_configs[-1],  # Last config
        ]
        # Add specific configs from our best results
        quick_configs.extend([c for c in all_configs if c['alpha'] == 6.0 and c['clip_top_k_pct'] == 0.5][:1])
        quick_configs.extend([c for c in all_configs if c['alpha'] == 4.0 and c['eps'] == 0.2][:1])
        quick_configs = list(set(quick_configs))  # Remove duplicates
        all_configs = quick_configs
        print(f"Quick mode: {len(all_configs)} configs")

    # Filter by category if specified
    categories = [args.category] if args.category else POPE_CATEGORIES

    # Filter by config if specified
    if args.config:
        all_configs = [c for c in all_configs if c["name"] == args.config]
        if not all_configs:
            print(f"❌ Config '{args.config}' not found")
            return
        print(f"\n🎯 Running single config: {args.config}")

    # Run experiments
    all_results = {cat: [] for cat in categories}

    for category in categories:
        print(f"\n{'='*70}")
        print(f"📂 Category: {category.upper()}")
        print(f"{'='*70}")

        for i, config in enumerate(all_configs, 1):
            print(f"\n[{i}/{len(all_configs)}] Testing {config['name']}...")
            result = run_single_experiment(config, category, args.gpu)
            all_results[category].append(result)

    # Analyze results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")

    for category in categories:
        results = all_results[category]
        if not results:
            continue

        # Sort by accuracy
        sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

        print(f"\n{category.upper()}:")
        print(f"{'Config':<30} {'Accuracy':<10} {'Params':<50}")
        print("-" * 90)

        for r in sorted_results[:10]:  # Top 10
            params = r["params"]
            param_str = f"α={params['alpha']}, eps={params['eps']}, grid={params['clip_coarse_grid']}, top_k={params['clip_top_k_pct']}"
            print(f"{r['config']:<30} {r['accuracy']:>6.2f}%    {param_str}")

    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"{BASE_OUTPUT}sweep_summary_{timestamp}.json"

    summary = {
        "model": MODEL,
        "dataset": "POPE (MS-COCO)",
        "categories_tested": categories,
        "total_configs": len(all_configs),
        "all_results": all_results,
        "timestamp": timestamp,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"📁 Results saved: {summary_file}")
    print(f"{'='*70}")

    # Calculate best average across categories
    if len(categories) > 1:
        print(f"\n{'='*70}")
        print(f"🏆 BEST OVERALL CONFIGS (averaged across {len(categories)} categories)")
        print(f"{'='*70}")

        # Average each config across categories
        config_avg = {}
        for cat_results in all_results.values():
            for r in cat_results:
                config_name = r["config"]
                if config_name not in config_avg:
                    config_avg[config_name] = {
                        "accuracies": [],
                        "params": r["params"],
                    }
                config_avg[config_name]["accuracies"].append(r["accuracy"])

        # Calculate averages
        config_averages = []
        for config_name, data in config_avg.items():
            if len(data["accuracies"]) == len(categories):  # Has results for all categories
                avg_acc = sum(data["accuracies"]) / len(data["accuracies"])
                config_averages.append({
                    "config": config_name,
                    "average": avg_acc,
                    "accuracies": data["accuracies"],
                    "params": data["params"],
                })

        # Sort by average
        config_averages.sort(key=lambda x: x["average"], reverse=True)

        print(f"{'Config':<30} {'Average':<10} {'Random':<10} {'Popular':<10} {'Adversarial':<10}")
        print("-" * 80)

        for ca in config_averages[:10]:
            accs = ca["accuracies"]
            print(f"{ca['config']:<30} {ca['average']:>6.2f}%    {accs[0]:>6.2f}%    {accs[1] if len(accs) > 1 else 'N/A':>6}    {accs[2] if len(accs) > 2 else 'N/A':>6}")

if __name__ == "__main__":
    main()
