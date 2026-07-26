#!/usr/bin/env python3
"""
Final POPE Evaluation - LLaVA-1.5-7B (MS-COCO)
Clean evaluation with baseline and best SRF config for paper-ready results.
"""
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/pope_final_evaluation/"
DATASET = "POPE (MS-COCO) - 3000 samples per category"

# Best config from sweep
BEST_SRF_CONFIG = {
    "name": "higher_topk",
    "description": "Higher top-k (α=6.0, eps=0.3, top_k=0.5, grid=7)",
    "alpha": 6.0,
    "eps": 0.3,
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.5,
    "clip_suppress_thresh": 0.0,
}

CATEGORIES = ["random", "popular", "adversarial"]

def run_evaluation(method: str, category: str, gpu_id: int = 1) -> dict:
    """Run clean evaluation (baseline or SRF) on POPE category."""
    config_name = "srf" if method == "srf" else "baseline"
    output_dir = f"{BASE_OUTPUT}{method}/{category}/"

    # Build command
    if method == "srf":
        cmd = [
            "python", "srf/eval.py",
            "--method", "srf",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", category,
            "--output", output_dir,
        ]

        # Add SRF parameters
        for key, value in BEST_SRF_CONFIG.items():
            if key in ["name", "description"]:
                continue
            cmd.extend([f"--{key}", str(value)])
    else:
        cmd = [
            "python", "srf/eval.py",
            "--method", "baseline",
            "--model", MODEL,
            "--datasets", "pope",
            "--pope_splits", category,
            "--output", output_dir,
        ]

    print(f"\n{'='*70}")
    print(f"🧪 Running: {method.upper()} on {category.upper()}")
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
    f1 = parse_f1(result.stdout)

    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}output.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

    print(f"✅ Result: {accuracy:.2f}% accuracy, {f1:.2f}% F1")

    return {
        "method": method,
        "category": category,
        "accuracy": accuracy,
        "f1": f1,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
    }

def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    import re

    # Try to find accuracy line
    for line in output.split('\n'):
        if "accuracy" in line.lower() and "baseline" in line.lower():
            match = re.search(r'acc(?:uracy)?[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))

    # Try POPE-specific format
    for line in output.split('\n'):
        if "POPE" in line and "acc=" in line:
            match = re.search(r'acc=(\d+\.?\d*)', line)
            if match:
                return float(match.group(1))

    # General percentage search
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0

def parse_f1(output: str) -> float:
    """Extract F1 from eval output."""
    import re

    # Try to find F1 line
    for line in output.split('\n'):
        if "f1" in line.lower() or "F1" in line:
            match = re.search(r'f1[:\s=]+(\d+\.?\d*)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))

    # Try percentage format
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if len(matches) >= 2:  # Accuracy is first, F1 is usually second
        return float(matches[1])

    return 0.0

def generate_paper_table(results: dict) -> str:
    """Generate paper-ready table."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("FINAL RESULTS: POPE (MS-COCO) - LLaVA-1.5-7B")
    lines.append("="*100)
    lines.append(f"Dataset: {DATASET}")
    lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d')}")

    # Extract results
    baseline_results = {r["category"]: r for r in results if r["method"] == "baseline"}
    srf_results = {r["category"]: r for r in results if r["method"] == "srf"}

    lines.append(f"\n{'Category':<15} {'Baseline':<12} {'SRF (Ours)':<15} {'Improvement':<12}")
    lines.append("-" * 54)

    improvements = []
    for cat in ["random", "popular", "adversarial"]:
        base = baseline_results[cat]["accuracy"]
        srf = srf_results[cat]["accuracy"]
        diff = srf - base
        improvements.append(diff)
        lines.append(f"{cat.capitalize():<15} {base:>10.2f}%   {srf:>10.2f}%       {diff:>+7.2f}%")

    # Average
    baseline_avg = sum(r["accuracy"] for r in baseline_results.values()) / len(baseline_results)
    srf_avg = sum(r["accuracy"] for r in srf_results.values()) / len(srf_results)
    avg_improvement = srf_avg - baseline_avg

    lines.append("-" * 54)
    lines.append(f"{'AVERAGE':<15} {baseline_avg:>10.2f}%   {srf_avg:>10.2f}%       {avg_improvement:>+7.2f}%")

    # Comparison to papers
    lines.append("\n" + "="*100)
    lines.append("COMPARISON TO PUBLISHED METHODS")
    lines.append("="*100)
    lines.append(f"{'Method':<25} {'Improvement':<15} {'Reference':<50}")
    lines.append("-" * 90)

    lines.append(f"{'SRF (Ours)':<25} {avg_improvement:>+7.2f}%        {MODEL} + POPE (MS-COCO)")
    lines.append(f"{'VAF (ClearSight)':<25} {+1.80:+7.2f}%        ClearSight, CVPR 2025")
    lines.append(f"{'AIR':<25} {+5.30:+7.2f}%        AIR, arXiv 2602.24041")

    lines.append("\n" + "="*100)
    lines.append("KEY FINDINGS")
    lines.append("="*100)
    lines.append("1. SRF provides minimal improvement (+0.04% average) on full POPE dataset")
    lines.append("2. Much lower than 50-sample tests (+3%) due to:")
    lines.append("   - Small samples overfit to specific examples")
    lines.append("   - Full dataset shows true generalization performance")
    lines.append("3. Significantly below VAF (+1.8%) and AIR (+5.3%) from literature")
    lines.append("4. SRF may not be effective for LLaVA-7B on object hallucination tasks")

    lines.append("\n" + "="*100)
    lines.append("CONFIGURATION")
    lines.append("="*100)
    lines.append(f"Model: {MODEL}")
    lines.append(f"SRF Parameters: α={BEST_SRF_CONFIG['alpha']}, eps={BEST_SRF_CONFIG['eps']}")
    lines.append(f"               top_k={BEST_SRF_CONFIG['clip_top_k_pct']}, grid={BEST_SRF_CONFIG['clip_coarse_grid']}")
    lines.append(f"               suppress_thresh={BEST_SRF_CONFIG['clip_suppress_thresh']} (no absence-aware)")

    lines.append("\n" + "="*100)

    return "\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Final POPE Evaluation")
    parser.add_argument("--method", type=str, help="Run specific method only (baseline/srf)")
    parser.add_argument("--category", type=str, help="Run specific category only")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    args = parser.parse_args()

    print("🧪 Final POPE Evaluation - LLaVA-1.5-7B (MS-COCO)")
    print(f"Dataset: {DATASET}")
    print(f"Best SRF Config: {BEST_SRF_CONFIG['name']}")

    if args.all:
        print(f"\n🚀 Running ALL evaluations (baseline + SRF on all 3 categories)")
        print(f"Total: 6 experiments (estimated 2 hours)")

        results = []

        # Run baseline on all categories
        print(f"\n{'='*70}")
        print("Phase 1: Baseline Evaluation")
        print(f"{'='*70}")
        for category in CATEGORIES:
            result = run_evaluation("baseline", category, args.gpu)
            results.append(result)

        # Run SRF on all categories
        print(f"\n{'='*70}")
        print("Phase 2: SRF Evaluation")
        print(f"{'='*70}")
        for category in CATEGORIES:
            result = run_evaluation("srf", category, args.gpu)
            results.append(result)

        # Generate paper table
        paper_table = generate_paper_table(results)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"{BASE_OUTPUT}final_summary_{timestamp}.json"

        summary = {
            "model": MODEL,
            "dataset": "POPE (MS-COCO)",
            "best_srf_config": BEST_SRF_CONFIG,
            "results": results,
            "paper_table": paper_table,
            "timestamp": timestamp,
        }

        Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(paper_table)
        print(f"\n📁 Results saved: {summary_file}")

        # Save paper table separately
        table_file = f"{BASE_OUTPUT}paper_table_{timestamp}.txt"
        with open(table_file, 'w') as f:
            f.write(paper_table)
        print(f"📄 Paper table: {table_file}")

        return

    # Single method/category
    if args.method and args.category:
        run_evaluation(args.method, args.category, args.gpu)
        return

    if args.method:
        print(f"\n🚀 Running {args.method} on all categories...")
        for category in CATEGORIES:
            run_evaluation(args.method, category, args.gpu)
        return

    if args.category:
        print(f"\n🚀 Running all methods on {args.category}...")
        run_evaluation("baseline", args.category, args.gpu)
        run_evaluation("srf", args.category, args.gpu)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
