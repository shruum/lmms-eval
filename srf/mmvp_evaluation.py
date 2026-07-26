#!/usr/bin/env python3
"""
MMVP Evaluation - LLaVA-1.5-7B
Phase 1: Baseline and SRF evaluation on MMVP dataset.
"""
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/mmvp_evaluation/"
DATASET = "MMVP (150 pairs, 300 images)"

# Best config from POPE sweep
BEST_SRF_CONFIG = {
    "alpha": 6.0,
    "eps": 0.3,
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.5,
    "clip_suppress_thresh": 0.0,
}

def run_evaluation(method: str, gpu_id: int = 0) -> dict:
    """Run evaluation (baseline or SRF) on MMVP."""
    output_dir = f"{BASE_OUTPUT}{method}/"

    # Build command
    if method == "srf":
        cmd = [
            "conda", "run", "-n", "mllm", "--no-capture-output",
            "python", "srf/eval.py",
            "--method", "srf",
            "--model", MODEL,
            "--datasets", "mmvp",
            "--output", output_dir,
        ]

        # Add SRF parameters
        for key, value in BEST_SRF_CONFIG.items():
            cmd.extend([f"--{key}", str(value)])
    else:
        cmd = [
            "conda", "run", "-n", "mllm", "--no-capture-output",
            "python", "srf/eval.py",
            "--method", "baseline",
            "--model", MODEL,
            "--datasets", "mmvp",
            "--output", output_dir,
        ]

    print(f"\n{'='*70}")
    print(f"🧪 Running: {method.upper()} on MMVP")
    print(f"GPU: {gpu_id}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}output.txt"

    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

    # Parse metrics
    pair_acc, img_acc = parse_mmvp_results(result.stdout)

    print(f"✅ Result: Pair accuracy={pair_acc:.4f}, Image accuracy={img_acc:.4f}")

    return {
        "method": method,
        "dataset": "MMVP",
        "pair_accuracy": pair_acc,
        "image_accuracy": img_acc,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
    }

def parse_mmvp_results(output: str) -> tuple:
    """Extract pair and image accuracy from MMVP eval output."""
    import re

    # Look for "MMVP baseline: pair=0.XXXX img=0.XXXX" or "SRF: pair=0.XXXX"
    pair_match = re.search(r'pair[=:]([\d.]+)', output)
    img_match = re.search(r'img[=:]([\d.]+)', output)

    pair_acc = float(pair_match.group(1)) if pair_match else 0.0
    img_acc = float(img_match.group(1)) if img_match else 0.0

    return pair_acc, img_acc

def main():
    """Run Phase 1: MMVP baseline and SRF evaluation."""
    print("="*70)
    print("🚀 MMVP Phase 1: Baseline + SRF Evaluation")
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print("="*70)

    results = []

    # Run baseline
    print("\n📊 Step 1/2: Running Baseline...")
    baseline_result = run_evaluation("baseline", gpu_id=0)
    results.append(baseline_result)

    # Run SRF
    print("\n📊 Step 2/2: Running SRF...")
    srf_result = run_evaluation("srf", gpu_id=0)
    results.append(srf_result)

    # Save summary
    summary = {
        "model": MODEL,
        "dataset": DATASET,
        "srf_config": BEST_SRF_CONFIG,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = f"{BASE_OUTPUT}summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("📊 FINAL RESULTS - MMVP Phase 1")
    print("="*70)

    baseline = results[0]
    srf = results[1]

    print(f"\nBaseline:")
    print(f"  Pair Accuracy: {baseline['pair_accuracy']:.4f}")
    print(f"  Image Accuracy: {baseline['image_accuracy']:.4f}")

    print(f"\nSRF:")
    print(f"  Pair Accuracy: {srf['pair_accuracy']:.4f}")
    print(f"  Image Accuracy: {srf['image_accuracy']:.4f}")

    # Calculate delta
    pair_delta = srf['pair_accuracy'] - baseline['pair_accuracy']
    img_delta = srf['image_accuracy'] - baseline['image_accuracy']

    print(f"\nΔ (SRF - Baseline):")
    print(f"  Pair Accuracy: {pair_delta:+.4f} ({pair_delta*100:+.2f}%)")
    print(f"  Image Accuracy: {img_delta:+.4f} ({img_delta*100:+.2f}%)")

    print(f"\n📁 Results saved to: {BASE_OUTPUT}")
    print(f"   - Baseline: {baseline['output_file']}")
    print(f"   - SRF: {srf['output_file']}")
    print(f"   - Summary: {summary_file}")

    # Decision
    print("\n" + "="*70)
    print("🎯 DECISION:")
    print("="*70)
    if pair_delta > 0.01:
        print("✅ SRF helps on MMVP! (>1% improvement)")
        print("   → SRF generalizes to visual pattern discrimination")
        print("   → Report: POPE + MMVP both show improvement")
    elif pair_delta > 0:
        print("⚠️  SRF shows minimal improvement on MMVP (<1%)")
        print("   → SRF may have limited generalization")
        print("   → Consider reporting as negative result")
    else:
        print("❌ SRF provides NO improvement on MMVP")
        print("   → SRF only works on POPE (very specific)")
        print("   → Conclusion: SRF has extremely limited applicability")

    print("="*70)

if __name__ == "__main__":
    main()
