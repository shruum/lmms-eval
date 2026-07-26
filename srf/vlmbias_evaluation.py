#!/usr/bin/env python3
"""
VLMs-Are-Biased Evaluation - LLaVA-1.5-7B
Phase 2: Baseline and SRF evaluation on VLMs-Are-Biased dataset.
"""
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
BASE_OUTPUT = "results/vlmbias_evaluation/"
DATASET = "VLMs-Are-Biased (7 domains, bias detection)"

# Best config from POPE sweep
BEST_SRF_CONFIG = {
    "alpha": 6.0,
    "eps": 0.3,
    "clip_coarse_grid": 7,
    "clip_top_k_pct": 0.5,
    "clip_suppress_thresh": 0.0,
}

def run_evaluation(method: str, n_per_cat: int = 10, gpu_id: int = 0) -> dict:
    """Run evaluation (baseline or SRF) on VLMs-Are-Biased."""
    output_dir = f"{BASE_OUTPUT}{method}/"

    # Build command - limit to 10 samples per category for quick test
    if method == "srf":
        cmd = [
            "conda", "run", "-n", "mllm", "--no-capture-output",
            "python", "srf/eval.py",
            "--method", "srf",
            "--model", MODEL,
            "--datasets", "vlmbias",
            "--n_vlmbias_per_cat", str(n_per_cat),
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
            "--datasets", "vlmbias",
            "--n_vlmbias_per_cat", str(n_per_cat),
            "--output", output_dir,
        ]

    print(f"\n{'='*70}")
    print(f"🧪 Running: {method.upper()} on VLMs-Are-Biased")
    print(f"Samples per category: {n_per_cat}")
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
    accuracy, bias_ratio = parse_vlmbias_results(result.stdout)

    print(f"✅ Result: Accuracy={accuracy:.4f}, Bias Ratio={bias_ratio:.4f}")

    return {
        "method": method,
        "dataset": "VLMs-Are-Biased",
        "samples_per_category": n_per_cat,
        "accuracy": accuracy,
        "bias_ratio": bias_ratio,
        "timestamp": datetime.now().isoformat(),
        "output_file": output_file,
    }

def parse_vlmbias_results(output: str) -> tuple:
    """Extract accuracy and bias_ratio from VLM-Bias eval output."""
    import re

    # Look for accuracy patterns
    acc_match = re.search(r'accuracy[=:]([\s\d.]+)', output, re.IGNORECASE)
    bias_match = re.search(r'bias_ratio[=:]([\s\d.]+)', output, re.IGNORECASE)

    accuracy = float(acc_match.group(1).strip()) if acc_match else 0.0
    bias_ratio = float(bias_match.group(1).strip()) if bias_match else 0.0

    return accuracy, bias_ratio

def main():
    """Run Phase 2: VLMs-Are-Biased baseline and SRF evaluation."""
    print("="*70)
    print("🚀 VLMs-Are-Biased Phase 2: Baseline + SRF Evaluation")
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print("="*70)

    results = []

    # Quick test with 10 samples per category (70 total)
    n_per_cat = 10

    # Run baseline
    print("\n📊 Step 1/2: Running Baseline...")
    baseline_result = run_evaluation("baseline", n_per_cat=n_per_cat, gpu_id=0)
    results.append(baseline_result)

    # Run SRF
    print("\n📊 Step 2/2: Running SRF...")
    srf_result = run_evaluation("srf", n_per_cat=n_per_cat, gpu_id=0)
    results.append(srf_result)

    # Save summary
    summary = {
        "model": MODEL,
        "dataset": DATASET,
        "samples_per_category": n_per_cat,
        "total_samples": n_per_cat * 7,  # 7 domains
        "srf_config": BEST_SRF_CONFIG,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = f"{BASE_OUTPUT}summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("📊 FINAL RESULTS - VLMs-Are-Biased Phase 2")
    print("="*70)

    baseline = results[0]
    srf = results[1]

    print(f"\nBaseline:")
    print(f"  Accuracy: {baseline['accuracy']:.4f}")
    print(f"  Bias Ratio: {baseline['bias_ratio']:.4f}")

    print(f"\nSRF:")
    print(f"  Accuracy: {srf['accuracy']:.4f}")
    print(f"  Bias Ratio: {srf['bias_ratio']:.4f}")

    # Calculate delta
    acc_delta = srf['accuracy'] - baseline['accuracy']
    bias_delta = srf['bias_ratio'] - baseline['bias_ratio']

    print(f"\nΔ (SRF - Baseline):")
    print(f"  Accuracy: {acc_delta:+.4f} ({acc_delta*100:+.2f}%)")
    print(f"  Bias Ratio: {bias_delta:+.4f} ({'lower is better' if bias_delta < 0 else 'higher is worse'})")

    print(f"\n📁 Results saved to: {BASE_OUTPUT}")
    print(f"   - Baseline: {baseline['output_file']}")
    print(f"   - SRF: {srf['output_file']}")
    print(f"   - Summary: {summary_file}")

    # Decision
    print("\n" + "="*70)
    print("🎯 DECISION:")
    print("="*70)

    if acc_delta > 0.01:
        print("✅ SRF helps on VLMs-Are-Biased! (>1% improvement)")
        print("   → SRF reduces bias and improves accuracy")
        print("   → This is a NOVEL finding!")
    elif acc_delta > 0:
        print("⚠️  SRF shows minimal improvement on VLMs-Are-Biased (<1%)")
        print("   → SRF may have limited bias reduction capability")
    else:
        print("❌ SRF provides NO improvement on VLMs-Are-Biased")
        print("   → Confirms SRF fails on precision/counting tasks")
        print("   → Consistent with MME counting failure (73% → 73%)")

    # Comparison to paper baseline
    print("\n" + "="*70)
    print("📚 COMPARISON TO PAPER:")
    print("="*70)
    print("Paper reports ~17.05% accuracy for state-of-the-art VLMs")
    print(f"Our baseline: {baseline['accuracy']*100:.2f}%")
    print(f"Our SRF:      {srf['accuracy']*100:.2f}%")
    if baseline['accuracy'] < 0.20:
        print("✅ Our baseline is in the expected range (models are biased)")
    else:
        print("⚠️  Our baseline differs from paper (may need more samples)")

    print("="*70)

if __name__ == "__main__":
    main()
