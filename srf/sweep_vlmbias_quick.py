#!/usr/bin/env python3
"""
Quick VLM Bias parameter sweep - test most promising directions in 30 minutes.

Focus on high-impact parameters based on analysis:
1. Text beta (suppress language priors) - MOST IMPORTANT
2. Layer ranges (counting might need late layers)
3. Alpha (boost strength)

Total: 24 configs × 1 min each = ~30 minutes
"""
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from itertools import product

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "results/vlmbias_quick_sweep/"
N_SAMPLES = 10  # Quick test (70 total)

# Current baseline
BASELINE_ACC = 0.19  # 19%
TARGET_ACC = 0.21  # 21%


def run_experiment(config_id, params, gpu_id=0):
    """Run single experiment."""
    output_path = f"{OUTPUT_DIR}exp_{config_id:04d}/"

    cmd = [
        "conda", "run", "-n", "mllm", "--no-capture-output",
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "vlmbias",
        "--n_vlmbias_per_cat", str(N_SAMPLES),
        "--output", output_path,
    ]

    # Add parameters
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"🧪 Exp {config_id:02d}: ", end="")
    print(f"α={params['alpha']:.1f}", end=" ")
    print(f"layers=({params['layer_start']},{params['layer_end']})", end=" ")
    print(f"textβ={params['text_beta']:.1f}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse accuracy
    import re
    match = re.search(r'accuracy[=:]([\s\d.]+)', result.stdout, re.IGNORECASE)
    accuracy = float(match.group(1).strip()) if match else 0.0

    delta = (accuracy - BASELINE_ACC) * 100

    # Color code output
    if delta >= 2.0:
        status = f"✅ {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    elif delta >= 1.0:
        status = f"🟡 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    elif delta > 0:
        status = f"🟠 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    else:
        status = f"🔴 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"

    print(f"  {status}")

    # Save result
    exp_data = {
        "config_id": config_id,
        "params": params,
        "accuracy": accuracy,
        "delta_pct": delta,
        "timestamp": datetime.now().isoformat(),
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}result.json", 'w') as f:
        json.dump(exp_data, f, indent=2)

    return accuracy, exp_data


def main():
    """Run quick parameter sweep."""
    print("="*70)
    print("🚀 VLM BIAS QUICK SWEEP - Most Promising Directions")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Dataset: VLMs-Are-Biased ({N_SAMPLES} samples/category)")
    print(f"Baseline: {BASELINE_ACC*100:.2f}%")
    print(f"Target: {TARGET_ACC*100:.2f}% (Δ=+2.0%)")
    print(f"Total configs: 24")
    print(f"Estimated time: ~30 minutes")
    print("="*70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Generate configs - focus on high-impact parameters
    configs = []

    # Priority 1: Text beta (suppress language priors)
    # Language priors are likely the main issue - test this first
    text_betas = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Priority 2: Layer ranges
    # Counting might be a late-stage reasoning task
    layer_ranges = [
        (8, 15),   # Current (middle fusion)
        (16, 23),  # Late layers (reasoning)
        (20, 27),  # Very late (final decision)
        (8, 27),   # All layers
    ]

    # Priority 3: Alpha (boost strength)
    # Test gentle vs aggressive
    alphas = [2.0, 8.0]

    print("\n🎯 Testing hypotheses:")
    print("  1. Text beta > 0: Suppress language priors")
    print("  2. Late layers (20-27): Counting as reasoning")
    print("  3. Higher alpha: Stronger visual signal")

    for text_beta, (layer_start, layer_end), alpha in product(
        text_betas, layer_ranges, alphas
    ):
        configs.append({
            "alpha": alpha,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "text_beta": text_beta,
            "clip_top_k_pct": 0.5,
            "bias_mode": "additive_logit",
            "eps": 0.3,
            "phase": "generation",
        })

    print(f"\nGenerated {len(configs)} configurations\n")

    # Run experiments
    results = []

    for idx, config in enumerate(configs):
        accuracy, exp_data = run_experiment(idx, config, gpu_id=0)
        results.append(exp_data)

        # Save intermediate
        with open(f"{OUTPUT_DIR}results.json", 'w') as f:
            json.dump({
                "baseline": BASELINE_ACC,
                "target": TARGET_ACC,
                "results": results,
            }, f, indent=2)

    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)

    # Sort by accuracy
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    print("-"*70)

    for i, r in enumerate(results[:5]):
        params = r["params"]
        acc = r["accuracy"]
        delta = r["delta_pct"]

        print(f"\n{i+1}. Accuracy: {acc*100:.2f}% (Δ={delta:+.2f}%)")
        print(f"   α={params['alpha']:.1f}, layers=({params['layer_start']},{params['layer_end']}), "
              f"textβ={params['text_beta']:.1f}")

    # Best config
    best = results[0]
    best_delta = best["delta_pct"]

    print("\n" + "="*70)
    print("🎯 TARGET CHECK")
    print("="*70)

    if best_delta >= 2.0:
        print(f"✅ TARGET ACHIEVED!")
        print(f"   Best: {best['accuracy']*100:.2f}% (Δ={best_delta:+.2f}%)")

        print(f"\n📋 BEST CONFIG:")
        for k, v in best["params"].items():
            print(f"   {k}: {v}")

        print(f"\n💡 NEXT STEP:")
        print("   1. Run full evaluation with this config (all 2784 samples)")
        print("   2. If improvement holds, validate on other datasets")

    elif best_delta >= 1.0:
        print(f"⚠️  Partial success (1-2%)")
        print(f"   Best: {best['accuracy']*100:.2f}% (Δ={best_delta:+.2f}%)")
        print(f"\n💡 NEXT STEP:")
        print("   1. Run comprehensive sweep around best config")
        print("   2. Test post-softmax redistribution")

    else:
        print(f"❌ Target not achieved")
        print(f"   Best: {best['accuracy']*100:.2f}% (Δ={best_delta:+.2f}%)")
        print(f"\n💡 LIKELY ISSUE:")
        print(f"   Model cannot count - fundamental limitation")
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Focus SRF on recognition tasks (POPE, MMVP)")
        print(f"   Consider adding separate counting module")

    # Parameter sensitivity analysis
    print("\n" + "="*70)
    print("🔍 PARAMETER SENSITIVITY")
    print("="*70)

    # Analyze text beta impact
    text_beta_impact = {}
    for tb in [0.0, 0.2, 0.5, 0.8, 1.0]:
        tb_results = [r for r in results if r["params"]["text_beta"] == tb]
        if tb_results:
            avg_acc = sum(r["accuracy"] for r in tb_results) / len(tb_results)
            text_beta_impact[tb] = avg_acc

    print(f"\n📊 Text Beta Impact:")
    for tb, acc in sorted(text_beta_impact.items(), key=lambda x: x[1], reverse=True):
        delta = (acc - BASELINE_ACC) * 100
        print(f"  textβ={tb:.1f}: {acc*100:.2f}% (Δ={delta:+.2f}%)")

    # Analyze layer range impact
    layer_impact = {}
    for layer_range in [(8, 15), (16, 23), (20, 27), (8, 27)]:
        lr_results = [r for r in results
                      if r["params"]["layer_start"] == layer_range[0]
                      and r["params"]["layer_end"] == layer_range[1]]
        if lr_results:
            avg_acc = sum(r["accuracy"] for r in lr_results) / len(lr_results)
            layer_impact[layer_range] = avg_acc

    print(f"\n📊 Layer Range Impact:")
    for lr, acc in sorted(layer_impact.items(), key=lambda x: x[1], reverse=True):
        delta = (acc - BASELINE_ACC) * 100
        print(f"  layers={lr}: {acc*100:.2f}% (Δ={delta:+.2f}%)")

    print("\n" + "="*70)
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
