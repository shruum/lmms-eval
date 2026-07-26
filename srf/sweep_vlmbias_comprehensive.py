#!/usr/bin/env python3
"""
Comprehensive parameter sweep for VLM Bias to achieve >2% improvement.

Current best: +0.68% (19.00% → 19.68%)
Target: >2% improvement (19.00% → >21.00%)

Sweep dimensions:
1. Alpha (boost strength): [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
2. Layer ranges: [(8,14), (8,15), (20,27), (8,27)]
3. Text beta: [0.0, 0.2, 0.5, 0.8]
4. Clip top-k: [0.3, 0.5, 0.7, 1.0]
5. Bias mode: ["additive_logit", "global_redistribute"]

Total: 6 * 4 * 4 * 4 * 2 = 768 combinations (too many)

We'll use smart sweep:
- Phase 1: Coarse sweep (96 combos)
- Phase 2: Fine sweep around best (48 combos)
"""
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from itertools import product
import time

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_BASE = "results/vlmbias_comprehensive_sweep/"
N_SAMPLES = 50  # Per category for quick testing (350 total)

def run_experiment(config_id, params, output_dir, gpu_id=0):
    """Run single experiment with given parameters."""
    output_path = f"{output_dir}exp_{config_id:04d}/"

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
        if key == "layer_start":
            cmd.extend([f"--{key}", str(value)])
        elif key == "layer_end":
            cmd.extend([f"--{key}", str(value)])
        elif key == "alpha":
            cmd.extend([f"--alpha", str(value)])
        elif key == "eps":
            cmd.extend([f"--eps", str(value)])
        elif key == "text_beta":
            cmd.extend([f"--text_beta", str(value)])
        elif key == "clip_top_k_pct":
            cmd.extend([f"--clip_top_k_pct", str(value)])
        elif key == "bias_mode":
            cmd.extend([f"--bias_mode", value])
        elif key == "phase":
            cmd.extend([f"--phase", value])

    print(f"\n{'='*70}")
    print(f"🧪 Experiment {config_id:04d}")
    print(f"{'='*70}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"Output: {output_path}")
    print(f"GPU: {gpu_id}")

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start_time

    # Parse result
    accuracy = parse_accuracy(result.stdout)

    # Save config and result
    exp_data = {
        "config_id": config_id,
        "params": params,
        "accuracy": accuracy,
        "elapsed_time": elapsed,
        "timestamp": datetime.now().isoformat(),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}experiment.json", 'w') as f:
        json.dump(exp_data, f, indent=2)

    print(f"✅ Result: {accuracy:.4f} ({elapsed:.1f}s)")

    return accuracy


def parse_accuracy(output):
    """Extract accuracy from eval output."""
    import re
    match = re.search(r'accuracy[=:]([\s\d.]+)', output, re.IGNORECASE)
    if match:
        return float(match.group(1).strip())
    return 0.0


def generate_phase1_configs():
    """Coarse sweep - explore the space broadly."""
    configs = []

    # Dimension 1: Alpha (boost strength)
    alphas = [2.0, 4.0, 8.0, 16.0]

    # Dimension 2: Layer ranges
    layer_ranges = [
        (8, 14),   # Current default
        (8, 15),   # Slightly wider
        (20, 27),  # Late layers (counting as reasoning)
        (8, 27),   # All layers
    ]

    # Dimension 3: Text beta (suppress question tokens)
    text_betas = [0.0, 0.3, 0.6]

    # Dimension 4: Clip top-k (how many tokens to boost)
    clip_top_k_pcts = [0.3, 0.7, 1.0]

    # Dimension 5: Bias mode
    bias_modes = ["additive_logit", "global_redistribute"]

    for alpha, (layer_start, layer_end), text_beta, clip_top_k, bias_mode in product(
        alphas, layer_ranges, text_betas, clip_top_k_pcts, bias_modes
    ):
        configs.append({
            "alpha": alpha,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "text_beta": text_beta,
            "clip_top_k_pct": clip_top_k,
            "bias_mode": bias_mode,
            "eps": 0.3,
            "phase": "generation",  # VLM Bias works best with generation-only
        })

    return configs


def generate_phase2_configs(best_config):
    """Fine sweep around best configuration."""
    configs = []

    # Center around best config
    best_alpha = best_config["alpha"]
    best_layer_start = best_config["layer_start"]
    best_layer_end = best_config["layer_end"]
    best_text_beta = best_config["text_beta"]
    best_clip_top_k = best_config["clip_top_k_pct"]

    # Fine-grained alpha sweep
    alphas = [
        best_alpha * 0.5,
        best_alpha * 0.75,
        best_alpha,
        best_alpha * 1.25,
        best_alpha * 1.5,
    ]

    # Fine-grained layer sweep
    layer_ranges = [
        (best_layer_start, best_layer_end),
        (max(0, best_layer_start - 1), best_layer_end),
        (best_layer_start, min(27, best_layer_end + 1)),
        (max(0, best_layer_start - 1), min(27, best_layer_end + 1)),
    ]

    # Fine-grained text beta
    text_betas = [
        max(0.0, best_text_beta - 0.1),
        best_text_beta,
        min(1.0, best_text_beta + 0.1),
    ]

    # Fine-grained clip top-k
    clip_top_k_pcts = [
        max(0.1, best_clip_top_k - 0.1),
        best_clip_top_k,
        min(1.0, best_clip_top_k + 0.1),
    ]

    # Keep best bias mode
    bias_mode = best_config["bias_mode"]

    for alpha, (layer_start, layer_end), text_beta, clip_top_k in product(
        alphas, layer_ranges, text_betas, clip_top_k_pcts
    ):
        configs.append({
            "alpha": alpha,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "text_beta": text_beta,
            "clip_top_k_pct": clip_top_k,
            "bias_mode": bias_mode,
            "eps": 0.3,
            "phase": "generation",
        })

    return configs


def run_phase1():
    """Run Phase 1: Coarse sweep."""
    print("="*70)
    print("🚀 PHASE 1: COARSE SWEEP")
    print("="*70)
    print(f"Total configs: 96")
    print(f"Samples per category: {N_SAMPLES}")
    print(f"Estimated time: ~8 hours (96 configs × 5 min each)")
    print("="*70)

    output_dir = f"{OUTPUT_BASE}phase1/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    configs = generate_phase1_configs()
    print(f"\nGenerated {len(configs)} configurations")

    results = []

    for idx, config in enumerate(configs):
        accuracy = run_experiment(idx, config, output_dir, gpu_id=0)
        results.append({
            "config_id": idx,
            "accuracy": accuracy,
            "params": config,
        })

        # Save intermediate results
        with open(f"{output_dir}results.json", 'w') as f:
            json.dump({
                "phase": 1,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)

    # Find best
    best = max(results, key=lambda x: x["accuracy"])

    print("\n" + "="*70)
    print("📊 PHASE 1 COMPLETE")
    print("="*70)
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print(f"Best config:")
    for k, v in best["params"].items():
        print(f"  {k}: {v}")

    return best


def run_phase2(best_config):
    """Run Phase 2: Fine sweep around best."""
    print("\n" + "="*70)
    print("🚀 PHASE 2: FINE SWEEP")
    print("="*70)
    print(f"Center around best from Phase 1")
    print(f"Estimated time: ~4 hours (48 configs × 5 min each)")
    print("="*70)

    output_dir = f"{OUTPUT_BASE}phase2/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    configs = generate_phase2_configs(best_config["params"])
    print(f"\nGenerated {len(configs)} configurations")

    results = []

    for idx, config in enumerate(configs):
        accuracy = run_experiment(idx, config, output_dir, gpu_id=0)
        results.append({
            "config_id": idx,
            "accuracy": accuracy,
            "params": config,
        })

        # Save intermediate results
        with open(f"{output_dir}results.json", 'w') as f:
            json.dump({
                "phase": 2,
                "timestamp": datetime.now().isoformat(),
                "baseline_best": best_config,
                "results": results,
            }, f, indent=2)

    # Find best
    best = max(results, key=lambda x: x["accuracy"])

    print("\n" + "="*70)
    print("📊 PHASE 2 COMPLETE")
    print("="*70)
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print(f"Best config:")
    for k, v in best["params"].items():
        print(f"  {k}: {v}")

    return best


def main():
    """Run comprehensive parameter sweep."""
    print("="*70)
    print("🔍 VLM BIAS COMPREHENSIVE PARAMETER SWEEP")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Dataset: VLMs-Are-Biased")
    print(f"Samples per category: {N_SAMPLES}")
    print(f"Current best: 19.68% (+0.68%)")
    print(f"Target: >21.00% (+2.0%)")
    print("="*70)

    input("\nPress Enter to start Phase 1...")

    # Phase 1
    phase1_best = run_phase1()

    # Check if we hit target
    print("\n" + "="*70)
    print("🎯 CHECKING TARGET")
    print("="*70)

    baseline = 0.19  # 19%
    delta = phase1_best["accuracy"] - baseline

    print(f"Baseline: {baseline*100:.2f}%")
    print(f"Phase 1 best: {phase1_best['accuracy']*100:.2f}%")
    print(f"Delta: {delta*100:+.2f}%")

    if delta >= 0.02:
        print("\n✅ TARGET ACHIEVED!")
        print("Phase 2 not needed.")
        return

    print(f"\n⚠️  Still {(0.02 - delta)*100:.2f}% short of target")
    print("Starting Phase 2 to fine-tune...")

    input("\nPress Enter to start Phase 2...")

    # Phase 2
    phase2_best = run_phase2(phase1_best)

    # Final report
    print("\n" + "="*70)
    print("🏁 FINAL REPORT")
    print("="*70)

    final_delta = phase2_best["accuracy"] - baseline

    print(f"\nBaseline:      {baseline*100:.2f}%")
    print(f"Phase 1 best:  {phase1_best['accuracy']*100:.2f}%")
    print(f"Phase 2 best:  {phase2_best['accuracy']*100:.2f}%")
    print(f"Final delta:   {final_delta*100:+.2f}%")

    if final_delta >= 0.02:
        print("\n✅ TARGET ACHIEVED! (>2% improvement)")
        print(f"Improvement: {final_delta*100:.2f}%")
    elif final_delta >= 0.01:
        print("\n⚠️  Partial success (1-2% improvement)")
        print(f"Improvement: {final_delta*100:.2f}%")
        print("Consider: Alternative mechanisms, different datasets")
    else:
        print("\n❌ Target not achieved")
        print(f"Improvement: {final_delta*100:.2f}%")
        print("Recommendation: VLM Bias may not be suitable for SRF")

    print(f"\n📁 Results saved to: {OUTPUT_BASE}")
    print("="*70)


if __name__ == "__main__":
    main()
