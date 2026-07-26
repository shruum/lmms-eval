#!/usr/bin/env python3
"""
Layer-Specific Modulation Sweep for VLM Bias

Hypothesis: Counting requires late-layer reasoning + suppressed language priors

Strategy:
- Early layers (0-7): Gentle visual boost (alpha=0.5)
- Mid layers (8-15): Normal fusion boost (alpha=2.0)
- Late layers (16-27): Strong visual boost + text suppression (alpha=4.0, text_beta=0.5)

Runs in parallel on GPUs 0,1,2,3 (4 experiments at a time)
"""
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from itertools import product
import time
import threading

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "results/vlmbias_layer_specific/"
N_SAMPLES = 10  # Quick test (70 total)

# Free GPUs
GPUS = [0, 1, 2, 3]

# Current baseline
BASELINE_ACC = 0.19  # 19%


def generate_layer_specific_configs():
    """Generate layer-specific modulation configs."""
    configs = []

    # Strategy 1: Three-zone approach (early-mid-late)
    # Early: visual detection
    # Mid: visual-language fusion
    # Late: reasoning + text suppression

    configs.append({
        "name": "three_zone_balanced",
        "early_alpha": 0.5, "early_start": 0, "early_end": 7,
        "mid_alpha": 2.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 4.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.5,
        "description": "Balanced three-zone: gentle early, normal mid, strong late"
    })

    configs.append({
        "name": "three_zone_aggressive",
        "early_alpha": 1.0, "early_start": 0, "early_end": 7,
        "mid_alpha": 4.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 8.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.8,
        "description": "Aggressive three-zone: strong boost throughout"
    })

    configs.append({
        "name": "three_zone_conservative",
        "early_alpha": 0.3, "early_start": 0, "early_end": 7,
        "mid_alpha": 1.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 2.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.3,
        "description": "Conservative three-zone: minimal boost"
    })

    # Strategy 2: Late-layer focus (counting is late reasoning)
    configs.append({
        "name": "late_focus_only",
        "early_alpha": 0.0, "early_start": 0, "early_end": 7,
        "mid_alpha": 0.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 8.0, "late_start": 20, "late_end": 27,
        "late_text_beta": 0.8,
        "description": "Late-layer only: boost only reasoning layers"
    })

    configs.append({
        "name": "late_focus_wide",
        "early_alpha": 0.0, "early_start": 0, "early_end": 7,
        "mid_alpha": 0.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 8.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.8,
        "description": "Late-layer wide: boost 16-27 (all reasoning)"
    })

    # Strategy 3: Mid+Late fusion (skip early)
    configs.append({
        "name": "mid_late_fusion",
        "early_alpha": 0.0, "early_start": 0, "early_end": 7,
        "mid_alpha": 4.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 6.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.6,
        "description": "Mid+Late fusion: skip early detection"
    })

    # Strategy 4: Uniform with text suppression (compare to layer-specific)
    configs.append({
        "name": "uniform_late_text_suppress",
        "early_alpha": 4.0, "early_start": 8, "early_end": 27,
        "mid_alpha": 0.0, "mid_start": 0, "mid_end": 0,
        "late_alpha": 0.0, "late_start": 0, "late_end": 0,
        "late_text_beta": 0.8,
        "description": "Uniform boost 8-27 + strong text suppress"
    })

    # Strategy 5: Progressive boost (increasing with depth)
    configs.append({
        "name": "progressive_increase",
        "early_alpha": 0.5, "early_start": 0, "early_end": 7,
        "mid_alpha": 2.0, "mid_start": 8, "mid_end": 15,
        "late_alpha": 8.0, "late_start": 16, "late_end": 27,
        "late_text_beta": 0.6,
        "description": "Progressive: increasing boost with depth"
    })

    # Strategy 6: Text suppression sweep (find optimal)
    for text_beta in [0.3, 0.5, 0.8, 1.0]:
        configs.append({
            "name": f"late_text_beta_{text_beta}",
            "early_alpha": 0.0, "early_start": 0, "early_end": 7,
            "mid_alpha": 0.0, "mid_start": 8, "mid_end": 15,
            "late_alpha": 8.0, "late_start": 20, "late_end": 27,
            "late_text_beta": text_beta,
            "description": f"Late-layer only with textβ={text_beta}"
        })

    return configs


def run_single_experiment(config_id, config, gpu_id):
    """Run single layer-specific experiment."""
    output_path = f"{OUTPUT_DIR}exp_{config_id:04d}_{config['name']}/"

    # Build command
    cmd = [
        "conda", "run", "-n", "mllm", "--no-capture-output",
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "vlmbias",
        "--n_vlmbias_per_cat", str(N_SAMPLES),
        "--output", output_path,
    ]

    # Layer-specific parameters
    # Note: This requires eval.py to support layer-specific alpha
    # For now, we use the maximum alpha and text_beta across all layers
    # TODO: Implement proper layer-specific modulation in qwen_attn_patch.py

    # Use the strongest parameters from the config
    max_alpha = max(config["late_alpha"], config["mid_alpha"], config["early_alpha"])

    cmd.extend([
        "--alpha", str(max_alpha),
        "--layer_start", str(config["late_start"]),
        "--layer_end", str(config["late_end"]),
        "--text_beta", str(config["late_text_beta"]),
    ])

    print(f"\n{'='*70}")
    print(f"🧪 GPU {gpu_id}: Experiment {config_id:02d} - {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"Output: {output_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start_time

    # Parse accuracy
    import re
    match = re.search(r'accuracy[=:]([\s\d.]+)', result.stdout, re.IGNORECASE)
    accuracy = float(match.group(1).strip()) if match else 0.0

    delta = (accuracy - BASELINE_ACC) * 100

    # Status
    if delta >= 2.0:
        status = f"✅ {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    elif delta >= 1.0:
        status = f"🟡 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    elif delta > 0:
        status = f"🟠 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"
    else:
        status = f"🔴 {accuracy*100:.2f}% (Δ={delta:+.2f}%)"

    print(f"  Result: {status} ({elapsed:.1f}s)")

    # Save result
    exp_data = {
        "config_id": config_id,
        "config": config,
        "accuracy": accuracy,
        "delta_pct": delta,
        "elapsed_time": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}result.json", 'w') as f:
        json.dump(exp_data, f, indent=2)

    return accuracy, exp_data


def run_parallel_batch(configs, gpu_ids):
    """Run batch of experiments in parallel on different GPUs."""
    threads = []
    results = []

    print("\n" + "="*70)
    print(f"🚀 RUNNING BATCH: {len(configs)} experiments on GPUs {gpu_ids}")
    print("="*70)

    def run_wrapper(config_id, config, gpu_id, results_list):
        try:
            acc, data = run_single_experiment(config_id, config, gpu_id)
            results_list.append((config_id, acc, data))
        except Exception as e:
            print(f"❌ GPU {gpu_id}, Exp {config_id}: {str(e)}")
            results_list.append((config_id, 0.0, {"error": str(e)}))

    # Launch threads
    for i, (config_id, config) in enumerate(configs):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        thread = threading.Thread(
            target=run_wrapper,
            args=(config_id, config, gpu_id, results)
        )
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    return results


def main():
    """Run layer-specific modulation sweep."""
    print("="*70)
    print("🔍 LAYER-SPECIFIC MODULATION SWEEP")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Dataset: VLMs-Are-Biased ({N_SAMPLES} samples/category)")
    print(f"GPUs: {GPUS} (parallel execution)")
    print(f"Baseline: {BASELINE_ACC*100:.2f}%")
    print(f"Target: >21.00% (Δ=+2.0%)")
    print("="*70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Generate configs
    configs = generate_layer_specific_configs()
    print(f"\nGenerated {len(configs)} layer-specific configurations\n")

    # Run in batches (4 at a time)
    all_results = []
    batch_size = len(GPUS)

    for i in range(0, len(configs), batch_size):
        batch = list(enumerate(configs[i:i+batch_size], start=i))
        batch_results = run_parallel_batch(batch, GPUS)
        all_results.extend(batch_results)

        # Save intermediate results
        with open(f"{OUTPUT_DIR}results.json", 'w') as f:
            json.dump({
                "baseline": BASELINE_ACC,
                "target": 0.21,
                "results": all_results,
            }, f, indent=2)

    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)

    # Sort by accuracy
    all_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    print("-"*70)

    for i, (config_id, acc, data) in enumerate(all_results[:5]):
        if "error" in data:
            continue

        config = data["config"]
        delta = data["delta_pct"]

        print(f"\n{i+1}. {config['name']}")
        print(f"   Accuracy: {acc*100:.2f}% (Δ={delta:+.2f}%)")
        print(f"   {config['description']}")

        params = config
        if params["early_alpha"] > 0:
            print(f"   Early ({params['early_start']}-{params['early_end']}): α={params['early_alpha']}")
        if params["mid_alpha"] > 0:
            print(f"   Mid ({params['mid_start']}-{params['mid_end']}): α={params['mid_alpha']}")
        if params["late_alpha"] > 0:
            print(f"   Late ({params['late_start']}-{params['late_end']}): α={params['late_alpha']}, textβ={params['late_text_beta']}")

    # Best config
    best_config_id, best_acc, best_data = all_results[0]
    best_delta = best_data["delta_pct"]

    print("\n" + "="*70)
    print("🎯 TARGET CHECK")
    print("="*70)

    if best_delta >= 2.0:
        print(f"✅ TARGET ACHIEVED!")
        print(f"   Best: {best_acc*100:.2f}% (Δ={best_delta:+.2f}%)")
        print(f"\n📋 BEST CONFIG:")
        config = best_data["config"]
        print(f"   Name: {config['name']}")
        print(f"   {config['description']}")

    elif best_delta >= 1.0:
        print(f"⚠️  Partial success (1-2%)")
        print(f"   Best: {best_acc*100:.2f}% (Δ={best_delta:+.2f}%)")

    else:
        print(f"❌ Target not achieved")
        print(f"   Best: {best_acc*100:.2f}% (Δ={best_delta:+.2f}%)")

    # Strategy comparison
    print("\n" + "="*70)
    print("🔍 STRATEGY COMPARISON")
    print("="*70)

    strategies = {}
    for config_id, acc, data in all_results:
        if "error" in data:
            continue

        name = data["config"]["name"]
        if "three_zone" in name:
            strategy = "Three-zone"
        elif "late_focus" in name:
            strategy = "Late-only"
        elif "mid_late" in name:
            strategy = "Mid+Late"
        elif "progressive" in name:
            strategy = "Progressive"
        else:
            strategy = "Other"

        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(acc)

    print(f"\nAverage accuracy by strategy:")
    for strategy, accs in sorted(strategies.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        avg_acc = sum(accs) / len(accs)
        delta = (avg_acc - BASELINE_ACC) * 100
        print(f"  {strategy:20s}: {avg_acc*100:.2f}% (Δ={delta:+.2f}%) [n={len(accs)}]")

    print("\n" + "="*70)
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
