#!/usr/bin/env python3
"""
Parallel Layer-Specific Modulation sweep across 4 GPUs.

Tests different layer zone configurations and modulation strengths.

Usage:
    python srf/sweep_layer_specific_parallel.py \
        --hard_samples srf/hard_samples_pope.json \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output results/sweep_layer_specific/ \
        --num_gpus 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import subprocess
import os
import time

_SRF_DIR = pathlib.Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Parallel Layer-Specific Modulation sweep")
    p.add_argument("--hard_samples", required=True,
                   help="Path to hard_samples JSON")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results/sweep_layer_specific/")
    p.add_argument("--num_gpus", type=int, default=4,
                   help="Number of GPUs to use (default: 4)")

    # Layer zone boundaries
    p.add_argument("--early_ends", nargs="+", type=int, default=[5, 7, 10],
                   help="Early zone end layers (default: 5 7 10)")

    p.add_argument("--mid_ends", nargs="+", type=int, default=[15, 17, 20],
                   help="Mid zone end layers (default: 15 17 20)")

    # Alpha values for each zone
    p.add_argument("--alpha_early", nargs="+", type=float, default=[0.3, 0.5, 1.0],
                   help="Early zone alpha values")

    p.add_argument("--alpha_mid", nargs="+", type=float, default=[1.0, 2.0, 4.0],
                   help="Mid zone alpha values")

    # Beta for late zone suppression
    p.add_argument("--beta_late", nargs="+", type=float, default=[0.05, 0.1, 0.2],
                   help="Late zone text suppression beta")

    return p.parse_args()


def generate_configs(args):
    """Generate layer-specific configurations."""
    configs = []

    # Baseline
    configs.append({
        "name": "baseline",
        "method": "baseline",
    })

    # Generate all combinations
    for early_end in args.early_ends:
        for mid_end in args.mid_ends:
            if mid_end <= early_end:
                continue  # Skip invalid ranges

            for alpha_e in args.alpha_early:
                for alpha_m in args.alpha_mid:
                    for beta_l in args.beta_late:
                        configs.append({
                            "name": f"ls_e{early_end}_m{mid_end}_ae{alpha_e}_am{alpha_m}_bl{beta_l}",
                            "method": "layer_specific",
                            "layer_early_end": early_end,
                            "layer_mid_end": mid_end,
                            "alpha_early": alpha_e,
                            "alpha_mid": alpha_m,
                            "beta_late": beta_l,
                        })

    return configs


def split_configs(configs, num_gpus):
    """Split configurations across GPUs."""
    baseline_configs = [c for c in configs if c["method"] == "baseline"]
    other_configs = [c for c in configs if c["method"] != "baseline"]

    chunks = [[] for _ in range(num_gpus)]
    chunks[0].extend(baseline_configs)

    for i, cfg in enumerate(other_configs):
        gpu_id = i % num_gpus
        chunks[gpu_id].append(cfg)

    return chunks


def run_gpu_worker(gpu_id, configs, args, output_dir):
    """Run worker on single GPU."""
    chunk_file = output_dir / f"gpu_{gpu_id}_configs.json"
    result_file = output_dir / f"gpu_{gpu_id}_results.json"
    log_file = output_dir / f"gpu_{gpu_id}.log"

    with open(chunk_file, "w") as f:
        json.dump(configs, f, indent=2)

    cmd = [
        "python", str(_SRF_DIR / "sweep_layer_specific_worker.py"),
        "--hard_samples", args.hard_samples,
        "--model", args.model,
        "--configs", str(chunk_file),
        "--output", str(result_file),
        "--gpu_id", str(gpu_id),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting with {len(configs)} configs...")

    start_time = time.time()

    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

    return process, len(configs), start_time


def main():
    print("="*70)
    print("Parallel Layer-Specific Modulation Sweep")
    print("="*70)

    args = parse_args()

    # Load hard samples
    print(f"\nLoading hard samples from: {args.hard_samples}")
    with open(args.hard_samples, "r") as f:
        hard_data = json.load(f)
    n_samples = len(hard_data["samples"])
    print(f"Loaded {n_samples} hard samples")

    # Generate configs
    configs = generate_configs(args)
    print(f"\nGenerated {len(configs)} configurations")

    # Split across GPUs
    config_chunks = split_configs(configs, args.num_gpus)

    print(f"\nDistributing across {args.num_gpus} GPUs:")
    for i, chunk in enumerate(config_chunks):
        print(f"  GPU {i}: {len(chunk)} configs")

    # Create output directory
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Launch workers
    print("\nLaunching GPU workers...")
    print("="*70)

    workers = []
    for gpu_id in range(args.num_gpus):
        if len(config_chunks[gpu_id]) == 0:
            continue

        process, n_configs, start_time = run_gpu_worker(
            gpu_id, config_chunks[gpu_id], args, output_dir
        )

        workers.append({
            "gpu_id": gpu_id,
            "process": process,
            "n_configs": n_configs,
            "start_time": start_time
        })

    # Monitor progress
    print("\nMonitoring workers...")
    print("="*70)

    try:
        while True:
            all_done = True
            for worker in workers:
                if worker["process"].poll() is None:
                    all_done = False
                    break

            if all_done:
                print("\nAll workers completed!")
                break

            elapsed = time.time()
            print(f"\n[{time.strftime('%H:%M:%S')}] Status:")

            for worker in workers:
                gpu_id = worker["gpu_id"]
                process = worker["process"]
                n_configs = worker["n_configs"]

                if process.poll() is None:
                    log_file = output_dir / f"gpu_{gpu_id}.log"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            for line in reversed(lines[-5:]):
                                if "Processed" in line:
                                    print(f"  GPU {gpu_id}: {line.strip()}")
                                    break
                    else:
                        print(f"  GPU {gpu_id}: Running...")
                else:
                    elapsed_time = elapsed - worker["start_time"]
                    print(f"  GPU {gpu_id}: Completed ({elapsed_time:.1f}s)")

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping workers...")
        for worker in workers:
            worker["process"].terminate()
        return

    # Aggregate results
    print("\n" + "="*70)
    print("Aggregating results...")
    print("="*70)

    all_results = {}
    all_summaries = []

    for gpu_id in range(args.num_gpus):
        result_file = output_dir / f"gpu_{gpu_id}_results.json"

        if not result_file.exists():
            continue

        with open(result_file, "r") as f:
            gpu_data = json.load(f)

        all_results.update(gpu_data.get("results", {}))
        all_summaries.extend(gpu_data.get("summary", []))

        print(f"GPU {gpu_id}: {len(gpu_data.get('results', {}))} configs")

    all_summaries.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "="*70)
    print("TOP 15 CONFIGURATIONS")
    print("="*70)

    for i, entry in enumerate(all_summaries[:15]):
        cfg = entry["config"]
        print(f"\n#{i+1}: {cfg['name']}")
        print(f"  Accuracy: {entry['accuracy']:.4f} ({entry['correct']}/{entry['total']})")
        if cfg["method"] == "layer_specific":
            print(f"  Early: 0-{cfg['layer_early_end']} (α={cfg['alpha_early']})")
            print(f"  Mid: {cfg['layer_early_end']}-{cfg['layer_mid_end']} (α={cfg['alpha_mid']})")
            print(f"  Late: {cfg['layer_mid_end']}-end (β={cfg['beta_late']})")

    aggregated_file = output_dir / "aggregated_results.json"
    with open(aggregated_file, "w") as f:
        json.dump({
            "all_results": all_results,
            "summary": all_summaries,
        }, f, indent=2)

    print(f"\nResults saved to: {aggregated_file}")


if __name__ == "__main__":
    main()
