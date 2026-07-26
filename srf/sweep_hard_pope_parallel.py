#!/usr/bin/env python3
"""
Parallel hyperparameter sweep on hard POPE samples across multiple GPUs.

Automatically detects available GPUs, distributes configurations, runs in parallel,
and aggregates results.

Usage:
    python srf/sweep_hard_pope_parallel.py \
        --hard_samples srf/hard_samples_pope.json \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output results/sweep_hard_pope_parallel/ \
        --num_gpus 8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import subprocess
import os
from collections import defaultdict
from itertools import product
import time

_SRF_DIR = pathlib.Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Parallel hyperparameter sweep across GPUs")
    p.add_argument("--hard_samples", required=True,
                   help="Path to hard_samples JSON from find_hard_pope_samples.py")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results/sweep_hard_pope_parallel/")
    p.add_argument("--num_gpus", type=int, default=4,
                   help="Number of GPUs to use (default: 4, other 4 for Layer-Specific Modulation)")

    # Layer ranges to test
    p.add_argument("--layer_ranges", nargs="+", default=["5-10", "8-15", "10-15", "15-25"],
                   help="Layer ranges to test (format: start-end)")

    # Head percentages
    p.add_argument("--head_pcts", nargs="+", type=float, default=[0.3, 0.5, 0.8],
                   help="Head top-k percentages to test")

    # Alpha values
    p.add_argument("--alphas", nargs="+", type=float, default=[0.15, 0.5, 1.0, 2.0],
                   help="Alpha values to test")

    # Text suppression options
    p.add_argument("--with_text_suppression", action="store_true",
                   help="Include text suppression (text_beta)")
    p.add_argument("--with_sys_suppression", action="store_true",
                   help="Include system prompt suppression (sys_beta)")

    # Post-softmax variant
    p.add_argument("--include_post_softmax", action="store_true",
                   help="Include post-softmax redistribution variant")

    return p.parse_args()


def generate_configs(args):
    """Generate all configurations to test."""
    configs = []

    # Parse layer ranges
    layer_ranges = []
    for lr in args.layer_ranges:
        start, end = map(int, lr.split("-"))
        layer_ranges.append((start, end))

    # Baseline config
    configs.append({
        "name": "baseline",
        "method": "baseline",
        "layer_start": None,
        "layer_end": None,
        "head_top_k_pct": None,
        "alpha": None,
        "text_beta": None,
        "sys_beta": None,
    })

    # Generate SRF configs
    for layer_start, layer_end in layer_ranges:
        for head_pct in args.head_pcts:
            for alpha in args.alphas:
                # No suppression
                configs.append({
                    "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}",
                    "method": "srf",
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                    "head_top_k_pct": head_pct,
                    "alpha": alpha,
                    "text_beta": None,
                    "sys_beta": None,
                })

                # Text suppression
                if args.with_text_suppression:
                    configs.append({
                        "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}_txt",
                        "method": "srf",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": 0.1,
                        "sys_beta": None,
                    })

                # Sys suppression
                if args.with_sys_suppression:
                    configs.append({
                        "name": f"srf_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}_sys",
                        "method": "srf",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": None,
                        "sys_beta": 0.1,
                    })

    # Post-softmax configs
    if args.include_post_softmax:
        for layer_start, layer_end in layer_ranges:
            for head_pct in args.head_pcts:
                for alpha in args.alphas:
                    configs.append({
                        "name": f"post_l{layer_start}-{layer_end}_h{head_pct}_a{alpha}",
                        "method": "post_softmax",
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "head_top_k_pct": head_pct,
                        "alpha": alpha,
                        "text_beta": None,
                        "sys_beta": None,
                    })

    return configs


def split_configs(configs, num_gpus):
    """Split configurations evenly across GPUs."""
    # Put baseline on first GPU
    baseline_configs = [c for c in configs if c["method"] == "baseline"]
    other_configs = [c for c in configs if c["method"] != "baseline"]

    # Split other configs evenly
    chunks = [[] for _ in range(num_gpus)]
    chunks[0].extend(baseline_configs)

    for i, cfg in enumerate(other_configs):
        gpu_id = i % num_gpus
        chunks[gpu_id].append(cfg)

    return chunks


def run_gpu_worker(gpu_id, configs, args, output_dir):
    """Run a subset of configurations on a single GPU."""
    chunk_file = output_dir / f"gpu_{gpu_id}_configs.json"
    result_file = output_dir / f"gpu_{gpu_id}_results.json"
    log_file = output_dir / f"gpu_{gpu_id}.log"

    # Save configs for this GPU
    with open(chunk_file, "w") as f:
        json.dump(configs, f, indent=2)

    # Build command
    cmd = [
        "python", str(_SRF_DIR / "sweep_hard_pope_worker.py"),
        "--hard_samples", args.hard_samples,
        "--model", args.model,
        "--configs", str(chunk_file),
        "--output", str(result_file),
        "--gpu_id", str(gpu_id),
    ]

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting with {len(configs)} configs...")

    # Run and log
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


def aggregate_results(output_dir, num_gpus):
    """Aggregate results from all GPUs."""
    print("\n" + "="*70)
    print("Aggregating results from all GPUs...")
    print("="*70)

    all_results = {}
    all_summaries = []

    for gpu_id in range(num_gpus):
        result_file = output_dir / f"gpu_{gpu_id}_results.json"

        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping GPU {gpu_id}")
            continue

        with open(result_file, "r") as f:
            gpu_data = json.load(f)

        all_results.update(gpu_data.get("results", {}))
        all_summaries.extend(gpu_data.get("summary", []))

        print(f"GPU {gpu_id}: {len(gpu_data.get('results', {}))} configs completed")

    # Sort by accuracy
    all_summaries.sort(key=lambda x: x["accuracy"], reverse=True)

    # Print top results
    print("\n" + "="*70)
    print("TOP 20 CONFIGURATIONS (All GPUs)")
    print("="*70)

    for i, entry in enumerate(all_summaries[:20]):
        cfg = entry["config"]
        acc = entry["accuracy"]
        correct = entry["correct"]
        total = entry["total"]

        print(f"\n#{i+1}: {cfg['name']}")
        print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
        print(f"  Method: {cfg['method']}")
        if cfg["layer_start"] is not None:
            print(f"  Layers: {cfg['layer_start']}-{cfg['layer_end']}")
        if cfg["head_top_k_pct"] is not None:
            print(f"  Heads: {cfg['head_top_k_pct']:.2f}")
        if cfg["alpha"] is not None:
            print(f"  Alpha: {cfg['alpha']}")
        if cfg["text_beta"] is not None:
            print(f"  Text beta: {cfg['text_beta']}")
        if cfg["sys_beta"] is not None:
            print(f"  Sys beta: {cfg['sys_beta']}")

    # Save aggregated results
    aggregated_file = output_dir / "aggregated_results.json"
    with open(aggregated_file, "w") as f:
        json.dump({
            "all_results": all_results,
            "summary": all_summaries,
            "total_configs": len(all_summaries),
        }, f, indent=2)

    print(f"\nAggregated results saved to: {aggregated_file}")
    print("="*70)


def main():
    print("="*70)
    print("Parallel Hard POPE Sample Hyperparameter Sweep")
    print("="*70)

    args = parse_args()

    # Load hard samples to get count
    print(f"\nLoading hard samples from: {args.hard_samples}")
    with open(args.hard_samples, "r") as f:
        hard_data = json.load(f)
    n_samples = len(hard_data["samples"])
    print(f"Loaded {n_samples} hard samples")

    # Generate all configurations
    configs = generate_configs(args)
    print(f"\nGenerated {len(configs)} configurations to test")

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
    print("\nMonitoring workers (Ctrl+C to stop)...")
    print("="*70)

    try:
        while True:
            # Check if all workers are done
            all_done = True
            for worker in workers:
                if worker["process"].poll() is None:
                    all_done = False
                    break

            if all_done:
                print("\nAll workers completed!")
                break

            # Print progress
            elapsed = time.time()
            print(f"\n[{time.strftime('%H:%M:%S')}] Worker status:")

            for worker in workers:
                gpu_id = worker["gpu_id"]
                process = worker["process"]
                n_configs = worker["n_configs"]
                start_time = worker["start_time"]

                if process.poll() is None:
                    # Still running - estimate progress from log
                    log_file = output_dir / f"gpu_{gpu_id}.log"
                    if log_file.exists():
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            # Find last progress line
                            for line in reversed(lines):
                                if "Processed" in line:
                                    print(f"  GPU {gpu_id}: {line.strip()}")
                                    break
                    else:
                        print(f"  GPU {gpu_id}: Starting...")
                else:
                    elapsed_time = elapsed - start_time
                    print(f"  GPU {gpu_id}: Completed ({n_configs} configs, {elapsed_time:.1f}s)")

            time.sleep(30)  # Update every 30 seconds

    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping workers...")
        for worker in workers:
            worker["process"].terminate()
        print("Workers stopped.")
        return

    # Aggregate results
    aggregate_results(output_dir, args.num_gpus)


if __name__ == "__main__":
    main()
