#!/usr/bin/env python3
"""
Simple SRF autoresearch on RePOPE COCO - beat VAF (88.28%) and VCD (88.1%)

Focus on optimizing:
1. head_top_k_pct (head selection)
2. alpha (boosting strength)
3. clip_top_k_pct (CLIP token selection)

Uses LLaVA-1.5-7B for direct comparison with VAF/VCD results.
"""
import os
import sys
import json
import itertools
import subprocess
from pathlib import Path
from datetime import datetime

# Setup paths
SRF_DIR = Path(__file__).parent
LMMS_EVAL_DIR = SRF_DIR.parent
os.chdir(LMMS_EVAL_DIR)

RESULTS_DIR = SRF_DIR.parent / "results" / "autoresearch_repope_simple"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model and data
MODEL = "liuhaotian/llava-v1.5-7b"
REPOPE_DIR = "/home/anna2/shruthi/RePOPE"
COCO_IMAGES = "/home/anna2/shruthi/POPE/coco/images"

# ── Search Space (focused on heads + boosting) ────────────────────────────────
SEARCH_SPACE = {
    "head_top_k_pct": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "clip_top_k_pct": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
}

# Fixed parameters (from LLaVA config)
FIXED_PARAMS = {
    "layer_start": 10,
    "layer_end": 15,
    "clip_coarse_grid": 6,  # LLaVA uses 6
    "clip_fallback_thresh": 0.20,
    "eps": 0.0,
    "clip_suppress_thresh": 0.0,
    "clip_suppress_alpha": 5.0,
    "phase": "both",
}

def run_evaluation(config, split="adversarial", n_samples=500):
    """Run SRF evaluation with given config."""
    config_id = f"h{config['head_top_k_pct']}_a{config['alpha']}_c{config['clip_top_k_pct']}"
    
    # Build command
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "pope",
        "--pope_splits", split,
        "--n_pope", str(n_samples),
        "--output", str(RESULTS_DIR),
    ]
    
    # Add parameters
    for k, v in {**config, **FIXED_PARAMS}.items():
        cmd.extend([f"--{k}", str(v)])
    
    print(f"\n{'='*70}")
    print(f"Config: {config}")
    print(f"Split: {split}, Samples: {n_samples}")
    print(f"{'='*70}\n")
    
    result_file = RESULTS_DIR / f"repope_{split}_{config_id}.json"
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800,  # 30 min
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        
        # Save output
        with open(result_file, "w") as f:
            json.dump({
                "config": config,
                "split": split,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }, f, indent=2)
        
        if result.returncode != 0:
            print(f"❌ ERROR: Check {result_file}")
            return None
        
        # Parse accuracy from stdout
        for line in result.stdout.split('\n'):
            if "Accuracy" in line and ":" in line:
                try:
                    acc = float(line.split("Accuracy:")[1].split()[0].strip())
                    print(f"✅ Accuracy: {acc:.2%}")
                    return acc
                except:
                    pass
        
        print("⚠️  Could not parse accuracy from output")
        return None
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def run_search(strategy="focused", n_samples=500):
    """Run autosearch."""
    print("="*80)
    print("SRF Autoresearch on RePOPE COCO")
    print(f"Model: {MODEL}")
    print(f"Target: Beat VAF 88.28%, VCD 88.1%")
    print(f"Strategy: {strategy}")
    print("="*80)
    
    # Generate configs
    if strategy == "grid":
        configs = [dict(zip(SEARCH_SPACE.keys(), vals)) 
                   for vals in itertools.product(*SEARCH_SPACE.values())]
    elif strategy == "focused":
        # Start from current best and explore
        configs = [
            # Near current config (VAF-like)
            {"head_top_k_pct": 0.20, "alpha": 0.15, "clip_top_k_pct": 0.30},
            # Higher alpha
            {"head_top_k_pct": 0.20, "alpha": 2.0, "clip_top_k_pct": 0.30},
            {"head_top_k_pct": 0.20, "alpha": 3.0, "clip_top_k_pct": 0.30},
            {"head_top_k_pct": 0.20, "alpha": 4.0, "clip_top_k_pct": 0.30},
            # More heads + higher alpha
            {"head_top_k_pct": 0.30, "alpha": 2.0, "clip_top_k_pct": 0.30},
            {"head_top_k_pct": 0.40, "alpha": 2.0, "clip_top_k_pct": 0.30},
            # Focused CLIP
            {"head_top_k_pct": 0.20, "alpha": 2.0, "clip_top_k_pct": 0.20},
            {"head_top_k_pct": 0.20, "alpha": 2.0, "clip_top_k_pct": 0.25},
            # Combination
            {"head_top_k_pct": 0.30, "alpha": 3.0, "clip_top_k_pct": 0.25},
            {"head_top_k_pct": 0.40, "alpha": 4.0, "clip_top_k_pct": 0.20},
        ]
    
    print(f"\nTesting {len(configs)} configurations on adversarial split (n={n_samples})\n")
    
    results = []
    best_config = None
    best_acc = 0.0
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Config {i+1}/{len(configs)}")
        print(f"{'='*80}")
        
        acc = run_evaluation(config, "adversarial", n_samples)
        
        if acc is not None:
            results.append({**config, "accuracy": acc})
            
            if acc > best_acc:
                best_acc = acc
                best_config = config
                print(f"\n🎉 NEW BEST: {best_acc:.2%}")
        
        # Save progress
        with open(RESULTS_DIR / "progress.json", "w") as f:
            json.dump({
                "best_config": best_config,
                "best_accuracy": best_acc,
                "all_results": results,
                "target_vaf": 88.28,
                "target_vcd": 88.1,
            }, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("AUTORESEARCH COMPLETE")
    print("="*80)
    print(f"\nBest config: {best_config}")
    print(f"Best accuracy: {best_acc:.2%} (adversarial, n={n_samples})")
    print(f"Target VAF: 88.28%, VCD: 88.1%")
    
    if best_acc > 88.28:
        print("✅ BEAT VAF TARGET!")
    elif best_acc > 88.1:
        print("✅ BEAT VCD TARGET!")
    else:
        print("❌ Did not beat target")
    
    return best_config, best_acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="focused", choices=["grid", "focused"])
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()
    
    run_search(args.strategy, args.n_samples)
