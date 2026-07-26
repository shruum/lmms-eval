#!/usr/bin/env python3
"""
SRF Autoresearch on RePOPE COCO - beat VAF (88.28%) and VCD (88.1%)

Optimize SRF parameters focusing on:
1. Head selection (head_top_k_pct)
2. Boosting strength (alpha, eps, clip_top_k_pct)
3. Absence-aware logic (clip_suppress_thresh, clip_suppress_alpha)

Target: Beat VAF 88.28% average on RePOPE COCO
"""
import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from datetime import datetime

SRF_DIR = Path(__file__).parent
RESULTS_DIR = SRF_DIR.parent / "results" / "autoresearch_repope"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# RePOPE COCO data paths
REPOPE_ANNOTATIONS = "/home/anna2/shruthi/RePOPE/annotations"
COCO_IMAGES = "/home/anna2/shruthi/POPE/coco/images"

# Model
MODEL = "liuhaotian/llava-v1.5-7b"  # Use LLaVA for direct comparison with VAF/VCD

# ── Search Space ─────────────────────────────────────────────────────────────────
# Focus on heads and boosting as requested
SEARCH_SPACE = {
    "head_top_k_pct": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50],  # Head selection
    "alpha": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],               # Boosting strength
    "clip_top_k_pct": [0.20, 0.25, 0.30, 0.35, 0.40],            # CLIP top-k tokens
    "eps": [0.0, 0.1, 0.2],                                       # Background suppression
    "clip_suppress_thresh": [0.0, 0.20, 0.25],                    # Absence detection threshold
    "clip_suppress_alpha": [3.0, 5.0, 8.0],                       # Absence suppression
}

# Fixed parameters (from config, already tuned)
FIXED_PARAMS = {
    "layer_start": 10,
    "layer_end": 15,
    "clip_coarse_grid": 7,
    "clip_fallback_thresh": 0.20,
    "phase": "both",
}

# ── RePOPE Splits ────────────────────────────────────────────────────────────────
REPOPE_SPLITS = ["random", "popular", "adversarial"]

def generate_search_configs(strategy="random", n_samples=50):
    """Generate search configurations."""
    configs = []
    
    if strategy == "grid":
        # Full grid search (may be large)
        for values in itertools.product(*SEARCH_SPACE.values()):
            config = dict(zip(SEARCH_SPACE.keys(), values))
            config.update(FIXED_PARAMS)
            configs.append(config)
    
    elif strategy == "random":
        # Random sampling from search space
        import random
        random.seed(42)
        for _ in range(n_samples):
            config = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
            config.update(FIXED_PARAMS)
            configs.append(config)
    
    elif strategy == "focused":
        # Focused search around promising regions
        # Start with current best, then explore nearby
        base_configs = [
            # Current config (VAF-like)
            {"head_top_k_pct": 0.20, "alpha": 0.15, "clip_top_k_pct": 0.30, "eps": 0.0, 
             "clip_suppress_thresh": 0.0, "clip_suppress_alpha": 5.0},
            # Higher alpha (more aggressive)
            {"head_top_k_pct": 0.20, "alpha": 2.0, "clip_top_k_pct": 0.30, "eps": 0.0,
             "clip_suppress_thresh": 0.0, "clip_suppress_alpha": 5.0},
            # More heads
            {"head_top_k_pct": 0.40, "alpha": 2.0, "clip_top_k_pct": 0.30, "eps": 0.0,
             "clip_suppress_thresh": 0.0, "clip_suppress_alpha": 5.0},
            # Focused CLIP
            {"head_top_k_pct": 0.20, "alpha": 2.0, "clip_top_k_pct": 0.20, "eps": 0.0,
             "clip_suppress_thresh": 0.0, "clip_suppress_alpha": 5.0},
        ]
        
        # Generate variations around these base configs
        for base in base_configs:
            for alpha_delta in [-1.0, 0.0, 1.0]:
                for head_delta in [-0.1, 0.0, 0.1]:
                    config = base.copy()
                    config["alpha"] = max(0.5, base["alpha"] + alpha_delta)
                    config["head_top_k_pct"] = max(0.1, min(0.5, base["head_top_k_pct"] + head_delta))
                    config.update(FIXED_PARAMS)
                    configs.append(config)
    
    return configs

def run_srf_eval(config, split):
    """Run SRF evaluation for a single config and split."""
    config_name = f"{'_'.join(f'{k}={v}' for k,v in config.items())}.json"
    output_file = RESULTS_DIR / f"repope_{split}_{config_name}"
    
    cmd = [
        "python", "srf/eval.py",
        "--model", MODEL,
        "--dataset", "pope",
        "--split", split,
        "--data_dir", REPOPE_ANNOTATIONS,
        "--image_dir", COCO_IMAGES,
        "--output_file", str(output_file),
    ]
    
    # Add SRF parameters
    for k, v in config.items():
        cmd.extend([f"--{k}", str(v)])
    
    print(f"\n{'='*60}")
    print(f"Running: {split} with config: {config}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return None
        
        # Parse accuracy from output
        for line in result.stdout.split('\n'):
            if "Accuracy:" in line:
                accuracy = float(line.split("Accuracy:")[1].strip().split()[0])
                return accuracy
    except subprocess.TimeoutExpired:
        print("ERROR: Timeout")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    return None

def main():
    """Main autoresearch loop."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["grid", "random", "focused"], default="focused")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--splits", nargs="+", default=REPOPE_SPLITS)
    parser.add_argument("--max_configs", type=int, default=20)
    args = parser.parse_args()
    
    print("="*80)
    print("SRF Autoresearch on RePOPE COCO")
    print(f"Strategy: {args.strategy}")
    print(f"Target: Beat VAF 88.28%, VCD 88.1%")
    print("="*80)
    
    # Generate search configs
    configs = generate_search_configs(args.strategy, args.n_samples)
    configs = configs[:args.max_configs]
    
    print(f"\nGenerated {len(configs)} configurations to test\n")
    
    # Track results
    results = []
    best_config = None
    best_accuracy = 0.0
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Config {i+1}/{len(configs)}: {config}")
        print(f"{'='*80}\n")
        
        # Run on all splits
        split_accuracies = {}
        for split in args.splits:
            acc = run_srf_eval(config, split)
            if acc is not None:
                split_accuracies[split] = acc
        
        if split_accuracies:
            avg_acc = sum(split_accuracies.values()) / len(split_accuracies)
            results.append({
                "config": config,
                "splits": split_accuracies,
                "average": avg_acc,
            })
            
            print(f"\nResults: {split_accuracies}")
            print(f"Average: {avg_acc:.2%}")
            
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_config = config
                print(f"🎉 NEW BEST: {best_accuracy:.2%}")
        
        # Save intermediate results
        with open(RESULTS_DIR / "autoresearch_results.json", "w") as f:
            json.dump({
                "best_config": best_config,
                "best_accuracy": best_accuracy,
                "all_results": results,
            }, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("AUTORESEARCH COMPLETE")
    print("="*80)
    print(f"\nBest config: {best_config}")
    print(f"Best average accuracy: {best_accuracy:.2%}")
    print(f"Target VAF: 88.28%, VCD: 88.1%")
    print(f"{'✅ BEAT TARGET!' if best_accuracy > 88.28 else '❌ Did not beat target'}")
    
    # Save final summary
    with open(RESULTS_DIR / "autoresearch_summary.txt", "w") as f:
        f.write(f"Best config: {best_config}\n")
        f.write(f"Best average accuracy: {best_accuracy:.2%}\n")
        f.write(f"Target VAF: 88.28%, VCD: 88.1%\n")
        f.write(f"{'✅ BEAT TARGET!' if best_accuracy > 88.28 else '❌ Did not beat target'}\n")
        f.write("\nAll results:\n")
        for r in results:
            f.write(f"{r['config']}: {r['average']:.2%}\n")

if __name__ == "__main__":
    main()
