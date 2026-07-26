#!/usr/bin/env python3
"""
Validation test for SRF improvements on RePOPE - 100 samples only.

Tests 5 configs before committing to full autosearch:
1. Baseline (current weak config)
2. Autoresearch best (α=4.0, ε=0.2)
3. v3 gate only
4. Best combo
5. Strong boost

Goal: If any config beats 82% on 100 samples, proceed to full autosearch.
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

SRF_DIR = Path(__file__).parent
LMMS_EVAL_DIR = SRF_DIR.parent
os.chdir(LMMS_EVAL_DIR)

RESULTS_DIR = SRF_DIR.parent / "results" / "validation_repope"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "liuhaotian/llava-v1.5-7b"
SPLIT = "adversarial"
N_SAMPLES = 100  # Small validation test
GPU_ID = 0  # Use GPU 0 for validation

# Test configurations
CONFIGS = [
    {
        "name": "baseline_current",
        "alpha": 0.15,
        "eps": 0.0,
        "enable_v3_gate": False,
        "layer_start": 10,
        "layer_end": 15,
        "head_top_k_pct": 0.20,
        "description": "Current weak config (VAF-like)"
    },
    {
        "name": "autoresearch_best",
        "alpha": 4.0,
        "eps": 0.2,
        "enable_v3_gate": False,
        "layer_start": 8,
        "layer_end": 15,
        "head_top_k_pct": 0.20,
        "description": "Autoresearch best (MMVP 70.67% accuracy)"
    },
    {
        "name": "v3_gate_only",
        "alpha": 0.15,
        "eps": 0.0,
        "enable_v3_gate": True,
        "layer_start": 10,
        "layer_end": 15,
        "head_top_k_pct": 0.20,
        "description": "v3 gate with weak boost"
    },
    {
        "name": "best_combo",
        "alpha": 4.0,
        "eps": 0.2,
        "enable_v3_gate": True,
        "layer_start": 8,
        "layer_end": 15,
        "head_top_k_pct": 0.20,
        "description": "Autoresearch best + v3 gate"
    },
    {
        "name": "strong_boost",
        "alpha": 5.0,
        "eps": 0.2,
        "enable_v3_gate": True,
        "layer_start": 8,
        "layer_end": 15,
        "head_top_k_pct": 0.20,
        "description": "Strong boost + v3 gate"
    },
]

def run_validation(config):
    """Run SRF validation with given config."""
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"Config: {config}")
    print(f"Split: {SPLIT}, Samples: {N_SAMPLES}")
    print(f"{'='*70}\n")
    
    output_file = RESULTS_DIR / f"validation_{config['name']}.json"
    
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "pope",
        "--pope_splits", SPLIT,
        "--n_pope", str(N_SAMPLES),
        "--alpha", str(config["alpha"]),
        "--eps", str(config["eps"]),
        "--layer_start", str(config["layer_start"]),
        "--layer_end", str(config["layer_end"]),
        "--head_top_k_pct", str(config["head_top_k_pct"]),
        "--clip_coarse_grid", "6",
        "--clip_top_k_pct", "0.30",
        "--clip_fallback_thresh", "0.20",
        "--phase", "both",
        "--bias_mode", "additive_logit",
        "--output", str(RESULTS_DIR),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU_ID)}
        )
        
        # Save full output
        with open(output_file, "w") as f:
            json.dump({
                "config": config,
                "split": SPLIT,
                "n_samples": N_SAMPLES,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }, f, indent=2)
        
        # Try to extract accuracy
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

def main():
    print("="*80)
    print("SRF Validation Test on RePOPE - 100 samples")
    print(f"Model: {MODEL}")
    print(f"GPU: {GPU_ID}")
    print(f"Target: If any config beats 82%, proceed to full autosearch")
    print("="*80)
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for config in CONFIGS:
        acc = run_validation(config)
        
        if acc is not None:
            results.append({
                **config,
                "accuracy": acc,
            })
            
            if acc > best_acc:
                best_acc = acc
                best_config = config
                print(f"\n🎉 NEW BEST: {best_acc:.2%} - {config['name']}\n")
        
        # Save progress
        with open(RESULTS_DIR / "validation_progress.json", "w") as f:
            json.dump({
                "best_config": best_config,
                "best_accuracy": best_acc,
                "all_results": results,
                "threshold": 82.0,
                "proceed_to_full": best_acc > 82.0,
            }, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nBest config: {best_config['name'] if best_config else 'None'}")
    print(f"Best accuracy: {best_acc:.2%}")
    print(f"Threshold: 82.0%")
    
    if best_acc > 82.0:
        print("✅ PROCEED TO FULL AUTOSEARCH!")
        print("   Config beat threshold - improvements working on RePOPE")
    elif best_acc > 80.40:
        print("⚠️  ABOVE BASELINE - Consider limited autosearch")
        print(f"   Current baseline: 80.40%, Got: {best_acc:.2%}")
    else:
        print("❌ DID NOT BEAT BASELINE")
        print(f"   Current baseline: 80.40%, Got: {best_acc:.2%}")
        print("   RePOPE is challenging - may need different approach")
    
    # Save summary
    with open(RESULTS_DIR / "validation_summary.txt", "w") as f:
        f.write(f"Validation Test Results\n")
        f.write(f"Best config: {best_config['name'] if best_config else 'None'}\n")
        f.write(f"Best accuracy: {best_acc:.2%}\n")
        f.write(f"Threshold: 82.0%\n")
        f.write(f"Decision: {'PROCEED' if best_acc > 82.0 else 'DO NOT PROCEED'}\n\n")
        f.write(f"All results:\n")
        for r in results:
            f.write(f"  {r['name']}: {r['accuracy']:.2%} - {r['description']}\n")

if __name__ == "__main__":
    main()
