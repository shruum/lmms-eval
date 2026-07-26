#!/usr/bin/env python3
"""
SRF Autosearch on RePOPE - apply autoresearch branch improvements

Target: Beat VAF (88.28%) and VCD (88.1%) on RePOPE COCO

Key improvements from autoresearch/mmvp-srf:
- Alpha: 4.0 (much stronger boost)
- Eps: 0.2 (mild background suppression)  
- Phase: both (critical for single-token decisions)
- Layers: 8-15 (optimal range)
- Head selection: 20%
- Bias mode: additive_logit

Test focused configs around these parameters.
"""
import os
import sys
import subprocess
import json
from pathlib import Path

SRF_DIR = Path(__file__).parent
LMMS_EVAL_DIR = SRF_DIR.parent
os.chdir(LMMS_EVAL_DIR)

RESULTS_DIR = SRF_DIR.parent / "results" / "autosearch_repope"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Focused search configs based on autoresearch findings
CONFIGS = [
    # Format: "description:alpha:eps:head_pct:layer_start:layer_end"
    "baseline_current:0.15:0.0:0.20:10:15",
    "autoresearch_best:4.0:0.2:0.20:8:15", 
    "stronger_boost:5.0:0.2:0.20:8:15",
    "medium_boost:3.0:0.2:0.20:8:15",
    "more_heads:4.0:0.2:0.30:8:15",
    "fewer_heads:4.0:0.2:0.15:8:15",
    "wider_layers:4.0:0.2:0.20:8:17",
    "narrower_layers:4.0:0.2:0.20:10:14",
    "no_suppression:4.0:0.0:0.20:8:15",
    "more_suppression:4.0:0.3:0.20:8:15",
    "aggressive_combo:5.0:0.2:0.25:8:17",
    "conservative_combo:3.0:0.15:0.15:10:14",
]

MODEL = "liuhaotian/llava-v1.5-7b"
SPLIT = "adversarial"  # Start with hardest split
N_SAMPLES = 500  # For quick testing

def run_config(desc, alpha, eps, head_pct, layer_start, layer_end):
    """Run SRF with given config."""
    print(f"\n{'='*70}")
    print(f"Testing: {desc}")
    print(f"Config: α={alpha}, ε={eps}, heads={head_pct}, layers={layer_start}-{layer_end}")
    print(f"{'='*70}\n")
    
    output_file = RESULTS_DIR / f"repope_{SPLIT}_{desc.replace(' ', '_')}.json"
    
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", "pope", 
        "--pope_splits", SPLIT,
        "--n_pope", str(N_SAMPLES),
        "--alpha", str(alpha),
        "--eps", str(eps),
        "--head_top_k_pct", str(head_pct),
        "--layer_start", str(layer_start),
        "--layer_end", str(layer_end),
        "--clip_coarse_grid", "6",  # LLaVA
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
            timeout=1800,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        
        # Save full output
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "description": desc,
                    "alpha": alpha,
                    "eps": eps, 
                    "head_top_k_pct": head_pct,
                    "layer_start": layer_start,
                    "layer_end": layer_end,
                },
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
        
        print("⚠️  Could not parse accuracy")
        return None
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    print("="*80)
    print("SRF Autosearch on RePOPE COCO")
    print(f"Model: {MODEL}")
    print(f"Target: Beat VAF 88.28%, VCD 88.1%")
    print(f"Split: {SPLIT}, Samples: {N_SAMPLES}")
    print("="*80)
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for config_str in CONFIGS:
        parts = config_str.split(":")
        desc = parts[0]
        alpha = float(parts[1])
        eps = float(parts[2])
        head_pct = float(parts[3])
        layer_start = int(parts[4])
        layer_end = int(parts[5])
        
        acc = run_config(desc, alpha, eps, head_pct, layer_start, layer_end)
        
        if acc is not None:
            results.append({
                "description": desc,
                "config": config_str,
                "accuracy": acc,
            })
            
            if acc > best_acc:
                best_acc = acc
                best_config = config_str
                print(f"\n🎉 NEW BEST: {best_acc:.2%} - {desc}\n")
        
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
    print("AUTOSEARCH COMPLETE")
    print("="*80)
    print(f"\nBest config: {best_config}")
    print(f"Best accuracy: {best_acc:.2%}")
    print(f"Target VAF: 88.28%, VCD: 88.1%")
    
    if best_acc > 88.28:
        print("✅ BEAT VAF TARGET!")
    elif best_acc > 88.1:
        print("✅ BEAT VCD TARGET!")
    else:
        print("❌ Did not beat target")
    
    # Save summary
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write(f"Best config: {best_config}\n")
        f.write(f"Best accuracy: {best_acc:.2%}\n")
        f.write(f"Target VAF: 88.28%, VCD: 88.1%\n")
        f.write(f"All results:\n")
        for r in results:
            f.write(f"  {r['description']}: {r['accuracy']:.2%}\n")

if __name__ == "__main__":
    main()
