# SRF AutoResearch Loop - LLaVA-7B + POPE Adversarial

## Goal
Automatically test SRF improvements on LLaVA-1.5-7B with POPE adversarial split (n=100 samples)

## Setup

**Model**: `llava-hf/llava-1.5-7b-hf`
**Dataset**: POPE adversarial (n=100)
**Baseline**: Current SRF config
**Metric**: Accuracy
**Time per experiment**: ~5-10 minutes

## Experimental Variations

### Option 1: Better Salience Detection

| # | Variation | Parameters |
|---|-----------|-------------|
| 1A | Multi-scale CLIP (5×5) | clip_coarse_grid=5 |
| 1B | Multi-scale CLIP (9×9) | clip_coarse_grid=9 |
| 1C | Multi-scale ensemble | Average of 5×5, 7×7, 9×9 |
| 1D | Better CLIP model | clip_model=ViT-L/14 |
| 1E | Cross-modal refinement | Use VLM attention to refine CLIP |

### Option 2: Graduated Multi-Stage Attention

| # | Variation | Parameters |
|---|-----------|-------------|
| 2A | 3-stage boost | α_high=4.0, α_mid=2.0, α_low=1.0, ε=0.2 |
| 2B | 4-stage boost | α_top5%=5.0, α_5-15%=3.0, α_15-30%=1.5, α_rest=0.5 |
| 2C | Text token suppression | Add text_token_suppression |
| 2D | System token suppression | Add sys_token_suppression |

## Implementation

### Step 1: Establish Baseline (CURRENT)

```bash
# Run current SRF on LLaVA-7B + POPE (n=100)
python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output results/baseline_llava7b/
```

**Expected baseline**: ~84% accuracy (from ALL_RESULTS_TABLE.md)

### Step 2: Create AutoResearch Script

**File**: `srf/autoresearch_llava.py`

```python
#!/usr/bin/env python3
"""
AutoResearch loop for SRF improvements on LLaVA-7B + POPE.
Tests each variation, logs results, keeps winners.
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 100
BASE_OUTPUT = "results/autoresearch_llava/"

# Experimental variations
EXPERIMENTS = {
    "baseline": {
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
    },
    # Option 1: Better Salience
    "1A_multiscale_5x5": {
        "alpha": 2.0,
        "clip_coarse_grid": 5,
        "clip_top_k_pct": 0.30,
    },
    "1B_multiscale_9x9": {
        "alpha": 2.0,
        "clip_coarse_grid": 9,
        "clip_top_k_pct": 0.30,
    },
    "1C_ensemble": {
        "alpha": 2.0,
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
        "use_ensemble": True,  # Special flag
    },
    "1D_clip_large": {
        "alpha": 2.0,
        "clip_model": "ViT-L/14",
        "clip_coarse_grid": 7,
        "clip_top_k_pct": 0.30,
    },
    # Option 2: Graduated Boost
    "2A_graduated_3stage": {
        "use_graduated": True,
        "alpha_high": 4.0,
        "alpha_mid": 2.0,
        "alpha_low": 1.0,
        "eps": 0.2,
        "graduated_stages": [0.1, 0.3, 0.5],
    },
    "2B_graduated_4stage": {
        "use_graduated": True,
        "alpha_top5": 5.0,
        "alpha_5_15": 3.0,
        "alpha_15_30": 1.5,
        "alpha_rest": 0.5,
        "eps": 0.2,
        "graduated_stages": [0.05, 0.15, 0.30, 1.0],
    },
}

def run_experiment(name: str, params: dict) -> dict:
    """Run single experiment, return results."""
    output_dir = f"{BASE_OUTPUT}{name}/"
    
    # Build command
    cmd = [
        "python", "srf/eval.py",
        "--method", "srf",
        "--model", MODEL,
        "--datasets", DATASET,
        "--pope_splits", SPLIT,
        "--n_pope", str(N_SAMPLES),
        "--output", output_dir,
    ]
    
    # Add params
    for key, value in params.items():
        if key == "use_graduated":
            continue  # Special handling in code
        cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Run experiment
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse accuracy from output
    accuracy = parse_accuracy(result.stdout)
    
    return {
        "name": name,
        "accuracy": accuracy,
        "params": params,
        "timestamp": datetime.now().isoformat(),
    }

def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    for line in output.split('\n'):
        if "accuracy" in line.lower():
            # Extract percentage
            import re
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))
    return 0.0

def main():
    """Run autoresearch loop."""
    print("SRF AutoResearch - LLaVA-7B + POPE Adversarial")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    
    results = []
    best_accuracy = 0.0
    best_experiment = None
    
    for name, params in EXPERIMENTS.items():
        result = run_experiment(name, params)
        results.append(result)
        
        # Track best
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_experiment = name
            
        print(f"Result: {result['accuracy']:.1f}%")
    
    # Save results
    output_file = f"{BASE_OUTPUT}summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "best_experiment": best_experiment,
            "best_accuracy": best_accuracy,
            "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"BEST: {best_experiment} ({best_accuracy:.1f}%)")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

### Step 3: Modify SRF Code to Support Variations

**Need to implement in `srf/srf.py`:**

1. **Multi-scale ensemble**: Compute saliency at 3 scales, average them
2. **Graduated boost**: Split tokens into stages by saliency percentile
3. **Cross-modal refinement**: Use VLM attention weights to refine CLIP

### Step 4: Create `program.md` for Claude Agent

```markdown
# SRF AutoResearch Program

## Goal
Improve SRF accuracy on LLaVA-7B + POPE adversarial through automated experimentation.

## Context
- Current baseline: ~84% accuracy
- Target: +1-3% improvement
- Time per experiment: ~5-10 minutes
- Samples: 100 (POPE adversarial)

## Your Role
You are an autonomous research agent. You will:

1. **Implement variations** in `srf/srf.py`:
   - Multi-scale CLIP ensemble
   - Graduated multi-stage attention
   - Cross-modal refinement

2. **Run experiments** using `python srf/autoresearch_llava.py`

3. **Analyze results** and suggest next improvements

4. **Keep winning changes** - if accuracy improves > 0.5%, update baseline

## Constraints
- Only modify `srf/srf.py` and `srf/saliency/clip_salience.py`
- Each experiment must complete in < 10 minutes
- Do not change evaluation logic or dataset loading
- Log all experiments with accuracy

## Success Criteria
- Find configuration that achieves > 85% accuracy
- Must beat baseline by > 1%
- Must be reproducible (same parameters = same results)
```

## Execution Plan

### Phase 1: Manual Verification (Day 1)

1. **Establish baseline**
   ```bash
   python srf/eval.py --method srf --model llava-hf/llava-1.5-7b-hf \
       --datasets pope --pope_splits adversarial --n_pope 100 \
       --output results/baseline_llava7b/
   ```

2. **Test one variation manually** (e.g., 5×5 grid)
   - Verify it works
   - Check accuracy
   - Time the experiment

### Phase 2: Implement Variations (Day 2-3)

**Implement in order:**

1. **Multi-scale CLIP** (easiest)
   - Modify `clip_salience.py` to compute multiple grids
   - Average saliency maps
   - Test 5×5, 7×7, 9×9 individually first

2. **Graduated boost** (moderate)
   - Modify `prepare_sample()` in `srf.py`
   - Split tokens into saliency buckets
   - Apply different α per bucket

3. **Cross-modal refinement** (complex)
   - Get VLM attention from first forward pass
   - Use it to weight CLIP saliency
   - Need to understand attention rollout

### Phase 3: Automated Loop (Day 4+)

1. **Implement autoresearch script**
2. **Run overnight** (~50-100 experiments)
3. **Analyze results** in morning
4. **Keep winners** as new baseline
5. **Repeat**

## Tracking Results

**Log format:**
```json
{
  "experiment_id": "1A_multiscale_5x5",
  "timestamp": "2026-04-28T20:15:30",
  "accuracy": 85.2,
  "delta_vs_baseline": +1.2,
  "parameters": {...},
  "status": "winner"
}
```

**Winner criteria:** 
- Δ > +0.5% → Keep as new baseline
- Δ > +1.0% → Major discovery, investigate further
- Δ < +0.5% → Discard

## Next Steps

**Today:**
1. Run baseline experiment
2. Implement multi-scale CLIP
3. Test manually

**Tomorrow:**
4. Implement graduated boost
5. Set up autoresearch script
6. Start automated loop

**This weekend:**
7. Let it run overnight
8. Analyze 50-100 experiments
9. Write up findings
