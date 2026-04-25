#!/usr/bin/env python3
"""
Autoresearch loop for POPE × LLaVA-1.5-7B using SRF.

Follows Karpathy's autoresearch principles:
- Single file modification (srf.py)
- Fixed time budget (4 hours)
- Autonomous iteration
"""
from __future__ import annotations
import os, pathlib, subprocess, sys, time, json, math, random
from dataclasses import dataclass, asdict
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "results.tsv"
CONFIG_FILE = SCRIPT_DIR / "best_config.json"

BASELINE_ACC = 0.83  # Measured baseline accuracy

# =============================================================================
# PARAMETER SWEEP RANGES
# =============================================================================
PARAM_RANGES = {
    # Layer ranges (ClearSight: 9-14 works well for LLaVA)
    "layer_start": [8, 9, 10],
    "layer_end": [14, 15, 16],

    # Bias mode
    "bias_mode": ["additive_logit", "prob_interp", "prob_scale"],

    # Boost strength
    "boost_alpha": [1.0, 1.5, 2.0, 2.5, 3.0],

    # CLIP saliency
    "clip_top_k_pct": [0.20, 0.30, 0.40],
    "clip_coarse_grid": [5, 7, 9],

    # System suppression
    "sys_beta": [0.0, 0.05, 0.10, 0.15],
}

# =============================================================================
# CONFIG DATA STRUCT
# =============================================================================
@dataclass
class Config:
    layer_start: int
    layer_end: int
    bias_mode: str
    boost_alpha: float
    clip_top_k_pct: float
    clip_coarse_grid: int
    sys_beta: float
    head_top_k_pct: float = 0.0  # Apply to all heads for now
    srf_background_eps: float = 0.0

    def __str__(self):
        return (f"L{self.layer_start}-{self.layer_end}_{self.bias_mode}_"
                f"a{self.boost_alpha}_top{int(self.clip_top_k_pct*100)}_"
                f"grid{self.clip_coarse_grid}_beta{self.sys_beta}")

    def to_dict(self):
        return asdict(self)

# =============================================================================
# AUTORESEARCH LOOP
# =============================================================================
def load_results() -> List[tuple]:
    """Load existing results from results.tsv."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.strip() and not line.startswith("commit"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    acc = float(parts[1])
                    results.append(acc)
    return results

def save_result(config: Config, accuracy: float, description: str, status: str = "keep"):
    """Save result to results.tsv."""
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    text=True).strip()
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{accuracy:.4f}\t{status}\t{description}\n")
    print(f"  [{status.upper()}] {accuracy:.4f} — {description}")

def update_srf_config(config: Config):
    """Update srf.py with the given config."""
    srf_path = SCRIPT_DIR / "srf.py"
    with open(srf_path) as f:
        content = f.read()

    # Update SALIENCY section
    content = content.replace(
        '"clip_top_k_pct": 0.30',
        f'"clip_top_k_pct": {config.clip_top_k_pct}'
    )
    content = content.replace(
        '"clip_coarse_grid": 7',
        f'"clip_coarse_grid": {config.clip_coarse_grid}'
    )

    # Update BIAS section
    content = content.replace(
        '"layer_start": 8',
        f'"layer_start": {config.layer_start}'
    )
    content = content.replace(
        '"layer_end": 15',
        f'"layer_end": {config.layer_end}'
    )
    content = content.replace(
        '"bias_mode": "additive_logit"',
        f'"bias_mode": "{config.bias_mode}"'
    )
    content = content.replace(
        '"boost_alpha": 2.0',
        f'"boost_alpha": {config.boost_alpha}'
    )
    content = content.replace(
        '"sys_beta": 0.10',
        f'"sys_beta": {config.sys_beta}'
    )

    with open(srf_path, "w") as f:
        f.write(content)

def run_evaluation(config: Config) -> float:
    """Run POPE evaluation with the given config."""
    print(f"\n  Testing: {config}")
    update_srf_config(config)

    # Run pope_eval.py
    python_path = os.path.expandvars("$HOME/miniconda3/envs/mllm/bin/python")
    result = subprocess.run(
        [python_path, "pope_eval.py"],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"}
    )

    # Parse accuracy from output
    for line in result.stdout.split("\n"):
        if "POPE accuracy:" in line:
            acc_str = line.split("POPE accuracy:")[1].split()[0]
            return float(acc_str)

    raise ValueError(f"Could not parse accuracy from output:\n{result.stdout}")

def generate_configs(n_configs: int = 100) -> List[Config]:
    """Generate random configs to try."""
    configs = []
    for _ in range(n_configs):
        config = Config(
            layer_start=random.choice(PARAM_RANGES["layer_start"]),
            layer_end=random.choice(PARAM_RANGES["layer_end"]),
            bias_mode=random.choice(PARAM_RANGES["bias_mode"]),
            boost_alpha=random.choice(PARAM_RANGES["boost_alpha"]),
            clip_top_k_pct=random.choice(PARAM_RANGES["clip_top_k_pct"]),
            clip_coarse_grid=random.choice(PARAM_RANGES["clip_coarse_grid"]),
            sys_beta=random.choice(PARAM_RANGES["sys_beta"]),
        )
        # Ensure layer_end > layer_start
        if config.layer_end <= config.layer_start:
            config.layer_end = config.layer_start + 1
        configs.append(config)
    return configs

def main():
    """Run autoresearch loop."""
    print("="*60)
    print("AUTORESEARCH LOOP: POPE × LLaVA-1.5-7B")
    print(f"Baseline: {BASELINE_ACC:.4f}")
    print(f"GPU: 2")
    print(f"Time budget: 4 hours")
    print("="*60)

    start_time = time.time()
    time_budget = 4 * 3600  # 4 hours
    best_acc = BASELINE_ACC
    best_config = None
    experiment_count = 0

    # Generate initial batch of configs
    configs = generate_configs(50)

    for config in configs:
        # Check time budget
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 600:  # Leave 10 min buffer
            print(f"\n  Time budget exceeded ({elapsed/3600:.1f}h). Stopping.")
            break

        try:
            acc = run_evaluation(config)
            experiment_count += 1

            # Determine status
            if acc > best_acc:
                status = "NEW_BEST"
                best_acc = acc
                best_config = config
            elif acc > BASELINE_ACC:
                status = "keep"
            else:
                status = "discard"

            # Save result
            description = str(config)
            save_result(config, acc, description, status)

            # Print progress
            print(f"  Progress: {experiment_count} experiments | "
                  f"Best: {best_acc:.4f} | "
                  f"Time: {elapsed/60:.0f}min")

        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {config}")
            save_result(config, 0.0, f"TIMEOUT: {config}", "crash")
        except Exception as e:
            print(f"  [ERROR] {e}")
            save_result(config, 0.0, f"ERROR: {e}", "crash")

    # Save best config
    if best_config:
        with open(CONFIG_FILE, "w") as f:
            json.dump(best_config.to_dict(), f, indent=2)
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Baseline: {BASELINE_ACC:.4f}")
        print(f"  Best:     {best_acc:.4f} ({(best_acc-BASELINE_ACC)*100:+.2f}%)")
        print(f"  Config:   {best_config}")
        print(f"  Experiments: {experiment_count}")
        print(f"  Time:    {time.time()-start_time:.0f}s")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
