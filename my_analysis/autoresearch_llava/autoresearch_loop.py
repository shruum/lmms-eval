#!/usr/bin/env python3
"""
Autoresearch loop for POPE × LLaVA-1.5-7B using SRF (ClearSight VAF).

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
# PARAMETER SWEEP RANGES (ClearSight VAF parameters)
# =============================================================================
PARAM_RANGES = {
    # Layer ranges (ClearSight: 9-14 works well for LLaVA)
    "layer_start": [8, 9, 10],
    "layer_end": [14, 15, 16],

    # VAF parameters (ClearSight: enh=1.15, sup=0.95)
    "enh_para": [1.05, 1.10, 1.15, 1.20, 1.25, 1.30],
    "sup_para": [0.85, 0.90, 0.95, 1.00],

    # CLIP saliency (for future per-token scaling)
    "clip_top_k_pct": [0.20, 0.30, 0.40],
    "clip_coarse_grid": [5, 7, 9],
}

# =============================================================================
# CONFIG DATA STRUCT
# =============================================================================
@dataclass
class Config:
    layer_start: int
    layer_end: int
    enh_para: float
    sup_para: float
    clip_top_k_pct: float = 0.20
    clip_coarse_grid: int = 9

    def __str__(self):
        return (f"L{self.layer_start}-{self.layer_end}_"
                f"enh{self.enh_para}_sup{self.sup_para}")

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
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    try:
                        acc = float(parts[1])
                        results.append((acc, parts[2], parts[3] if len(parts) > 3 else ""))
                    except ValueError:
                        pass
    return results

def save_result(config: Config, accuracy: float, description: str, status: str = "keep"):
    """Save result to results.tsv."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        text=True).strip()
    except:
        commit = "unknown"
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{accuracy:.4f}\t{status}\t{description}\n")
    print(f"  [{status.upper()}] {accuracy:.4f} — {description}")

def update_srf_config(config: Config):
    """Update srf.py with the given config."""
    srf_path = SCRIPT_DIR / "srf.py"
    with open(srf_path) as f:
        content = f.read()

    # Update layer range
    content = content.replace(
        f'"layer_start": 9',
        f'"layer_start": {config.layer_start}'
    )
    content = content.replace(
        f'"layer_end": 14',
        f'"layer_end": {config.layer_end}'
    )

    # Update VAF parameters
    content = content.replace(
        f'"enh_para": 1.15',
        f'"enh_para": {config.enh_para}'
    )
    content = content.replace(
        f'"sup_para": 0.95',
        f'"sup_para": {config.sup_para}'
    )

    # Update CLIP saliency
    content = content.replace(
        f'"clip_top_k_pct": 0.2',
        f'"clip_top_k_pct": {config.clip_top_k_pct}'
    )
    content = content.replace(
        f'"clip_coarse_grid": 9',
        f'"clip_coarse_grid": {config.clip_coarse_grid}'
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
            enh_para=random.choice(PARAM_RANGES["enh_para"]),
            sup_para=random.choice(PARAM_RANGES["sup_para"]),
            clip_top_k_pct=random.choice(PARAM_RANGES["clip_top_k_pct"]),
            clip_coarse_grid=random.choice(PARAM_RANGES["clip_coarse_grid"]),
        )
        # Ensure layer_end > layer_start
        if config.layer_end <= config.layer_start:
            config.layer_end = config.layer_start + 1
        configs.append(config)
    return configs

def main():
    """Run autoresearch loop."""
    print("="*60)
    print("AUTORESEARCH LOOP: POPE × LLaVA-1.5-7B (ClearSight VAF)")
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
