#!/usr/bin/env python3
"""
SRF AutoResearch - Prepare

Fixed utilities for dataset loading, evaluation, and runtime helpers.
DO NOT MODIFY - this file contains stable infrastructure.

Based on https://github.com/karpathy/autoresearch
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Constants ────────────────────────────────────────────────────────────────

# GPU Configuration
DEFAULT_GPU = 1
GPU_MEMORY = 49140  # MB for RTX A6000

# Experiment Configuration
DEFAULT_N_SAMPLES = 100  # Quick experiments for autoresearch
TIMEOUT_MINUTES = 10     # Max time per experiment

# Dataset Paths
POPE_DATA = "lmms-lab/POPE"
MMVP_DATA = "csv"  # Will be loaded from HF_HOME

# Baseline Accuracy (for comparison)
BASELINE_ACCURACY = {
    "llava-hf/llava-1.5-7b-hf": {
        "pope_adversarial": 84.0,
        "mmvp": 67.0,
        "mme": 69.5,
    },
}


# ── Dataset Loading ────────────────────────────────────────────────────────────

def load_pope_adversarial(n_samples: int = DEFAULT_N_SAMPLES, seed: int = 42) -> List[Dict]:
    """
    Load POPE adversarial split for quick experiments.

    Returns list of samples with keys: image, question, answer, category
    """
    from datasets import load_dataset as hf_load
    from collections import defaultdict

    print(f"  [prepare] Loading POPE adversarial (n={n_samples})...")

    ds = hf_load(POPE_DATA, split="test")

    # Filter adversarial
    samples = []
    for row in ds:
        cat = str(row.get("category", "")).strip().lower()
        if cat != "adversarial":
            continue

        samples.append({
            "image": row["image"].convert("RGB"),
            "question": str(row.get("question", "")).strip() + "\nAnswer with Yes or No only.",
            "answer": str(row.get("answer", "")).strip(),
            "category": cat,
        })

        if len(samples) >= n_samples:
            break

    print(f"  [prepare] Loaded {len(samples)} samples")
    return samples


def load_mmvp(n_samples: int = DEFAULT_N_SAMPLES, seed: int = 42) -> List[Dict]:
    """Load MMVP dataset for testing."""
    # TODO: Implement MMVP loading
    raise NotImplementedError("MMVP loading not yet implemented")


# ── Evaluation Utilities ─────────────────────────────────────────────────────

def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate accuracy for Yes/No questions.

    Args:
        predictions: List of model predictions ("Yes" or "No")
        ground_truth: List of ground truth answers

    Returns:
        Accuracy as percentage (0-100)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(ground_truth)}")

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip().lower() == g.strip().lower())
    accuracy = (correct / len(predictions)) * 100

    return accuracy


def parse_model_output(output: str) -> str:
    """
    Parse model output to extract Yes/No answer.

    Handles various formats:
    - "Yes"
    - "Answer: Yes"
    - "The answer is Yes."
    """
    output = output.strip().lower()

    # Look for yes/no in output
    if "yes" in output and "no" not in output:
        return "Yes"
    elif "no" in output and "yes" not in output:
        return "No"
    else:
        # Fallback: return first word
        first_word = output.split()[0]
        if first_word in ["yes", "no"]:
            return first_word.capitalize()
        else:
            return "Yes"  # Default fallback


# ── Runtime Utilities ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_id() -> int:
    """Get GPU ID from environment or default."""
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", str(DEFAULT_GPU)))


def check_gpu_memory(gpu_id: int = None) -> Dict[str, Any]:
    """Check available GPU memory."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}

    if gpu_id is None:
        gpu_id = get_gpu_id()

    try:
        props = torch.cuda.get_device_properties(gpu_id)
        free = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3  # GB

        return {
            "available": True,
            "gpu_id": gpu_id,
            "name": torch.cuda.get_device_name(gpu_id),
            "total_memory_gb": props.total_memory / 1024**3,
            "free_memory_gb": free,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def time_experiment(start_time, max_minutes: int = TIMEOUT_MINUTES) -> bool:
    """
    Check if experiment has exceeded time budget.

    Returns True if should stop, False otherwise.
    """
    import time
    elapsed = (time.time() - start_time) / 60
    return elapsed >= max_minutes


# ── Logging ────────────────────────────────────────────────────────────────────

def log_experiment_start(config: Dict, output_dir: str):
    """Log experiment start configuration."""
    from datetime import datetime

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = f"{output_dir}/config.json"
    config["timestamp"] = datetime.now().isoformat()
    config["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import json
    with open(log_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  [prepare] Logged config to {log_file}")


def log_experiment_result(result: Dict, output_dir: str):
    """Log experiment result."""
    from datetime import datetime

    result_file = f"{output_dir}/result.json"
    result["timestamp"] = datetime.now().isoformat()

    import json
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [prepare] Logged result to {result_file}")


# ── Git Integration ───────────────────────────────────────────────────────────

def get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return "unknown"


def get_git_diff() -> str:
    """Get git diff of srf.py (to see what agent changed)."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "diff", "srf/srf.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {e}"


# ── Main (for testing) ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SRF AutoResearch - Prepare")
    print("=" * 60)

    # Test dataset loading
    samples = load_pope_adversarial(n_samples=10)
    print(f"\nLoaded {len(samples)} samples")
    if samples:
        print(f"Sample: {samples[0]['question'][:50]}...")

    # Check GPU
    gpu_info = check_gpu_memory()
    print(f"\nGPU: {gpu_info}")
