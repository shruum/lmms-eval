#!/usr/bin/env python3
"""
Advanced CLIP and attention experiments - temporarily modifies code to test improvements.
Uses safe patching and reverts changes after each experiment.
"""
import subprocess
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Configuration
MODEL = "llava-hf/llava-1.5-7b-hf"
DATASET = "pope"
SPLIT = "adversarial"
N_SAMPLES = 50
BASE_OUTPUT = "results/autoresearch_advanced_cl/"


def backup_files():
    """Backup original files before modification."""
    backups = {}
    files_to_backup = [
        "srf/saliency/clip_salience.py",
        "my_analysis/llava_attn_patch.py",
    ]
    for file in files_to_backup:
        if os.path.exists(file):
            backup_name = f"{file}.backup"
            shutil.copy(file, backup_name)
            backups[file] = backup_name
    return backups


def restore_files(backups):
    """Restore original files."""
    for original, backup in backups.items():
        if os.path.exists(backup):
            shutil.copy(backup, original)
            os.remove(backup)


def modify_clip_salience(use_large_model: bool = False, use_hidden_layers: bool = False):
    """Temporarily modify clip_salience.py to use advanced features."""
    clip_file = "srf/saliency/clip_salience.py"

    # Read original
    with open(clip_file, 'r') as f:
        content = f.read()

    # Modify model name
    if use_large_model:
        content = content.replace(
            '_CLIP_DEFAULT_MODEL  = "openai/clip-vit-base-patch32"',
            '_CLIP_DEFAULT_MODEL  = "openai/clip-vit-large-patch14"'
        )

    # Modify to use hidden layers if requested
    if use_hidden_layers:
        # Add import for hidden layer features
        if "from transformers import CLIPModel, CLIPProcessor" not in content:
            content = content.replace(
                "from transformers import CLIPModel, CLIPProcessor",
                "from transformers import CLIPModel, CLIPProcessor\n    CLIPModel.get_image_features = lambda self, **kwargs: self.vision_model(**kwargs).pooler_output"
            )

    # Write modified version
    with open(clip_file, 'w') as f:
        f.write(content)


def modify_attention_patch(strategy: str = "multiplicative"):
    """Temporarily modify attention patch to use different boosting strategy."""
    patch_file = "my_analysis/llava_attn_patch.py"

    with open(patch_file, 'r') as f:
        content = f.read()

    # The llava patch uses multiplicative scaling by default
    # For additive strategy, we'd need to modify the patched_softmax function
    if strategy == "additive":
        # This is a placeholder - actual implementation would be more complex
        pass
    elif strategy == "temperature":
        # Modify to use temperature scaling
        content = content.replace(
            'attn_weights[:, :, :, img_start : img_end + 1] *= enh_para',
            'temp = 1.0 / enh_para\n                attn_weights[:, :, :, img_start : img_end + 1] = F.softmax(torch.log(attn_weights[:, :, :, img_start : img_end + 1] + 1e-8) / temp, dim=-1)'
        )

    with open(patch_file, 'w') as f:
        f.write(content)


def run_advanced_experiment(name: str, config: dict, gpu_id: int = 1) -> dict:
    """Run an experiment with temporary code modifications."""
    output_dir = f"{BASE_OUTPUT}{name}/"

    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Description: {config['description']}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*70}")

    # Backup files
    backups = backup_files()

    try:
        # Apply modifications
        if config.get("use_large_clip"):
            print(f"  → Using large CLIP model (ViT-L/14)")
            modify_clip_salience(use_large_model=True)

        if config.get("use_hidden_layers"):
            print(f"  → Using hidden layer features")
            modify_clip_salience(use_hidden_layers=True)

        if config.get("attention_strategy"):
            print(f"  → Using attention strategy: {config['attention_strategy']}")
            modify_attention_patch(config["attention_strategy"])

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

        # Add standard parameters
        for key, value in config.items():
            if key in ["description", "use_large_clip", "use_hidden_layers", "attention_strategy"]:
                continue
            cmd.extend([f"--{key}", str(value)])

        # Set GPU and run
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Parse accuracy
        accuracy = parse_accuracy(result.stdout)

        # Save output
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_dir}output.txt"
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

        return {
            "name": name,
            "accuracy": accuracy,
            "config": config["description"],
            "timestamp": datetime.now().isoformat(),
            "output_file": output_file,
        }

    finally:
        # Always restore original files
        restore_files(backups)
        print(f"  → Restored original files")


def parse_accuracy(output: str) -> float:
    """Extract accuracy from eval output."""
    for line in output.split('\n'):
        if "SRF:" in line and "acc=" in line:
            import re
            match = re.search(r'acc=(\d+\.?\d*)%', line)
            if match:
                return float(match.group(1))

    import re
    matches = re.findall(r'(\d+\.?\d*)%', output)
    if matches:
        return float(matches[-1])

    return 0.0


# Advanced experiments that require code modification
ADVANCED_CL_EXPERIMENTS = {
    "baseline": {
        "description": "Original baseline (ViT-B/32, multiplicative)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
    },

    # Large CLIP model experiments
    "clip_large": {
        "description": "Large CLIP model (ViT-L/14)",
        "alpha": 2.0,
        "eps": 0.0,
        "clip_coarse_grid": 7,
        "use_large_clip": True,
    },

    "clip_large_strong": {
        "description": "Large CLIP + strong boost (ViT-L/14, α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 7,
        "use_large_clip": True,
    },

    # Parameter experiments with large CLIP
    "clip_large_fine": {
        "description": "Large CLIP + fine grid (ViT-L/14, 5×5, α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 5,
        "use_large_clip": True,
    },

    "clip_large_coarse": {
        "description": "Large CLIP + coarse grid (ViT-L/14, 9×9, α=4.0)",
        "alpha": 4.0,
        "eps": 0.2,
        "clip_coarse_grid": 9,
        "use_large_clip": True,
    },

    # Strong boost combinations
    "clip_large_stronger": {
        "description": "Large CLIP + stronger boost (ViT-L/14, α=6.0)",
        "alpha": 6.0,
        "eps": 0.3,
        "clip_coarse_grid": 7,
        "use_large_clip": True,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced CLIP + Attention Experiments")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick subset (4 experiments)")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    print("Advanced CLIP + Attention Experiments")
    print(f"Model: {MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Total experiments: {len(ADVANCED_CL_EXPERIMENTS)}")

    if args.baseline_only:
        print("\n🎯 Running BASELINE only...")
        result = run_advanced_experiment("baseline", ADVANCED_CL_EXPERIMENTS["baseline"], args.gpu)
        print(f"\n✓ Baseline: {result['accuracy']:.1f}%")
        return

    if args.experiment:
        if args.experiment in ADVANCED_CL_EXPERIMENTS:
            print(f"\n🎯 Running: {args.experiment}")
            result = run_advanced_experiment(args.experiment, ADVANCED_CL_EXPERIMENTS[args.experiment], args.gpu)
            print(f"\n✓ Result: {result['accuracy']:.1f}%")
        else:
            print(f"❌ Unknown experiment: {args.experiment}")
            print(f"Available: {list(ADVANCED_CL_EXPERIMENTS.keys())}")
        return

    # Select experiments
    if args.quick:
        experiments_to_run = {
            "baseline": ADVANCED_CL_EXPERIMENTS["baseline"],
            "clip_large": ADVANCED_CL_EXPERIMENTS["clip_large"],
            "clip_large_strong": ADVANCED_CL_EXPERIMENTS["clip_large_strong"],
            "clip_large_fine": ADVANCED_CL_EXPERIMENTS["clip_large_fine"],
        }
        print(f"\n🚀 Running QUICK subset ({len(experiments_to_run)} experiments)...")
    else:
        experiments_to_run = ADVANCED_CL_EXPERIMENTS
        print(f"\n🚀 Running ALL {len(experiments_to_run)} experiments...")

    results = []
    best_accuracy = 0.0
    best_experiment = None
    baseline_accuracy = 0.0

    for i, (name, config) in enumerate(experiments_to_run.items()):
        gpu = 1 if i % 2 == 0 else 2

        result = run_advanced_experiment(name, config, gpu)
        results.append(result)

        if name == "baseline":
            baseline_accuracy = result["accuracy"]

        delta = result["accuracy"] - baseline_accuracy
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_experiment = name

        status = "🏆" if delta > 1.0 else "✓" if delta > 0 else "✗"
        print(f"{status} {result['name']}: {result['accuracy']:.1f}% (Δ{delta:+.1f})")

    # Save results
    import json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"{BASE_OUTPUT}advanced_cl_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "baseline": baseline_accuracy,
            "best_experiment": best_experiment,
            "best_accuracy": best_accuracy,
            "best_delta": best_accuracy - baseline_accuracy,
            "all_results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏆 BASELINE: {baseline_accuracy:.1f}%")
    print(f"🏆 BEST: {best_experiment} ({best_accuracy:.1f}%, Δ{best_accuracy - baseline_accuracy:+.1f})")
    print(f"📊 Results: {summary_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
