#!/usr/bin/env python3
"""
Analyze grid search results and find best hyperparameters for LLaVA.

Usage:
    python analyze_grid_search.py --output_dir grid_search_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys


def load_results(output_dir: Path) -> List[Dict]:
    """Load all results from grid search."""

    results = []
    result_files = list(output_dir.glob("results_gpu*_final.json"))

    if not result_files:
        print(f"No result files found in {output_dir}")
        print(f"Looking for: results_gpu*_final.json")
        print(f"Available files: {list(output_dir.glob('*.json'))}")
        return []

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)
            results.extend(data.get("experiments", []))

    print(f"Loaded {len(results)} experiments from {len(result_files)} GPU(s)")
    return results


def extract_metrics(exp: Dict) -> Dict[str, float]:
    """Extract key metrics from an experiment."""

    if exp.get("status") != "success" or "results" not in exp:
        return None

    results = exp["results"]

    # Extract SRF results if available
    if "srf" not in results:
        return None

    srf_results = results["srf"]

    # Get average accuracy across splits
    baseline_acc = srf_results.get("baseline_acc", 0.0)
    srf_acc = srf_results.get("srf_acc", 0.0)
    improvement = srf_acc - baseline_acc

    return {
        "exp_id": exp["exp_id"],
        "layer_start": exp["layer_start"],
        "layer_end": exp["layer_end"],
        "boost_alpha": exp["boost_alpha"],
        "suppress_alpha": exp["suppress_alpha"],
        "head_top_k_pct": exp["head_top_k_pct"],
        "saliency_method": exp["saliency_config"]["method"],
        "baseline_acc": baseline_acc,
        "srf_acc": srf_acc,
        "improvement": improvement,
        "adversarial_acc": srf_results.get("adversarial_acc", 0.0),
        "popular_acc": srf_results.get("popular_acc", 0.0),
    }


def print_top_results(metrics_list: List[Dict], top_n: int = 10):
    """Print top performing configurations."""

    # Sort by improvement
    sorted_metrics = sorted(metrics_list, key=lambda x: x["improvement"], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP {top_n} CONFIGURATIONS BY IMPROVEMENT")
    print(f"{'='*100}")

    for i, metrics in enumerate(sorted_metrics[:top_n], 1):
        print(f"\n#{i} Experiment {metrics['exp_id']} — Improvement: {metrics['improvement']:+.4f}")
        print(f"  Layers: {metrics['layer_start']}-{metrics['layer_end']} | "
              f"Boost: {metrics['boost_alpha']} | Suppress: {metrics['suppress_alpha']} | "
              f"Head top-k: {metrics['head_top_k_pct']}")
        print(f"  Saliency: {metrics['saliency_method']}")
        print(f"  Baseline: {metrics['baseline_acc']:.4f} → SRF: {metrics['srf_acc']:.4f}")
        print(f"  Adversarial: {metrics['adversarial_acc']:.4f} | Popular: {metrics['popular_acc']:.4f}")


def analyze_parameter_effects(metrics_list: List[Dict]):
    """Analyze how individual parameters affect performance."""

    print(f"\n{'='*100}")
    print("PARAMETER EFFECTS ANALYSIS")
    print(f"{'='*100}")

    # Layer ranges
    print(f"\n📍 Layer Ranges:")
    layer_ranges = {}
    for m in metrics_list:
        key = f"{m['layer_start']}-{m['layer_end']}"
        if key not in layer_ranges:
            layer_ranges[key] = []
        layer_ranges[key].append(m['improvement'])

    for key, improvements in sorted(layer_ranges.items()):
        avg_improvement = sum(improvements) / len(improvements)
        print(f"  Layers {key}: {avg_improvement:+.4f} (n={len(improvements)})")

    # Boost/suppress ratios
    print(f"\n📍 Boost/Suppress Ratios:")
    ratios = {}
    for m in metrics_list:
        key = f"{m['boost_alpha']}/{m['suppress_alpha']}"
        if key not in ratios:
            ratios[key] = []
        ratios[key].append(m['improvement'])

    for key, improvements in sorted(ratios.items()):
        avg_improvement = sum(improvements) / len(improvements)
        print(f"  {key}: {avg_improvement:+.4f} (n={len(improvements)})")

    # Saliency methods
    print(f"\n📍 Saliency Methods:")
    methods = {}
    for m in metrics_list:
        method = m['saliency_method']
        if method not in methods:
            methods[method] = []
        methods[method].append(m['improvement'])

    for method, improvements in sorted(methods.items()):
        avg_improvement = sum(improvements) / len(improvements)
        print(f"  {method}: {avg_improvement:+.4f} (n={len(improvements)})")


def find_best_per_method(metrics_list: List[Dict]):
    """Find best configuration for each saliency method."""

    print(f"\n{'='*100}")
    print("BEST CONFIGURATION PER SALIENCY METHOD")
    print(f"{'='*100}")

    methods = set(m['saliency_method'] for m in metrics_list)

    for method in methods:
        method_results = [m for m in metrics_list if m['saliency_method'] == method]
        best = max(method_results, key=lambda x: x['improvement'])

        print(f"\n🔹 {method.upper()}")
        print(f"  Best improvement: {best['improvement']:+.4f}")
        print(f"  Layers: {best['layer_start']}-{best['layer_end']}")
        print(f"  Boost/Suppress: {best['boost_alpha']}/{best['suppress_alpha']}")
        print(f"  Head top-k: {best['head_top_k_pct']}")
        print(f"  Baseline: {best['baseline_acc']:.4f} → SRF: {best['srf_acc']:.4f}")


def generate_recommendation(metrics_list: List[Dict]) -> Dict:
    """Generate recommended configuration based on results."""

    # Find overall best
    best = max(metrics_list, key=lambda x: x['improvement'])

    recommendation = {
        "recommended_config": {
            "layer_start": best['layer_start'],
            "layer_end": best['layer_end'],
            "boost_alpha": best['boost_alpha'],
            "suppress_alpha": best['suppress_alpha'],
            "head_top_k_pct": best['head_top_k_pct'],
            "saliency_config": {
                "method": best['saliency_method']
            }
        },
        "expected_performance": {
            "baseline_acc": best['baseline_acc'],
            "srf_acc": best['srf_acc'],
            "improvement": best['improvement']
        },
        "experiment_id": best['exp_id']
    }

    return recommendation


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--output_dir", type=str, default="grid_search_results",
                       help="Grid search output directory")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top results to show")
    parser.add_argument("--save_config", action="store_true",
                       help="Save recommended config to file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)

    # Load results
    results = load_results(output_dir)

    if not results:
        sys.exit(1)

    # Extract metrics
    metrics_list = [extract_metrics(exp) for exp in results]
    metrics_list = [m for m in metrics_list if m is not None]

    if not metrics_list:
        print("No valid experiments found")
        sys.exit(1)

    print(f"Successfully extracted metrics from {len(metrics_list)} experiments")

    # Print top results
    print_top_results(metrics_list, args.top_n)

    # Analyze parameter effects
    analyze_parameter_effects(metrics_list)

    # Find best per method
    find_best_per_method(metrics_list)

    # Generate recommendation
    recommendation = generate_recommendation(metrics_list)

    print(f"\n{'='*100}")
    print("🎯 RECOMMENDED CONFIGURATION FOR LLAVA")
    print(f"{'='*100}")
    print(f"\nBased on {len(metrics_list)} experiments:")
    print(f"  Layer range: {recommendation['recommended_config']['layer_start']}-"
          f"{recommendation['recommended_config']['layer_end']}")
    print(f"  Boost alpha: {recommendation['recommended_config']['boost_alpha']}")
    print(f"  Suppress alpha: {recommendation['recommended_config']['suppress_alpha']}")
    print(f"  Head top-k: {recommendation['recommended_config']['head_top_k_pct']}")
    print(f"  Saliency method: {recommendation['recommended_config']['saliency_config']['method']}")
    print(f"\nExpected performance:")
    print(f"  Baseline: {recommendation['expected_performance']['baseline_acc']:.4f}")
    print(f"  SRF: {recommendation['expected_performance']['srf_acc']:.4f}")
    print(f"  Improvement: {recommendation['expected_performance']['improvement']:+.4f}")

    # Save recommendation
    if args.save_config:
        config_file = output_dir / "recommended_config.json"
        with open(config_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        print(f"\n✅ Saved recommended config to {config_file}")

    # Generate bash script with recommended parameters
    script_file = output_dir / "run_recommended_llava.sh"
    generate_run_script(recommendation, script_file)
    print(f"✅ Generated run script: {script_file}")


def generate_run_script(recommendation: Dict, output_path: Path):
    """Generate a bash script to run with recommended parameters."""

    script_content = f'''#!/usr/bin/env bash
# =============================================================================
# LLaVA POPE SRF — Recommended Configuration
# =============================================================================
# Based on grid search results
# Expected improvement: {recommendation['expected_performance']['improvement']:+.4f}
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/home/anna2/.cache/huggingface"
export TRANSFORMERS_CACHE="${{HF_HOME}}/hub"
export HF_DATASETS_CACHE="${{HF_HOME}}/datasets"
export HF_HUB_CACHE="${{HF_HOME}}/hub"

CONDA_ENV="mllm"
PYTHON_PATH="$HOME/miniconda3/envs/${{CONDA_ENV}}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "Running LLaVA with recommended SRF configuration..."
echo "Layers: {recommendation['recommended_config']['layer_start']}-{recommendation['recommended_config']['layer_end']}"
echo "Boost: {recommendation['recommended_config']['boost_alpha']}, Suppress: {recommendation['recommended_config']['suppress_alpha']}"
echo "Saliency: {recommendation['recommended_config']['saliency_config']['method']}"

"${{PYTHON_PATH}}" "${{SCRIPT_DIR}}/pope_srf_eval.py" \\
    --arch llava \\
    --model llava-hf/llava-1.5-7b-hf \\
    --n_samples 500 \\
    --mode both \\
    --output "${{SCRIPT_DIR}}/pope_full_results/llava_recommended.json"

echo "Done!"
'''

    output_path.write_text(script_content)
    output_path.chmod(0o755)


if __name__ == "__main__":
    main()
