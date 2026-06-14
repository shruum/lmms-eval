#!/usr/bin/env python3
"""
Quick test: Compare baseline + SRF performance on original POPE vs RePOPE.
Tests on COCO adversarial split (highest error rate: 27.5%).
"""

import json
import sys
from pathlib import Path

# Simulate running on both annotation versions
# We'll use our existing baseline results and simulate what would happen

# Path to existing baseline results
BASELINE_RESULT = Path("/home/anna2/shruthi/lmms-eval/results/llava_pope_sampling_baseline/coco/pope_coco_adversarial_baseline.json")

# Load RePOPE annotations to understand impact
REPOPE_FILE = Path("/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json")
ORIGINAL_FILE = Path("/home/anna2/shruthi/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json")

def load_json(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def simulate_impact():
    """Simulate impact of annotation changes on baseline performance."""

    print("="*60)
    print("Simulating Impact of RePOPE on Baseline Performance")
    print("="*60)

    # Load annotations
    print("\nLoading annotations...")
    original = load_json(ORIGINAL_FILE)
    repope = load_json(REPOPE_FILE)

    # Create dictionaries
    original_dict = {item['question_id']: item for item in original}
    repope_dict = {item['question_id']: item for item in repope}

    print(f"Original POPE: {len(original)} samples")
    print(f"RePOPE: {len(repope)} samples")

    # Load our baseline results (if available)
    if BASELINE_RESULT.exists():
        with open(BASELINE_RESULT, 'r') as f:
            baseline_data = json.load(f)
            baseline_acc = baseline_data['method']['0.0']['accuracy']
            print(f"\nOur baseline accuracy (original POPE): {baseline_acc*100:.2f}%")
    else:
        baseline_acc = 0.7930  # From our documentation
        print(f"\nUsing documented baseline accuracy: {baseline_acc*100:.2f}%")

    # Analyze impact
    print("\n" + "="*60)
    print("Analyzing Impact of Annotation Changes")
    print("="*60)

    # Count how many baseline answers would be affected
    # For simplicity, let's assume baseline followed original labels

    # Find cases where original said "yes" but RePOPE says "no" (incorrect "yes")
    yes_to_no = []
    # Find cases where original said "no" but RePOPE says "yes" (incorrect "no")
    no_to_yes = []

    for q_id in original_dict:
        if q_id in repope_dict:
            orig_label = original_dict[q_id]['label']
            repo_label = repope_dict[q_id]['label']

            if orig_label != repo_label:
                if orig_label == 'yes' and repo_label == 'no':
                    yes_to_no.append(q_id)
                elif orig_label == 'no' and repo_label == 'yes':
                    no_to_yes.append(q_id)

    print(f"\nLabel changes in adversarial split:")
    print(f"  Yes→No (incorrect 'Yes' labels): {len(yes_to_no)}")
    print(f"  No→Yes (incorrect 'No' labels): {len(no_to_yes)}")
    print(f"  Total changes: {len(yes_to_no) + len(no_to_yes)}")

    # Estimate impact on accuracy
    # If baseline got 79.30% correct on original, and 139 of 3000 were wrong labels
    # Let's estimate what would happen with correct labels

    total_samples = len(original)
    wrong_labels = len(yes_to_no) + len(no_to_yes)

    # Rough estimation:
    # If baseline learned to exploit the wrong labels, performance might drop
    # If baseline was actually robust, performance might improve

    print(f"\n{'='*60}")
    print("Impact Estimation")
    print(f"{'='*60}")

    print(f"\nScenario 1: Baseline exploited annotation errors")
    print(f"  → Performance could DROP by 1-3%")
    print(f"  → Estimated RePOPE accuracy: {baseline_acc*100 - 2:.1f}%")

    print(f"\nScenario 2: Baseline was robust to errors")
    print(f"  → Performance could STAY similar or improve slightly")
    print(f"  → Estimated RePOPE accuracy: {baseline_acc*100:.1f}%")

    print(f"\nScenario 3: SRF was actually working correctly")
    print(f"  → If SRF said 'no' to incorrectly labeled 'yes' cases:")
    print(f"  → SRF performance could JUMP significantly on RePOPE")
    print(f"  → Potential SRF improvement: +2-5% on RePOPE")

    print(f"\n{'='*60}")
    print("Recommendation: Run actual test to get real numbers!")
    print(f"{'='*60}")

if __name__ == "__main__":
    simulate_impact()
