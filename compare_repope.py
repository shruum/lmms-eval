#!/usr/bin/env python3
"""
Compare original POPE vs RePOPE annotations to understand annotation changes.
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
ORIGINAL_DIR = Path("/home/anna2/shruthi/VCD/experiments/data/POPE/coco")
REPOPE_DIR = Path("/home/anna2/shruthi/RePOPE/annotations")

SPLITS = ["random", "popular", "adversarial"]

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compare_annotations(split):
    """Compare original vs RePOPE annotations for a split."""
    original_file = ORIGINAL_DIR / f"coco_pope_{split}.json"
    repope_file = REPOPE_DIR / f"coco_repope_{split}.json"

    original = load_json(original_file)
    repope = load_json(repope_file)

    # Create dictionaries for comparison
    original_dict = {item['question_id']: item for item in original}
    repope_dict = {item['question_id']: item for item in repope}

    # Statistics
    stats = {
        'split': split,
        'original_count': len(original),
        'repope_count': len(repope),
        'removed': len(original) - len(repope),
        'label_changes': 0,
        'yes_to_no': 0,
        'no_to_yes': 0,
        'ambiguous_removed': 0,
    }

    # Track changes
    for q_id in original_dict:
        if q_id not in repope_dict:
            # Question was removed
            stats['removed'] += 1
        else:
            orig_item = original_dict[q_id]
            repo_item = repope_dict[q_id]

            if orig_item['label'] != repo_item['label']:
                stats['label_changes'] += 1
                if orig_item['label'] == 'yes' and repo_item['label'] == 'no':
                    stats['yes_to_no'] += 1
                elif orig_item['label'] == 'no' and repo_item['label'] == 'yes':
                    stats['no_to_yes'] += 1

    return stats, original_dict, repope_dict

def show_examples(split, num_examples=5):
    """Show examples of annotation changes."""
    _, original_dict, repope_dict = compare_annotations(split)

    print(f"\n{'='*60}")
    print(f"Examples of Changes in {split.upper()} split:")
    print(f"{'='*60}")

    changes = []
    for q_id in original_dict:
        if q_id not in repope_dict:
            # Removed question
            orig = original_dict[q_id]
            changes.append({
                'type': 'REMOVED',
                'question_id': q_id,
                'question': orig['text'],
                'original_label': orig['label'],
                'new_label': 'REMOVED'
            })
        elif original_dict[q_id]['label'] != repope_dict[q_id]['label']:
            # Changed label
            orig = original_dict[q_id]
            repo = repope_dict[q_id]
            changes.append({
                'type': 'CHANGED',
                'question_id': q_id,
                'question': orig['text'],
                'original_label': orig['label'],
                'new_label': repo['label']
            })

    # Show first few examples
    for i, change in enumerate(changes[:num_examples]):
        print(f"\nExample {i+1}: {change['type']}")
        print(f"  Question ID: {change['question_id']}")
        print(f"  Question: {change['question']}")
        print(f"  Original: {change['original_label']} → RePOPE: {change['new_label']}")

def main():
    print("="*60)
    print("POPE vs RePOPE Annotation Comparison")
    print("="*60)

    all_stats = []

    for split in SPLITS:
        stats, _, _ = compare_annotations(split)
        all_stats.append(stats)

    # Print summary table
    print(f"\n{'Split':<15} {'Original':<10} {'RePOPE':<10} {'Removed':<10} {'Label Changes':<15} {'Yes→No':<10} {'No→Yes':<10}")
    print("-" * 80)

    for stats in all_stats:
        removal_pct = (stats['removed'] / stats['original_count']) * 100
        label_change_pct = (stats['label_changes'] / stats['repope_count']) * 100
        print(f"{stats['split']:<15} {stats['original_count']:<10} {stats['repope_count']:<10} "
              f"{stats['removed']:<10} ({removal_pct:.1f}%) "
              f"{stats['label_changes']:<10} ({label_change_pct:.1f}%) "
              f"{stats['yes_to_no']:<10} {stats['no_to_yes']:<10}")

    # Total statistics
    total_original = sum(s['original_count'] for s in all_stats)
    total_repope = sum(s['repope_count'] for s in all_stats)
    total_removed = sum(s['removed'] for s in all_stats)
    total_label_changes = sum(s['label_changes'] for s in all_stats)
    total_yes_to_no = sum(s['yes_to_no'] for s in all_stats)
    total_no_to_yes = sum(s['no_to_yes'] for s in all_stats)

    print("-" * 80)
    print(f"{'TOTAL':<15} {total_original:<10} {total_repope:<10} "
          f"{total_removed:<10} ({(total_removed/total_original)*100:.1f}%) "
          f"{total_label_changes:<10} ({(total_label_changes/total_repope)*100:.1f}%) "
          f"{total_yes_to_no:<10} {total_no_to_yes:<10}")

    print(f"\nKey Findings:")
    print(f"  • Total samples removed: {total_removed} ({(total_removed/total_original)*100:.1f}%)")
    print(f"  • Total label changes: {total_label_changes} ({(total_label_changes/total_repope)*100:.1f}%)")
    print(f"  • Yes→No changes: {total_yes_to_no} (incorrect 'Yes' labels)")
    print(f"  • No→Yes changes: {total_no_to_yes} (incorrect 'No' labels)")

    # Show examples for each split
    for split in SPLITS:
        show_examples(split, num_examples=3)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
