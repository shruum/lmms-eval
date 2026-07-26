#!/usr/bin/env python3
"""
Diagnostic script to investigate why good CLIP saliency ≠ improved accuracy on VLM Bias.

Tests 5 hypotheses:
1. Model can't count at all
2. Language priors dominate visual signal
3. Wrong layers for counting
4. Attention changes don't affect output
5. CLIP saliency highlights wrong features
"""
import sys
import os
from pathlib import Path
import torch
import json
from datetime import datetime

# Setup paths
_SRF_DIR = Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import config as CFG
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset as hf_load
from qwen_vl_utils import process_vision_info

# Qwen2.5-VL specific imports
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


def load_model_and_processor(model_id=CFG.DEFAULT_MODEL):
    """Load model and processor."""
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def load_vlmbias_samples(n_samples=10):
    """Load VLM Bias samples for testing."""
    # Load from HuggingFace
    dataset = hf_load("tommomeister/VLMs-Are-Biased", split="test")

    # Focus on Animals category (counting task)
    samples = []
    for item in dataset:
        if item.get("category") == "Animals":
            samples.append(item)
            if len(samples) >= n_samples:
                break

    print(f"Loaded {len(samples)} Animals samples")
    return samples


def test_hypothesis_1_counting_capability(model, processor, samples):
    """
    Hypothesis 1: Model can't count at all.

    Test with varying prompt strengths:
    - Default: "How many legs does this animal have?"
    - Explicit: "Count the number of legs in this image. Answer with a single number."
    - Counterevidence: "This image may have an unusual number of legs. Count carefully."
    """
    print("\n" + "="*70)
    print("🧪 HYPOTHESIS 1: Can the model count?")
    print("="*70)

    results = {
        "default": {"correct": 0, "total": 0, "predictions": []},
        "explicit": {"correct": 0, "total": 0, "predictions": []},
        "counterevidence": {"correct": 0, "total": 0, "predictions": []},
    }

    prompts = {
        "default": "How many legs does this animal have? Answer with a single number.",
        "explicit": "Count the number of legs in this image. Answer with a single number.",
        "counterevidence": "This image may have an unusual number of legs. Count carefully and answer with a single number.",
    }

    for idx, sample in enumerate(samples[:10]):  # Test 10 samples
        image = sample["image"]
        question = sample["question"]  # "How many legs...?"
        ground_truth = sample["answer"]  # Expected number

        # Ground truth extraction
        try:
            gt_number = int(ground_truth.strip())
        except:
            print(f"Sample {idx}: Can't parse ground truth '{ground_truth}', skipping")
            continue

        messages = [[] for _ in range(3)]

        # Test each prompt type
        for prompt_type, prompt in prompts.items():
            # Create message
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Generate
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=10)

            # Decode
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            response = response.split("ASSISTANT:")[-1].strip()

            # Extract number
            try:
                predicted_number = int(''.join(filter(str.isdigit, response.split()[0])))
            except:
                predicted_number = -1  # Parse error

            # Record
            correct = (predicted_number == gt_number)
            results[prompt_type]["correct"] += int(correct)
            results[prompt_type]["total"] += 1
            results[prompt_type]["predictions"].append({
                "sample_id": idx,
                "ground_truth": gt_number,
                "predicted": predicted_number,
                "response": response,
                "correct": correct
            })

            print(f"Sample {idx} ({prompt_type}): GT={gt_number}, Pred={predicted_number}, Correct={correct}")

    # Summary
    print("\n" + "="*70)
    print("📊 HYPOTHESIS 1 RESULTS")
    print("="*70)

    for prompt_type, stats in results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{prompt_type:20s}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")

    # Interpretation
    print("\n" + "="*70)
    print("🎯 INTERPRETATION:")
    print("="*70)

    explicit_acc = results["explicit"]["correct"] / results["explicit"]["total"] * 100
    if explicit_acc > 20:
        print("✅ Model CAN count with explicit prompting")
        print("   → Problem: Current prompts are too weak")
        print("   → Solution: Better prompts + stronger visual signal")
    elif explicit_acc > 0:
        print("⚠️  Model has LIMITED counting ability")
        print("   → Can count in some cases but not reliable")
        print("   → Solution: Hybrid approach (visual counting module)")
    else:
        print("❌ Model CANNOT count")
        print("   → Fundamental limitation, can't be fixed with SRF")
        print("   → Solution: Focus on recognition/classification tasks")

    return results


def test_hypothesis_3_wrong_layers(model, processor, samples):
    """
    Hypothesis 3: Counting happens at different layers.

    Test different layer ranges for counting tasks.
    This requires SRF patch to be active.
    """
    print("\n" + "="*70)
    print("🧪 HYPOTHESIS 3: Wrong layer range for counting?")
    print("="*70)
    print("⚠️  This requires SRF patch - skipping for now")
    print("   Run layer sweep manually:")
    print("   python srf/eval.py --datasets vlmbias --layer_start 0 --layer_end 7")
    print("   python srf/eval.py --datasets vlmbias --layer_start 8 --layer_end 15")
    print("   python srf/eval.py --datasets vlmbias --layer_start 20 --layer_end 27")


def main():
    """Run diagnostic tests."""
    print("="*70)
    print("🔍 VLM BIAS DIAGNOSTIC: Why Good Saliency ≠ Good Accuracy?")
    print("="*70)
    print(f"Model: {CFG.DEFAULT_MODEL}")
    print(f"Dataset: VLMs-Are-Biased (Animals category)")
    print("="*70)

    # Load model
    model, processor = load_model_and_processor()

    # Load samples
    samples = load_vlmbias_samples(n_samples=20)

    # Run tests
    print("\n" + "="*70)
    print("🚀 RUNNING DIAGNOSTIC TESTS")
    print("="*70)

    # Hypothesis 1: Can model count?
    h1_results = test_hypothesis_1_counting_capability(model, processor, samples)

    # Hypothesis 3: Wrong layers?
    # test_hypothesis_3_wrong_layers(model, processor, samples)

    # Save results
    output_dir = Path("results/diagnostic_vlmbias")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(h1_results, f, indent=2)

    print(f"\n📁 Results saved to: {results_file}")

    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)

    explicit_acc = h1_results["explicit"]["correct"] / h1_results["explicit"]["total"] * 100
    if explicit_acc > 20:
        print("\n✅ Model can count with explicit prompts")
        print("\nNext steps:")
        print("1. Improve VLM Bias prompts to be more explicit")
        print("2. Add stronger text suppression (srf_text_beta > 0)")
        print("3. Test if SRF + explicit prompts improves over baseline")
    else:
        print("\n❌ Model fundamentally cannot count")
        print("\nNext steps:")
        print("1. Focus SRF on recognition/classification tasks")
        print("2. Consider adding a separate counting module")
        print("3. Test on datasets where visual attention helps (POPE, MMVP)")

    print("="*70)


if __name__ == "__main__":
    main()
