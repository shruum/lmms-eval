#!/usr/bin/env python3
"""Evaluate on all 3 POPE categories: random, popular, adversarial."""
from __future__ import annotations
import os, pathlib, random, sys, time, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, LlavaForConditionalGeneration
import llava_attn_patch_working as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
N_SAMPLES = 500
SEED = 42
GEN_KWARGS = dict(max_new_tokens=10, do_sample=False)

SPLITS = ["random", "popular", "adversarial"]


def load_pope_split(split: str, n: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading POPE ({split}, n={n}, seed={seed})…")
    ds = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds if str(r.get("category", "")).strip().lower() == split]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    samples = [
        {
            "image": r["image"].convert("RGB"),
            "question": str(r.get("question", "")).strip() + "\nAnswer with Yes or No only.",
            "ground_truth": "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"
        }
        for r in rows
    ]
    print(f"  → {len(samples)} samples loaded")
    return samples


def evaluate_split(model, processor, samples: list[dict], method: str = "baseline",
                   enh_para: float = 1.0, sup_para: float = 1.0) -> float:
    """Evaluate on samples."""
    correct, total = 0, len(samples)
    for idx, sample in enumerate(samples, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"    [{idx}/{total}]…", flush=True)

        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["question"]}]}],
            add_generation_prompt=True
        )
        inputs = processor(images=sample["image"], text=prompt, return_tensors="pt").to(model.device)

        img_start, img_end = patch.get_image_token_range(inputs, model)
        patch.update_sample(img_start, img_end)
        patch._STATE["method"] = method
        patch._STATE["enh_para"] = enh_para
        patch._STATE["sup_para"] = sup_para

        with torch.no_grad():
            outputs = model.generate(**inputs, **GEN_KWARGS)

        response = processor.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip().lower() if "ASSISTANT:" in response else response.split(prompt)[-1].strip().lower()
        if ("yes" if response.startswith("yes") else "no") == sample["ground_truth"]:
            correct += 1

    return correct / total


def main():
    print("="*60)
    print("POPE EVALUATION: ALL 3 SPLITS")
    print(f"Model: {MODEL_ID}")
    print(f"Samples per split: {N_SAMPLES}")
    print("="*60)

    print(f"\nLoading model...")
    t = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time()-t:.1f}s")

    # Patch model (use ClearSight parameters)
    patch.patch_model(model, "baseline", 1.0, 1.0, layer_start=8, layer_end=14)

    results = {"baseline": {}, "vaf": {}}

    # Evaluate baseline
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION")
    print(f"{'='*60}")
    for split in SPLITS:
        print(f"\n{split.upper()}:")
        samples = load_pope_split(split, N_SAMPLES)
        t = time.time()
        acc = evaluate_split(model, processor, samples, method="baseline")
        elapsed = time.time() - t
        results["baseline"][split] = acc
        print(f"  Accuracy: {acc:.2%} ({elapsed:.1f}s)")

    # Evaluate with VAF
    print(f"\n{'='*60}")
    print("VAF EVALUATION (ClearSight: enh=1.15, sup=0.95)")
    print(f"{'='*60}")
    for split in SPLITS:
        print(f"\n{split.upper()}:")
        samples = load_pope_split(split, N_SAMPLES)
        t = time.time()
        acc = evaluate_split(model, processor, samples, method="srf", enh_para=1.15, sup_para=0.95)
        elapsed = time.time() - t
        results["vaf"][split] = acc
        print(f"  Accuracy: {acc:.2%} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Split':<15} {'Baseline':<12} {'VAF':<12} {'Change':<10}")
    print("-" * 60)

    total_baseline, total_vaf = 0, 0
    for split in SPLITS:
        baseline = results["baseline"][split]
        vaf = results["vaf"][split]
        change = (vaf - baseline) * 100
        symbol = "✓" if change > 0 else ("=" if change == 0 else "✗")
        print(f"{split.capitalize():<15} {baseline:>10.2%}   {vaf:>10.2%}   {symbol} {change:+.2f}%")
        total_baseline += baseline
        total_vaf += vaf

    avg_baseline = total_baseline / len(SPLITS)
    avg_vaf = total_vaf / len(SPLITS)
    avg_change = (avg_vaf - avg_baseline) * 100

    print("-" * 60)
    print(f"{'AVERAGE':<15} {avg_baseline:>10.2%}   {avg_vaf:>10.2%}   {avg_change:+.2f}%")
    print(f"{'='*60}")

    patch.unpatch_model(model)


if __name__ == "__main__":
    main()
