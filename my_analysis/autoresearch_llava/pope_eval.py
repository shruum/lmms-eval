#!/usr/bin/env python3
from __future__ import annotations
import os, pathlib, random, sys, time, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, LlavaForConditionalGeneration
import srf

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SPLIT, N_SAMPLES, SEED = "adversarial", 100, 42
GEN_KWARGS = dict(max_new_tokens=10, do_sample=False)

def load_pope(n: int, seed: int = SEED) -> list[dict]:
    print(f"  Loading POPE ({SPLIT}, n={n}, seed={seed})…")
    ds = hf_load("lmms-lab/POPE", split="test")
    rows = [r for r in ds if str(r.get("category", "")).strip().lower() == SPLIT]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    samples = [{"image": r["image"].convert("RGB"), "question": str(r.get("question", "")).strip() + "\nAnswer with Yes or No only.", "ground_truth": "yes" if str(r.get("answer", "")).strip().lower() == "yes" else "no"} for r in rows]
    print(f"  → {len(samples)} samples loaded")
    return samples

def run() -> float:
    samples = load_pope(N_SAMPLES)
    print(f"\n  Loading {MODEL_ID}…")
    t = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  → Model loaded in {time.time()-t:.1f}s")
    srf.setup(model, processor)
    correct, total = 0, len(samples)
    print(f"\n  Running evaluation ({total} samples)…")
    t = time.time()
    for idx, sample in enumerate(samples, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"    [{idx}/{total}] ETA: {(time.time()-t)/idx*(total-idx+1):.0f}s…", flush=True)
        prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["question"]}]}], add_generation_prompt=True)
        inputs = processor(images=sample["image"], text=prompt, return_tensors="pt").to(model.device)
        srf.prepare_sample(inputs, 0, 0, sample["image"], prompt, model, processor)
        with torch.no_grad():
            outputs = model.generate(**inputs, **GEN_KWARGS)
        srf.cleanup()
        response = processor.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip().lower() if "ASSISTANT:" in response else response.split(prompt)[-1].strip().lower()
        if ("yes" if response.startswith("yes") else "no") == sample["ground_truth"]:
            correct += 1
    acc = correct / total
    print(f"\n{'='*60}\nPOPE accuracy: {acc:.4f}  ({SPLIT}, n={N_SAMPLES})\n  Correct: {correct}/{total}\n{'='*60}\n")
    return acc

if __name__ == "__main__":
    run()
