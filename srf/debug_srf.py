#!/usr/bin/env python3
"""DEBUG SCRIPT: Verify SRF is actually being applied"""
import os
import sys
import torch
import pathlib

_SRF_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_SRF_DIR / "saliency"))

os.environ.setdefault("HF_HOME", str(_SRF_DIR.parent / "hf_home"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config as CFG
from transformers import AutoProcessor, LlavaForCausalLM
from qwen_vl_utils import process_vision_info
from datasets import load_dataset as hf_load

print("🔍 DEBUG: Verifying SRF is Actually Applied")
print("="*70)

model_id = "llava-hf/llava-1.5-7b-hf"
print(f"\nLoading model: {model_id}")

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
).eval()

# Setup patch
import importlib
import llava_attn_patch
importlib.reload(llava_attn_patch)
patch = llava_attn_patch

print(f"✅ Model loaded")

# Setup SRF
import srf
srf.setup(model, processor, calib_dataset="pope", model_id=model_id)
srf.reset_for_dataset(dataset="pope", phase="both", alpha=6.0, eps=0.3,
                      clip_coarse_grid=7, clip_top_k_pct=0.5, clip_suppress_thresh=0.0)

print("✅ SRF setup complete")

# Load test sample
ds = hf_load("lmms-lab/POPE", split="test")
sample = ds[0]
image = sample["image"].convert("RGB")
question = str(sample["question"]).strip() + "\nAnswer with Yes or No only."

# Prepare input
msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> {question} ASSISTANT:"
vis, _ = process_vision_info(msgs)
inp = processor(text=[text], images=vis, return_tensors="pt", padding=True).to("cuda")

# Get image token range
img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
ids = inp["input_ids"][0].tolist()
s = ids.index(img_token_id)
e = len(ids) - 1 - ids[::-1].index(img_token_id)

print(f"Image tokens: {s} to {e}")

yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
no_id = processor.tokenizer.convert_tokens_to_ids("No")

# BASELINE
print("\n" + "="*70)
print("BASELINE")
print("="*70)
patch._STATE["method"] = "baseline"
patch._STATE["salience_mask"] = None

with torch.inference_mode():
    logits_base = model(**inp).logits[:, -1, :].float()

logit_yes_base = logits_base[0, yes_id].item()
logit_no_base = logits_base[0, no_id].item()
print(f"Yes: {logit_yes_base:.4f}, No: {logit_no_base:.4f}")

# SRF
print("\n" + "="*70)
print("SRF")
print("="*70)
srf.prepare_sample(inp, s, e, image, question, model, processor)

print(f"SRF State: method={patch._STATE.get('method')}, enh_para={patch._STATE.get('enh_para')}")
print(f"salience_mask: {patch._STATE.get('salience_mask') is not None}")

with torch.inference_mode():
    logits_srf = model(**inp).logits[:, -1, :].float()

logit_yes_srf = logits_srf[0, yes_id].item()
logit_no_srf = logits_srf[0, no_id].item()
print(f"Yes: {logit_yes_srf:.4f}, No: {logit_no_srf:.4f}")

srf.cleanup()

# COMPARISON
print("\n" + "="*70)
print("RESULTS")
print("="*70)
delta_yes = logit_yes_srf - logit_yes_base
delta_no = logit_no_srf - logit_no_base

print(f"Yes Δ: {delta_yes:+.4f}")
print(f"No  Δ: {delta_no:+.4f}")

if delta_yes != 0 or delta_no != 0:
    print("\n✅ SUCCESS: SRF IS being applied! Logits changed.")
else:
    print("\n❌ FAILURE: Logits IDENTICAL! SRF NOT applied!")
