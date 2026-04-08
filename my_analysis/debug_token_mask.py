#!/usr/bin/env python3
"""
Tiny debug script: verify image token mask for 1 sample.

Prints:
  - sequence length
  - n_image_tokens, n_text_tokens
  - first/last 8 token ids + their decoded strings
  - the exact span [start, end) of image tokens in the sequence
  - VTAR per layer and overall

Usage (from repo root):
    PYTHONPATH=. python my_analysis/debug_token_mask.py
"""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise SystemExit("Install qwen-vl-utils: uv add qwen-vl-utils")

from analysis_utils import cap_image_size, to_rgb

# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_IMAGE_SIZE = 320
IMAGE_TOKEN = "<|image_pad|>"
# ---------------------------------------------------------------------------

print(f"Loading model {MODEL_NAME} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
image_token_id: int = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
print(f"  image_token_id  = {image_token_id}  (token string = {IMAGE_TOKEN!r})")

# ---------------------------------------------------------------------------
# Load 1 sample
# ---------------------------------------------------------------------------
print("\nLoading 1 sample from vlms-are-biased ...")
ds = load_dataset("anvo25/vlms-are-biased", split="main")
sample = ds[0]
image = cap_image_size(to_rgb(sample["image"]), MAX_IMAGE_SIZE)
prompt = str(sample.get("prompt", "")).strip()
print(f"  topic  = {sample.get('topic')}")
print(f"  prompt = {prompt[:80]!r}")
print(f"  image  = {image.size}")

# ---------------------------------------------------------------------------
# Build inputs
# ---------------------------------------------------------------------------
messages = [
    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
).to("cuda:0")

input_ids = inputs["input_ids"][0].cpu()  # [S]
S = input_ids.shape[0]
image_mask = input_ids == image_token_id  # bool [S]
n_img = int(image_mask.sum())
n_txt = int((~image_mask).sum())

# ---------------------------------------------------------------------------
# 1. Basic counts
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Sequence length  : {S}")
print(f"  n_image_tokens   : {n_img}")
print(f"  n_text_tokens    : {n_txt}")

# ---------------------------------------------------------------------------
# 2. First and last 8 token ids + decoded strings
# ---------------------------------------------------------------------------
def decode_ids(ids):
    return [
        f"{tid:6d} {processor.tokenizer.decode([tid.item()], skip_special_tokens=False)!r:20s}"
        for tid in ids
    ]

print(f"\n  First 8 tokens:")
for line in decode_ids(input_ids[:8]):
    print(f"    {line}")

print(f"\n  Last 8 tokens:")
for line in decode_ids(input_ids[-8:]):
    print(f"    {line}")

# ---------------------------------------------------------------------------
# 3. Vision token span
# ---------------------------------------------------------------------------
img_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)
span_start = int(img_positions[0])
span_end   = int(img_positions[-1]) + 1
is_contiguous = (span_end - span_start) == n_img

print(f"\n  Vision token span : [{span_start}, {span_end})  "
      f"length={span_end - span_start}  contiguous={is_contiguous}")
if not is_contiguous:
    gaps = []
    for i in range(len(img_positions) - 1):
        if img_positions[i + 1] - img_positions[i] > 1:
            gaps.append((int(img_positions[i]), int(img_positions[i + 1])))
    print(f"  Gaps in vision span: {gaps}")

# Tokens immediately before and after the vision span
print(f"\n  Token before span [{span_start-1}]: "
      f"id={int(input_ids[span_start-1])}  "
      f"{processor.tokenizer.decode([int(input_ids[span_start-1])], skip_special_tokens=False)!r}")
print(f"  Token after  span [{span_end}]:   "
      f"id={int(input_ids[span_end])}  "
      f"{processor.tokenizer.decode([int(input_ids[span_end])], skip_special_tokens=False)!r}")

# ---------------------------------------------------------------------------
# 4. image_grid_thw
# ---------------------------------------------------------------------------
image_grid_thw = inputs.get("image_grid_thw")
if image_grid_thw is not None:
    g = image_grid_thw[0]
    t_dim, h_dim, w_dim = int(g[0]), int(g[1]), int(g[2])
    merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
    tok_h, tok_w = h_dim // merge_size, w_dim // merge_size
    print(f"\n  image_grid_thw   : t={t_dim}  h_patches={h_dim}  w_patches={w_dim}  (merge_size={merge_size})")
    print(f"  token grid       : {tok_h} × {tok_w} = {tok_h * tok_w}  (should == n_img={n_img})")
    if t_dim * tok_h * tok_w != n_img:
        print(f"  *** MISMATCH: {t_dim}×{tok_h}×{tok_w} = {t_dim*tok_h*tok_w} ≠ {n_img} ***")
    else:
        print(f"  ✓ token grid × t matches n_img")

# ---------------------------------------------------------------------------
# 5. VTAR — capture per-layer attention at last_pos → image tokens
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Computing VTAR (last_pos → image tokens, per layer) ...")
last_pos = S - 1

captured = []

def make_hook(storage):
    def hook_fn(module, _inp, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            # output[1]: [batch, n_heads, S, S]
            row = output[1][0, :, last_pos, :].detach().cpu()  # [n_heads, S]
            storage.append(row)
            return (output[0], None) + output[2:]
        return output
    return hook_fn

lm = getattr(model, "language_model", None) or model.model
hooks = [layer.self_attn.register_forward_hook(make_hook(captured)) for layer in lm.layers]
with torch.inference_mode():
    model(**inputs, output_attentions=True)
for h in hooks:
    h.remove()

print(f"  Captured {len(captured)} layers  (expected {len(lm.layers)})")

vtar_per_layer = []
for layer_idx, row in enumerate(captured):
    # row: [n_heads, S]
    vis = row[:, image_mask].sum(dim=-1)   # [n_heads]
    tot = row.sum(dim=-1).clamp(min=1e-9)  # [n_heads]
    vtar_per_layer.append(float((vis / tot).mean()))

overall_vtar = float(sum(vtar_per_layer) / len(vtar_per_layer))

print(f"\n  Per-layer VTAR (first 5 / last 5):")
for i, v in enumerate(vtar_per_layer[:5]):
    print(f"    layer {i:2d}: {v*100:6.2f}%")
print(f"    ...")
for i, v in enumerate(vtar_per_layer[-5:]):
    print(f"    layer {len(vtar_per_layer)-5+i:2d}: {v*100:6.2f}%")

print(f"\n  Overall VTAR : {overall_vtar*100:.2f}%  vision")
print(f"               : {(1-overall_vtar)*100:.2f}%  text")
print(f"\n  n_heads per layer: {captured[0].shape[0]}")
print(f"  last_pos         : {last_pos}")
print(f"  image span       : [{span_start}, {span_end})  n={n_img}")
