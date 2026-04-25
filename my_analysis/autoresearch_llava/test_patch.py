#!/usr/bin/env python3
"""Quick test of LLava attention patch."""
from __future__ import annotations
import os, pathlib, sys, time, torch
os.environ["HF_HOME"] = "/home/anna2/.cache/huggingface"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from transformers import AutoProcessor, LlavaForConditionalGeneration
import llava_attn_patch as patch

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def test_patch():
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID}…")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded")

    # Create a simple test input
    from PIL import Image
    import numpy as np

    # Create a dummy image (just for testing - actual content doesn't matter)
    image = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)).convert("RGB")
    question = "Is there a dog in the image?\nAnswer with Yes or No only."

    # Prepare input
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Get image token range
    img_start, img_end = patch.get_image_token_range(inputs, model)
    print(f"  Image token range: [{img_start}, {img_end}] ({img_end - img_start + 1} tokens)")

    # Run sanity checks
    print("\n=== Running sanity checks ===")
    patch.run_sanity_checks(model, inputs, method="srf", value=2.0)

    # Test baseline vs intervention
    print("\n=== Testing baseline vs intervention ===")
    patch.patch_model(model, "baseline", 1.0)
    patch.update_sample(img_start, img_end)
    out_base = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    print(f"Baseline output: {processor.decode(out_base[0], skip_special_tokens=True)}")

    # Test with strong boost
    patch.patch_model(model, "srf", 3.0)
    patch._STATE["salience_mask"] = torch.ones(img_end - img_start + 1, dtype=torch.float32).cuda()
    patch.update_sample(img_start, img_end)
    out_srf = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    print(f"SRF output:     {processor.decode(out_srf[0], skip_special_tokens=True)}")

    # Check if outputs differ
    if torch.equal(out_base, out_srf):
        print("\n⚠️  WARNING: Outputs are identical - patch may not be working!")
    else:
        print("\n✓ Outputs differ - patch is working!")

    patch.unpatch_model(model)
    print("\n✓ Test completed successfully")

if __name__ == "__main__":
    test_patch()
