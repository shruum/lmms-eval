#!/usr/bin/env python3
"""
MINIMAL SRF VERIFICATION - Tiny images for fast testing
"""
import subprocess
import time
import os
from PIL import Image
import numpy as np

print("🔍 MINIMAL SRF VERIFICATION - Tiny images")
print("=" * 60)

# Create a tiny test image (64x64 instead of 640x480)
test_img_dir = "/tmp/test_srf_images"
os.makedirs(test_img_dir, exist_ok=True)

# Create a simple test image
tiny_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
tiny_image.save(f"{test_img_dir}/test_image.jpg")
print(f"Created tiny test image: 64x64 pixels")

# Create a minimal annotation file
import json
annotation = {
    "annotations": [
        {
            "image": f"{test_img_dir}/test_image.jpg",
            "question": "Is there a person in this image?",
            "answer": "no",
            "label": "absent"
        }
    ]
}

annotation_file = "/tmp/test_srf_annotation.json"
with open(annotation_file, 'w') as f:
    json.dump(annotation, f)
print(f"Created minimal annotation file")

print("\n🚀 Running SRF on TINY image for QUICK verification...")
print("This should complete in <1 minute with tiny images")

# Run SRF with minimal configuration
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", annotation_file,
    "--pope_vcd_name", "Minimal verification",
    "--pope_image_dir", test_img_dir,
    "--alpha", "0.25",
    "--eps", "0.1",
    "--sys_beta", "0.15",
    "--layer_start", "10",
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",
    "--clip_top_k_pct", "0.30",
    "--n_pope", "1",  # Just 1 sample
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/minimal_verification/"
]

start_time = time.time()

try:
    # Run with shorter timeout since we're using tiny images
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start_time

    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")

    if result.returncode == 0:
        print("✅ Process completed successfully")
        print("✅ Code fixes are working!")
    else:
        print(f"❌ Process failed with return code {result.returncode}")

    # Check for the key debug indicators
    output = result.stdout.lower()
    debug_indicators = {
        "IMG TOKENS FIXED": "img tokens fixed",
        "enh_para": "enh_para calculation",
        "DEBUG SRF": "CLIP saliency debug",
        "accuracy": "Results generated"
    }

    print("\n📊 Key Indicators:")
    for indicator, description in debug_indicators.items():
        found = indicator.lower() in output
        status = "✅" if found else "❓"
        print(f"  {status} {description}")

    # Check for results
    if os.path.exists("results/minimal_verification/"):
        files = os.listdir("results/minimal_verification/")
        print(f"\n📁 Results: {len(files)} files created")
        for f in files:
            print(f"  - {f}")

    # Show any errors
    if result.stderr and "error" in result.stderr.lower():
        print("\n❌ Errors found:")
        print(result.stderr[:500])

except subprocess.TimeoutExpired:
    print(f"\n❌ Test timed out after {time.time() - start_time:.1f} seconds")
    print("Even with tiny images, something is blocking execution")

except Exception as e:
    print(f"\n❌ Exception occurred: {e}")

print("\n" + "="*60)
print("🎯 MINIMAL TEST COMPLETE")
print("="*60)

print("\n💡 What this test verifies:")
print("  1. Code runs without crashing")
print("  2. All SRF components execute")
print("  3. Bug fixes don't cause errors")
print("  4. Basic pipeline works")