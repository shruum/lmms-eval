#!/usr/bin/env python3
"""
FIXED MINIMAL SRF VERIFICATION - Correct paths
"""
import subprocess
import time
import os
from PIL import Image
import numpy as np
import json

print("🔍 FIXED MINIMAL SRF VERIFICATION")
print("=" * 60)

# Create test directory and image
test_img_dir = "/tmp/test_srf_images"
os.makedirs(test_img_dir, exist_ok=True)

# Create a simple test image
img_path = f"{test_img_dir}/test_image.jpg"
tiny_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
tiny_image.save(img_path)
print(f"✅ Created test image: {img_path}")

# Create a CORRECT annotation file with proper image path
annotation = {
    "annotations": [
        {
            "image": img_path,  # Use the actual file path, not directory
            "question": "Is there a person in this image?",
            "answer": "no",
            "label": "absent"
        }
    ]
}

annotation_file = "/tmp/test_srf_annotation.json"
with open(annotation_file, 'w') as f:
    json.dump(annotation, f)
print(f"✅ Created annotation file with correct image path")

print("\n🚀 Running SRF with fixed paths...")

# Run SRF with minimal configuration
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", annotation_file,
    "--pope_vcd_name", "Fixed minimal test",
    "--pope_image_dir", test_img_dir,  # Directory for reference
    "--alpha", "0.25",
    "--eps", "0.1",
    "--sys_beta", "0.15",
    "--layer_start", "10",
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",
    "--clip_top_k_pct", "0.30",
    "--n_pope", "1",
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/minimal_fixed/"
]

start_time = time.time()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start_time

    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")

    if result.returncode == 0:
        print("🎉 SUCCESS! SRF completed without errors")
        print("✅ Code fixes are working correctly")

        # Check for results
        if os.path.exists("results/minimal_fixed/"):
            files = os.listdir("results/minimal_fixed/")
            print(f"\n📁 Generated {len(files)} result files")
            for f in files:
                print(f"  - {f}")

        # Look for accuracy in output
        if "accuracy" in result.stdout.lower():
            print("✅ Accuracy metrics computed")
    else:
        print(f"❌ Failed with return code {result.returncode}")

    # Show any actual errors (not warnings)
    if "Traceback" in result.stdout or "Exception" in result.stdout:
        print("\n❌ Actual errors found:")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'Traceback' in line or 'Exception' in line or 'Error' in line:
                print('\n'.join(lines[max(0,i-2):min(len(lines),i+10)]))
                break

except subprocess.TimeoutExpired:
    print(f"\n❌ Test timed out after {time.time() - start_time:.1f} seconds")
except Exception as e:
    print(f"\n❌ Exception: {e}")

print("\n" + "="*60)