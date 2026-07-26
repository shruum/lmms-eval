#!/usr/bin/env python3
"""
PROPER FORMAT MINIMAL TEST - Use correct RePOPE JSONL format
"""
import subprocess
import time
import os
from PIL import Image
import numpy as np

print("🔍 PROPER FORMAT MINIMAL TEST")
print("=" * 60)

# Create test directory
test_img_dir = "/tmp/test_srf_images"
os.makedirs(test_img_dir, exist_ok=True)

# Create test image with the expected filename
img_filename = "test_image.jpg"
img_path = f"{test_img_dir}/{img_filename}"
tiny_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
tiny_image.save(img_path)
print(f"✅ Created: {img_path}")

# Create proper JSONL format annotation (one JSON per line)
annotation_file = "/tmp/test_srf_annotation.jsonl"
with open(annotation_file, 'w') as f:
    # Write as JSONL format (one line per question)
    f.write(f'{{"question_id": 1, "image": "{img_filename}", "text": "Is there a person in this image?", "label": "no"}}\\n')
print(f"✅ Created: {annotation_file} (correct JSONL format)")

print("\n🚀 Running SRF with proper format...")

cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", annotation_file,
    "--pope_vcd_name", "Proper format test",
    "--pope_image_dir", test_img_dir,
    "--alpha", "0.25",
    "--eps", "0.1",
    "--sys_beta", "0.15",
    "--layer_start", "10",
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",
    "--clip_top_k_pct", "0.30",
    "--n_pope", "1",
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/proper_format_test/"
]

start_time = time.time()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start_time

    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")

    if result.returncode == 0:
        print("🎉 SUCCESS! SRF completed")
        print("✅ Code fixes verified working")
    else:
        print(f"❌ Failed with return code {result.returncode}")

    # Check for samples loaded
    if "Loaded" in result.stdout:
        for line in result.stdout.split('\n'):
            if "Loaded" in line and "samples" in line:
                print(f"📊 {line.strip()}")

    # Show actual errors
    if "Traceback" in result.stdout:
        print("\n❌ Error found:")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'Traceback' in line:
                print('\\n'.join(lines[i:min(len(lines), i+15)]))
                break

except subprocess.TimeoutExpired:
    print(f"\n❌ Timed out after {time.time() - start_time:.1f} seconds")
except Exception as e:
    print(f"\n❌ Exception: {e}")

print("\n" + "="*60)