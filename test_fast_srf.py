#!/usr/bin/env python3
"""
Fast SRF test - minimal debug output, quick verification
"""
import subprocess
import sys
import time
import os

print("🚀 FAST SRF TEST - 1 sample, minimal debug")
print("=" * 60)

# Run SRF with minimal output
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "-c",  # Run Python code directly
    """
import sys
sys.argv = [
    'eval.py',
    '--method', 'srf',
    '--model', 'llava-hf/llava-1.5-7b-hf',
    '--datasets', 'pope_vcd',
    '--calib_dataset', 'pope',
    '--calib_n_samples', '5',  # Reduce calibration samples
    '--pope_vcd_file', '/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json',
    '--pope_vcd_name', 'Fast test',
    '--pope_image_dir', '/home/anna2/shruthi/dataset/POPE_images/images/val2014',
    '--alpha', '0.25',
    '--eps', '0.1',
    '--sys_beta', '0.15',
    '--layer_start', '10',
    '--layer_end', '15',
    '--head_top_k_pct', '0.50',
    '--clip_top_k_pct', '0.30',
    '--n_pope', '1',  # Just 1 test sample
    '--do_sample',
    '--temperature', '0.7',
    '--top_p', '0.9',
    '--output', 'results/fast_test/'
]

# Suppress extensive debug output
import logging
logging.basicConfig(level=logging.WARNING)

# Import and run
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')
from srf import eval
eval.main()
"""
]

print("Running with 5 calibration samples + 1 test sample...")
print("This should be much faster with reduced debug output")

start_time = time.time()
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start_time

    print(f"\\n⏱️  Completed in {elapsed:.1f} seconds")

    if result.returncode == 0:
        print("✅ SUCCESS - Process completed without errors")
    else:
        print(f"❌ Failed with return code {result.returncode}")

    # Check for results
    if os.path.exists("results/fast_test/"):
        files = os.listdir("results/fast_test/")
        print(f"\\n📁 Results directory has {len(files)} files")

        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            print(f"✅ Found {len(json_files)} result file(s)")

            # Show accuracy
            import json
            for json_file in json_files:
                with open(f"results/fast_test/{json_file}") as f:
                    data = json.load(f)
                    if "accuracy" in str(data):
                        print(f"📊 Results contain accuracy metric")
    else:
        print("\\n❌ No results directory created")

    # Show any errors (last 20 lines)
    if result.stderr:
        print("\\n⚠️  Errors (last 20 lines):")
        print('\\n'.join(result.stderr.split('\\n')[-20:]))

except subprocess.TimeoutExpired:
    print("\\n⏱️  Timed out after 10 minutes")
except Exception as e:
    print(f"\\n❌ Exception: {e}")

print("\\n" + "=" * 60)
print("🎯 Test Complete")
print("=" * 60)