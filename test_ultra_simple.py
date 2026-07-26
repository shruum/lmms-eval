#!/usr/bin/env python3
"""
ULTRA-SIMPLE DIRECT TEST - Just verify SRF runs and produces results
"""
import subprocess
import time
import os

print("🔍 ULTRA-SIMPLE SRF TEST - 1 sample only")
print("=" * 100)

# Run with just 1 sample to be super fast
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", "/home/anna2/shruthi/RePOPE/annotations/coco_repope_adversarial.json",
    "--pope_vcd_name", "Ultra-simple test",
    "--pope_image_dir", "/home/anna2/shruthi/dataset/POPE_images/images/val2014",
    "--alpha", "0.25",
    "--eps", "0.1",
    "--sys_beta", "0.15",
    "--layer_start", "10",
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",
    "--clip_top_k_pct", "0.30",
    "--n_pope", "1",  # JUST 1 SAMPLE
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/ultra_simple/"
]

print("Running: 1 sample test...")
print("This should take ~2-3 minutes")

start_time = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
elapsed = time.time() - start_time

print(f"\n⏱️  Completed in {elapsed:.1f} seconds")

# Check for basic success
if result.returncode == 0:
    print("✅ Process completed successfully")
else:
    print(f"❌ Process failed with return code {result.returncode}")

# Show some output
print("\n📊 First 1000 chars of output:")
print(result.stdout[:1000])

if "error" in result.stdout.lower() or "exception" in result.stdout.lower():
    print("\n❌ ERRORS FOUND IN OUTPUT")
    print([line for line in result.stdout.split('\n') if 'error' in line.lower() or 'exception' in line.lower()][:5])

# Check for results
import os
if os.path.exists("results/ultra_simple/"):
    files = os.listdir("results/ultra_simple/")
    print(f"\n📁 Results directory contains {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    # Look for JSON results
    json_files = [f for f in files if f.endswith('.json')]
    if json_files:
        print(f"\n✅ Found {len(json_files)} JSON result files")
        for json_file in json_files:
            print(f"  Reading {json_file}...")
            try:
                with open(f"results/ultra_simple/{json_file}") as f:
                    import json
                    data = json.load(f)
                    print(f"  ✅ Results loaded successfully")
                    if "accuracy" in str(data):
                        print(f"  📊 Accuracy found in results!")
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
    else:
        print("\n❌ No JSON files found in results")
else:
    print("\n❌ Results directory not created")

print("\n" + "="*100)
print("🎯 HOW BOOSTING VARIES FROM ARGS:")
print("="*100)

print("\nWith --alpha 0.25:")
print("  1. CLI argument: α = 0.25")
print("  2. enh_para calculation: 1.0 + 0.25 = 1.25")
print("  3. Per-token formula: scaling = 1.0 + 0.25 × saliency")
print("  4. Variable boost: 0% to 25% based on saliency")

print("\nSo you control the boost range directly:")
print("  --alpha 0.15 → 0% to 15% boost (VAF-like)")
print("  --alpha 0.25 → 0% to 25% boost (moderate)")
print("  --alpha 0.50 → 0% to 50% boost (strong)")
print("  --alpha 1.00 → 0% to 100% boost (aggressive)")

print("\n✅ The alpha value is the MAXIMUM boost for perfect CLIP matches!")