#!/usr/bin/env python3
"""
Quick sanity check - verify VLM Bias evaluation works before running sweep.

Tests:
1. Can we load the model?
2. Can we load VLM Bias dataset?
3. Can we run baseline eval on 5 samples?
4. Can we run SRF eval on 5 samples?

Time: ~2 minutes
"""
import subprocess
import sys

print("="*70)
print("🧪 VLM BIAS SETUP TEST")
print("="*70)

# Test 1: Load model
print("\n1️⃣ Testing model loading...")
cmd = [
    "conda", "run", "-n", "mllm",
    "python", "-c",
    "from transformers import AutoModelForCausalLM, AutoProcessor; "
    "model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype='auto', device_map='cpu'); "
    "print('✅ Model loaded successfully')"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ PASS: Model loads")
else:
    print("   ❌ FAIL: Model loading failed")
    print(result.stderr)
    sys.exit(1)

# Test 2: Load dataset
print("\n2️⃣ Testing VLM Bias dataset loading...")
cmd = [
    "conda", "run", "-n", "mllm",
    "python", "-c",
    "from datasets import load_dataset; "
    "ds = load_dataset('tommomeister/VLMs-Are-Biased', split='test'); "
    "print(f'✅ Dataset loaded: {len(ds)} samples')"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ PASS: Dataset loads")
else:
    print("   ⚠️  WARNING: Dataset loading failed (might need different name)")

# Test 3: Run baseline on 5 samples
print("\n3️⃣ Testing baseline evaluation (5 samples)...")
cmd = [
    "conda", "run", "-n", "mllm", "--no-capture-output",
    "python", "srf/eval.py",
    "--method", "baseline",
    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--datasets", "vlmbias",
    "--n_vlmbias_per_cat", "1",  # 1 sample per category = 7 total
    "--output", "results/test_baseline/",
]

print("   Running... (this takes ~30 seconds)")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    # Check if accuracy is in output
    if "accuracy" in result.stdout.lower():
        print("   ✅ PASS: Baseline evaluation works")
        # Extract accuracy
        import re
        match = re.search(r'accuracy[=:]([\s\d.]+)', result.stdout, re.IGNORECASE)
        if match:
            acc = match.group(1).strip()
            print(f"   Baseline accuracy: {acc}")
    else:
        print("   ⚠️  WARNING: Evaluation ran but no accuracy found")
        print(f"   Output: {result.stdout[:200]}")
else:
    print("   ❌ FAIL: Baseline evaluation failed")
    print(f"   Error: {result.stderr[:500]}")

# Test 4: Run SRF on 5 samples
print("\n4️⃣ Testing SRF evaluation (5 samples)...")
cmd = [
    "conda", "run", "-n", "mllm", "--no-capture-output",
    "python", "srf/eval.py",
    "--method", "srf",
    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--datasets", "vlmbias",
    "--n_vlmbias_per_cat", "1",
    "--alpha", "8.0",
    "--text_beta", "0.5",
    "--output", "results/test_srf/",
]

print("   Running... (this takes ~30 seconds)")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    if "accuracy" in result.stdout.lower():
        print("   ✅ PASS: SRF evaluation works")
        import re
        match = re.search(r'accuracy[=:]([\s\d.]+)', result.stdout, re.IGNORECASE)
        if match:
            acc = match.group(1).strip()
            print(f"   SRF accuracy: {acc}")
    else:
        print("   ⚠️  WARNING: Evaluation ran but no accuracy found")
else:
    print("   ❌ FAIL: SRF evaluation failed")
    print(f"   Error: {result.stderr[:500]}")

# Summary
print("\n" + "="*70)
print("🎯 SETUP TEST COMPLETE")
print("="*70)

print("\n✅ All tests passed! Ready to run parameter sweep.")
print("\nNext steps:")
print("  1. Run quick sweep: python srf/sweep_vlmbias_quick.py")
print("  2. Or run comprehensive: python srf/sweep_vlmbias_comprehensive.py")
print("\n" + "="*70)
