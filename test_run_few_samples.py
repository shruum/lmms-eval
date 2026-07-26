#!/usr/bin/env python3
"""
Simple SRF test on few samples with detailed boost calculation tracing
"""
import subprocess
import sys

print("🔍 SRF FEW SAMPLES TEST - With Boost Calculation Tracing")
print("=" * 100)

# Test on just 2 samples to keep it fast and clear
cmd = [
    "/home/anna2/miniconda3/envs/mllm/bin/python",
    "/home/anna2/shruthi/lmms-eval/srf/eval.py",
    "--method", "srf",
    "--model", "llava-hf/llava-1.5-7b-hf",
    "--datasets", "pope_vcd",
    "--calib_dataset", "pope",
    "--pope_vcd_file", "/home/anna2/shruthi/RePOPE/annotations/coco_repoe_adversarial.json",
    "--pope_vcd_name", "Debug test - 2 samples",
    "--pope_image_dir", "/home/anna2/shruthi/dataset/POPE_images/images/val2014",
    "--alpha", "0.25",  # This is the CLI arg we provide
    "--eps", "0.1",
    "--sys_beta", "0.15",
    "--layer_start", "10",
    "--layer_end", "15",
    "--head_top_k_pct", "0.50",
    "--clip_top_k_pct", "0.30",
    "--n_pope", "2",  # Only 2 samples
    "--do_sample", "--temperature", "0.7", "--top_p", "0.9",
    "--output", "results/debug_few_samples/"
]

print("📊 BOOST CALCULATION TRACE FROM CLI ARGUMENTS")
print("-" * 100)

# Trace the calculation manually first
alpha_cli = 0.25
print(f"Step 1: CLI argument")
print(f"  --alpha {alpha_cli}")

print(f"\nStep 2: What gets stored in BIAS['boost_alpha']")
print(f"  BIAS['boost_alpha'] = {alpha_cli} (directly from CLI)")

print(f"\nStep 3: What enh_para becomes (AFTER FIX)")
print(f"  enh_para = 1.0 + abs(BIAS['boost_alpha'])")
print(f"  enh_para = 1.0 + {alpha_cli} = {1.0 + alpha_cli}")

print(f"\nStep 4: How per-token scaling is calculated")
print(f"  Formula: scaling = 1.0 + (enh_para - 1.0) * saliency")
print(f"  scaling = 1.0 + ({1.0 + alpha_cli} - 1.0) * saliency")
print(f"  scaling = 1.0 + {alpha_cli} * saliency")

print(f"\nStep 5: Example boosting amounts for different saliency values")
enh_para_fixed = 1.0 + alpha_cli
for sal in [0.0, 0.2, 0.5, 0.8, 1.0]:
    scaling = 1.0 + (enh_para_fixed - 1.0) * sal
    boost = (scaling - 1.0) * 100
    print(f"  saliency={sal:.1f} → scaling={scaling:.4f} → boost={boost:+5.1f}%")

print(f"\n📊 SUMMARY: From CLI arg --alpha {alpha_cli}")
print(f"  → Average boost: ~{alpha_cli * 50:.1f}% (for high saliency tokens)")
print(f"  → Max boost: {alpha_cli * 100:.1f}% (for saliency=1.0)")

print("\n" + "="*100)
print("🚀 RUNNING SRF ON 2 SAMPLES...")
print("="*100 + "\n")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    print("📊 OUTPUT:")
    print(result.stdout[:2000])  # Show first 2000 chars

    if result.stderr:
        print("\n⚠️  ERRORS:")
        print(result.stderr[:1000])

    print("\n" + "="*100)
    print("🎯 CHECKING FOR SUCCESS INDICATORS...")
    print("="*100)

    # Check for success indicators
    success_indicators = [
        ("IMG TOKENS FIXED", "Image token detection working"),
        ("DEBUG SRF", "CLIP saliency debug output"),
        ("accuracy", "Results computed"),
        ("enh_para", "enh_para value displayed"),
    ]

    found = []
    for indicator, description in success_indicators:
        if indicator in result.stdout:
            found.append(description)
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - NOT FOUND")

    if len(found) >= 3:
        print(f"\n✅ {len(found)}/{len(success_indicators)} indicators found - TEST RUNNING!")
        print("\n🔍 Checking for enh_para values in output...")

        # Look for the actual enh_para values
        if "enh_para=" in result.stdout:
            print("Found enh_para values - checking calculation...")
            # Extract a few enh_para values to verify they're correct
            lines = result.stdout.split('\n')
            for line in lines:
                if 'enh_para=' in line and 'DEBUG' in line:
                    print(f"  {line.strip()}")
                    try:
                        enh_val = float(line.split('enh_para=')[1].split(',')[0])
                        if enh_val > 1.0:
                            print(f"    ✅ enh_para={enh_val:.4f} > 1.0 (BOOSTING CORRECT!)")
                        else:
                            print(f"    ❌ enh_para={enh_val:.4f} ≤ 1.0 (STILL BUGGY!)")
                    except:
                        pass
    else:
        print(f"\n❌ Only {len(found)}/{len(success_indicators)} indicators found - TEST MAY HAVE ISSUES")

except subprocess.TimeoutExpired:
    print("\n⏱️  Test timed out after 10 minutes")
except Exception as e:
    print(f"\n❌ Test failed: {e}")

print("\n" + "="*100)
print("📊 FINAL BOOST VARIABLE CALCULATION SUMMARY")
print("="*100)

print(f"\nHow boosting variables are determined from args (AFTER FIX):")
print(f"  CLI: --alpha {{value}}")
print(f"  ↓")
print(f"  BIAS['boost_alpha'] = {{value}}")
print(f"  ↓")
print(f"  enh_para = 1.0 + BIAS['boost_alpha'] = 1.0 + {{value}}")
print(f"  ↓")
print(f"  scaling = 1.0 + (enh_para - 1.0) * saliency")
print(f"  ↓")
print(f"  Variable boost per token based on its saliency")

print(f"\nWith --alpha 0.25:")
print(f"  enh_para = 1.25")
print(f"  Max boost (saliency=1.0): +25%")
print(f"  Min boost (saliency=0.0): +0%")
print(f"  Average boost (saliency=0.5): +12.5%")

print("\n" + "="*100)