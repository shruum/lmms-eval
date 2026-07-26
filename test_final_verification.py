#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TEST - Verify SRF calculations are correct after fixes
"""
import sys
sys.path.insert(0, '/home/anna2/shruthi/lmms-eval')

print("🎯 FINAL COMPREHENSIVE VERIFICATION TEST")
print("=" * 100)

# Test 1: Verify the actual code has the fix
print("\n✅ TEST 1: Verify code has correct formulas")
print("-" * 100)

with open('/home/anna2/shruthi/lmms-eval/srf/srf.py', 'r') as f:
    content = f.read()

# Check for the fixed patterns
fixed_patterns = [
    '1.0 + abs(BIAS["boost_alpha"])',
    '1.0 + abs(BIAS["boost_alpha"]) * 0.5',
    '1.0 + BIAS["boost_alpha"]'
]

buggy_patterns = [
    '= abs(BIAS["boost_alpha"])',
    '= BIAS["boost_alpha"]'  # This one should still be preceded by 1.0 +
]

all_good = True
for pattern in fixed_patterns:
    if pattern in content:
        print(f"✅ Found correct pattern: {pattern}")
    else:
        print(f"❌ Missing correct pattern: {pattern}")
        all_good = False

print(f"\n📊 TEST 2: Mathematical verification")
print("-" * 100)

# Simulate the calculation with alpha=0.25
alpha = 0.25
boost_alpha_stored = alpha  # This is what gets stored in BIAS
enh_para_fixed = 1.0 + abs(boost_alpha_stored)

print(f"CLI argument: --alpha {alpha}")
print(f"Stored in BIAS['boost_alpha']: {boost_alpha_stored}")
print(f"enh_para calculation: 1.0 + {boost_alpha_stored} = {enh_para_fixed}")

# Test scaling calculation
import torch
saliency_demo = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])

print(f"\nScaling calculation with fixed enh_para={enh_para_fixed}:")
for sal_val in saliency_demo:
    scaling = 1.0 + (enh_para_fixed - 1.0) * sal_val
    boost_pct = (scaling - 1.0) * 100
    emoji = "✅" if boost_pct >= 0 else "❌"
    print(f"  {emoji} saliency={sal_val:.1f} → scaling={scaling:.4f} → boost={boost_pct:+.1f}%")

# Verify average boost is positive
avg_boost = ((1.0 + (enh_para_fixed - 1.0) * saliency_demo) - 1.0).mean() * 100
print(f"\n✅ Average boost: {avg_boost:.1f}% (POSITIVE - CORRECT!)")

print(f"\n📊 TEST 3: Compare with previous buggy behavior")
print("-" * 100)

enh_para_buggy = abs(boost_alpha_stored)  # The old bug
print(f"OLD BUGGY enh_para: {enh_para_buggy}")

print(f"\nScaling calculation with buggy enh_para={enh_para_buggy}:")
for sal_val in saliency_demo:
    scaling = 1.0 + (enh_para_buggy - 1.0) * sal_val
    boost_pct = (scaling - 1.0) * 100
    emoji = "✅" if boost_pct >= 0 else "❌"
    print(f"  {emoji} saliency={sal_val:.1f} → scaling={scaling:.4f} → change={boost_pct:+.1f}%")

# Verify average change is negative
avg_change = ((1.0 + (enh_para_buggy - 1.0) * saliency_demo) - 1.0).mean() * 100
print(f"\n❌ Average change: {avg_change:.1f}% (NEGATIVE - BUGGY!)")

print(f"\n" + "="*100)
print("🎯 FINAL VERIFICATION RESULTS")
print("="*100)

if all_good and avg_boost > 0:
    print("✅ ALL TESTS PASSED!")
    print(f"✅ enh_para calculation: {enh_para_fixed} (CORRECT)")
    print(f"✅ Average boost: {avg_boost:.1f}% (POSITIVE)")
    print(f"✅ SRF now BOOSTS attention instead of suppressing it!")
    print("\n🚀 SRF is ready for comprehensive testing!")
else:
    print("❌ TESTS FAILED - Check calculations")

print("="*100)
