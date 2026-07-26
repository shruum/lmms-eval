# 📊 How Boosting Variables are Determined from Args (AFTER FIX)

## **Complete Calculation Chain:**

### **Step 1: CLI Argument → Config**
```bash
# User provides:
--alpha 0.25

# Gets stored in:
BIAS["boost_alpha"] = 0.25
```

### **Step 2: BIAS → enh_para (THE FIX!)**
```python
# OLD (BUGGY) CODE:
enh_para = abs(BIAS["boost_alpha"])  # = 0.25 ❌

# NEW (FIXED) CODE:
enh_para = 1.0 + abs(BIAS["boost_alpha"])  # = 1.25 ✅
```

### **Step 3: enh_para → Per-Token Scaling**
```python
# In llava_attn_patch.py line 141:
scaling = 1.0 + (enh_para - 1.0) * saliency

# With enh_para = 1.25:
scaling = 1.0 + (1.25 - 1.0) * saliency
scaling = 1.0 + 0.25 * saliency
```

## **Examples with --alpha 0.25:**

| Saliency | Calculation | Result | Boost |
|----------|-------------|--------|-------|
| **0.0** | `1.0 + 0.25 × 0.0` | 1.000 | 0.0% |
| **0.2** | `1.0 + 0.25 × 0.2` | 1.050 | +5.0% |
| **0.5** | `1.0 + 0.25 × 0.5` | 1.125 | +12.5% |
| **0.8** | `1.0 + 0.25 × 0.8` | 1.200 | +20.0% |
| **1.0** | `1.0 + 0.25 × 1.0` | 1.250 | +25.0% |

**Key insight:** The CLI `--alpha` value directly controls the **maximum boost** for highest saliency tokens.

## **How Different Alpha Values Translate:**

| Alpha | enh_para | Max Boost | Avg Boost | Effect |
|-------|----------|-----------|-----------|---------|
| **0.15** | 1.15 | +15% | ~7.5% | VAF-like (conservative) |
| **0.25** | 1.25 | +25% | ~12.5% | Moderate boost |
| **0.50** | 1.50 | +50% | ~25% | Strong boost |
| **1.00** | 2.00 | +100% | ~50% | Aggressive boost |
| **4.00** | 5.00 | +400% | ~200% | Extreme (previous SRF bug!) |

## **So Now From Args:**

```bash
# You provide:
--alpha 0.25

# This directly determines:
- Max boost: 25% (for perfect saliency match)
- Average boost: ~12.5% (for typical saliency)
- Min boost: 0% (for no saliency)
```

**The alpha value is the "boost strength multiplier" - simpler than dealing with raw enh_para values!**

---

## **🔍 Let's Check if Test Completed:**