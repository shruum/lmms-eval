# VAF (ClearSight) vs SRF - Architecture Analysis

**Date:** 2026-06-18  
**Purpose:** Understand LLaVA architecture and compare VAF vs SRF token selection strategies

---

## **🏗️ LLaVA Architecture Overview**

### **Image Token Processing**
```
Input Image → Vision Encoder (ViT-L/14) → Image Features → Projector → Image Tokens
                                                                          ↓
                                                                    24×24 = 576 tokens
```

### **Input Sequence Structure**
```
[<image>] + System Prompt + Question + Answer
   ↓           (35 tokens)  + Question  + Answer
(1 placeholder)              (varies)
```

**Key Point:** LLaVA internally expands `<image>` placeholder → 576 actual image tokens

### **Token Positions in Attention Matrix**
```
Position 0: <image> placeholder (used for processing)
Positions 1-35: System prompt tokens
Positions 35-610: Actual image tokens (576 tokens)
Positions 610+: Question and answer tokens
```

---

## **🎯 VAF (ClearSight) - Simple Uniform Approach**

### **Implementation** (`AttnAdapter.py` lines 70-78)
```python
# Hardcoded for LLaVA-1.5-7B
SYS_LEN = 35      # System prompt length
IMG_LEN = 576     # Image token count (24×24 patches)

if q_len > SYS_LEN + IMG_LEN:
    # Boost attention TO image tokens FROM text tokens
    attn_weights[:, :, SYS_LEN+IMG_LEN:, SYS_LEN:SYS_LEN+IMG_LEN] = self.enh_para * attn_weights[:, :, :, SYS_LEN:SYS_LEN+IMG_LEN]
    
    # Suppress attention TO system tokens FROM text tokens  
    attn_weights[:, :, SYS_LEN+IMG_LEN:, :SYS_LEN] = self.sup_para * attn_weights[:, :, :, :SYS_LEN]
else:
    # Alternative case for shorter sequences
    attn_weights[:, :, :, SYS_LEN:SYS_LEN+IMG_LEN] = self.enh_para * attn_weights[:, :, :, SYS_LEN:SYS_LEN+IMG_LEN]
    attn_weights[:, :, :, :SYS_LEN] = self.sup_para * attn_weights[:, :, :, :SYS_LEN]
```

### **VAF Strategy**
- **Uniform enhancement**: ALL 576 image tokens get same boost
- **Uniform suppression**: ALL 35 system tokens get same suppression  
- **No CLIP**: No external guidance, just simple multipliers
- **Parameters**: `enh_para=1.15`, `sup_para=0.9` (15% boost, 10% suppression)
- **Layer range**: 9-14 (fusion layers where vision and language meet)
- **All heads**: Affects all attention heads equally

### **VAF Advantages**
✅ **Simple** - just multiply attention weights by constants  
✅ **Fast** - no CLIP computation, noun extraction  
✅ **Robust** - works well on RePOPE (88.28% average)  
✅ **Consistent** - uniform treatment avoids edge cases

---

## **🎯 SRF (Our Method) - Complex Selective Approach**

### **Implementation Strategy**
```
1. Extract query nouns from questions (e.g., "dog" from "Is there a dog?")
2. Compute CLIP saliency maps for noun + image  
3. Upsample saliency from 6×6 (36) → 576 (image tokens)
4. Selectively boost ONLY salient image tokens
5. Suppress non-salient image tokens + system tokens
```

### **SRF Code Flow** (simplified)
```python
# Step 1: Noun extraction
noun = extract_query_noun(question, dataset)  # "dog"

# Step 2: CLIP saliency computation
result = compute_clip_salience(image, noun, 
                               grid_h=6, grid_w=6,    # 6×6 coarse grid
                               target_n_tokens=576)    # Upsample to 576
saliency = result.saliency  # Should be 576 elements

# Step 3: Selective enhancement in attention
if saliency is not None and saliency.numel() == 576:  # DIMENSION CHECK
    scaling = 1.0 + (enh_para - 1.0) * saliency  # Per-token scaling
    attn_weights[:, :, :, img_start:img_end+1] *= scaling
else:
    # Fallback to uniform (THIS IS WHAT'S HAPPENING!)
    attn_weights[:, :, :, img_start:img_end+1] *= enh_para
```

### **🐛 Current Bug - Dimension Mismatch**
```python
# Expected: saliency.numel() == 576
# Actual: saliency.numel() == 36 (upsampling not working)
# Result: Falls back to uniform enhancement = SRF becomes VAF!
```

---

## **📊 Comparison: VAF vs SRF**

| Aspect | VAF (ClearSight) | SRF (Our Method) |
|--------|-----------------|------------------|
| **Approach** | Uniform enhancement | Selective enhancement |
| **Token Selection** | ALL 576 image tokens | Salient image tokens only |
| **External Guidance** | None | CLIP saliency + noun extraction |
| **Complexity** | Simple (2 multipliers) | Complex (CLIP, upsampling, nouns) |
| **Parameters** | enh_para=1.15, sup_para=0.9 | α, ε, layer range, heads, CLIP params |
| **Speed** | Fast (no CLIP) | Slower (CLIP computation) |
| **RePOPE Performance** | **88.28%** (best) | **80.07%** (broken) |
| **Working Status** | ✅ Working | ❌ Dimension bug |

---

## **🔧 LLaVA Token Position Details**

### **LLaVA-1.5-7B Image Processing**
```
1. Input: 336×336 pixel image
2. ViT-L/14 encoder: 24×24 patches = 576 patch features
3. Projector: 576 patch features → 576 tokens (4096-dim each)
4. Sequence: [35 system tokens] + [576 image tokens] + [question tokens] + [answer tokens]
```

### **Attention Weight Indexing**
```python
attn_weights.shape = (batch_size, num_heads, q_len, kv_seq_len)

# For generation step:
attn_weights[:, :, query_positions, key_positions]

# VAF example:
# query_positions = SYS_LEN+IMG_LEN: = 35+576: = 611: (text generation tokens)
# key_positions = SYS_LEN:SYS_LEN+IMG_LEN = 35:611 (image tokens)
# Result: Boost attention FROM image tokens TO text generation
```

---

## **💡 Key Insights**

### **Why VAF Works So Well**
1. **Simplicity**: All image tokens contain useful information, uniform boost works
2. **Layer selection**: 9-14 are cross-modal fusion layers (where vision meets language)
3. **Right strength**: 15% boost, 10% suppression (tuned parameters)
4. **No bugs**: Simple code, fewer edge cases

### **Why SRF Should Work (in theory)**
1. **Selectivity**: Only boost relevant image tokens (e.g., "dog" regions)
2. **Noise reduction**: Suppress background/irrelevant image regions
3. **Object-aware**: CLIP knows which image regions contain the query object
4. **Adaptive**: Different saliency for different questions

### **Why SRF Currently Fails**
1. **Dimension bug**: CLIP saliency 36 elements ≠ 576 image tokens
2. **Fallback behavior**: Dimension check fails → uniform enhancement (becomes VAF!)
3. **Wrong parameters**: If using VAF parameters with VAF behavior, VAF wins
4. **Complexity**: More components = more potential bugs

---

## **🎯 Action Items**

### **Immediate: Fix Dimension Bug**
1. Verify CLIP upsampling is actually called
2. Check `target_n_tokens` parameter reaches saliency computation
3. Add debug output showing saliency.shape before/after upsampling

### **Testing: Head-to-Head Comparison**
1. Run VAF on RePOPE (should get 88.28%)
2. Run FIXED SRF on RePOPE (target: beat 88.28%)
3. Compare same base model, same dataset, same evaluation

### **If SRF Still Loses: Consider Hybrid Approaches**
1. **VAF + CLIP**: Use VAF simple structure, add CLIP saliency for token selection
2. **SRF simplified**: Remove CLIP upsampling, use coarse grid directly
3. **Adaptive VAF**: Learn per-token enhancement weights instead of CLIP

---

## **📚 Reference Files**

### **ClearSight Implementation**
- `/home/anna2/shruthi/ClearSight/visaug/inference/AttnAdapter.py` - VAF attention adapter
- `/home/anna2/shruthi/ClearSight/visaug/inference/infer_pope.py` - POPE evaluation
- `/home/anna2/shruthi/ClearSight/LLaVA/run_vaf_all_pope.sh` - Run script

### **Our Implementation**
- `/home/anna2/shruthi/lmms-eval/srf/srf.py` - Main SRF implementation
- `/home/anna2/shruthi/lmms-eval/my_analysis/llava_attn_patch.py` - Attention patching
- `/home/anna2/shruthi/lmms-eval/srf/saliency/clip_salience.py` - CLIP saliency
- `/home/anna2/shruthi/lmms-eval/srf/methods/vaf.py` - VAF implementation

---

## **🏆 Conclusion**

**VAF Strategy**: Simple, uniform enhancement of all image tokens works extremely well  
**SRF Strategy**: Complex, selective enhancement SHOULD work better but needs bug fixes  

**Current Status**: SRF dimension bug makes it fall back to uniform enhancement → SRF behaves like broken VAF  

**Fix Priority**: 
1. Fix CLIP upsampling bug (make saliency 576 elements)
2. Verify selective enhancement works
3. Compare FIXED SRF vs VAF on RePOPE

**If FIXED SRF still loses**: Consider that uniform enhancement (VAF) might be optimal for LLaVA-1.5-7B architecture.

---

*This analysis reveals that SRF's current poor performance is likely due to the dimension bug making it fall back to uniform enhancement, but with wrong parameters. Once fixed, SRF can truly test whether selective enhancement beats uniform enhancement.*