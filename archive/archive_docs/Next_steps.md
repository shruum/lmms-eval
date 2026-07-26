# Next Steps: SRF Debugging and Alternative Approaches

**Date:** 2026-04-30
**Status:** Investigating why SRF gets 0.00% delta across datasets

---

## 🔍 Current Status

### Confirmed Findings:
- ✅ **Saliency masks are NOT zero** - Means range 0.31-0.73 (31-73% of tokens selected)
- ✅ **Saliency images look mostly correct** - CLIP focuses on right regions in most cases
- ✅ **Code is working correctly** - enh_para is set, masks are applied, logits differ
- ❌ **But accuracy delta = 0.00%** on MMVP, MME, POPE

### Key Mystery:
Different logits → Same accuracy suggests boosting doesn't affect final decision.

---

## 📚 Literature Review: Techniques to Reduce Language Bias

### Primary Research Papers (2024-2025):

1. **[Looking Beyond Text: Reducing Language Bias in Large Vision-Language Models via Multimodal Dual-Attention and Soft-Image Guidance](https://arxiv.org/abs/2411.14279)** (EMNLP 2025)
   - **Key Contribution:** Multimodal Dual-Attention (MDA) + Soft-Image Guidance (SIG)
   - **Approach:** Parallel dual-attention mechanism for visual and text inputs
   - **Relevance:** Shows that separating visual/text attention streams reduces language bias

2. **[Eliminating Language Bias in Visual Question Answering with Causal Inference and Dual-Attention](https://www.sciencedirect.com/science/article/abs/pii/S0957417425019153)** (2025)
   - **Key Contribution:** Causal inference framework treating language bias as confounder
   - **Approach:** Explicit causal intervention on language pathways
   - **Relevance:** Theoretically sound method for bias reduction

3. **[Understanding and Mitigating Bias in Vision-Language Models via Causal Mediation Analysis](https://arxiv.org/html/2407.02814v2)** (2024)
   - **Key Contribution:** Causal mediation analysis to map bias propagation
   - **Approach:** Identifies bias generation pathways
   - **Relevance:** Helps understand WHERE bias comes from

4. **[Improving Vision-Language-Action models with Active Visual Attention](https://arxiv.org/html/2511.18960v1)** (2024)
   - **Key Contribution:** Active visual attention filtering
   - **Approach:** Model actively filters irrelevant information
   - **Relevance:** Layer-specific attention modulation

5. **[Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Model Behavior](https://arxiv.org/html/2510.17771v1)** (2024)
   - **Key Contribution:** Deep-layer attention as guidance signal
   - **Approach:** Leverage deep-layer attention patterns
   - **Relevance:** Shows attention doesn't always align with behavior

6. **[ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation](https://arxiv.org/pdf/2503.13107)** (2025)
   - **Key Contribution:** Visual Amplification Fusion (VAF)
   - **Approach:** Boost ALL visual tokens with α=0.15, suppress system prompts with β=0.1
   - **Results:** +1.8% on POPE, +7% on MME
   - **Key Insight:** Less is more - simple uniform boost works better

---

## 🔧 Current SRF Approach (Baseline)

```python
# Pre-softmax additive logit boost
logit[i, j, image_tokens] += alpha * saliency[j]

# Parameters:
alpha = 2.0  # Boost strength (13x stronger than VAF!)
clip_top_k_pct = 0.3  # Only boost top 30% salient tokens
clip_suppress_thresh = 0.248  # Absence-aware threshold
```

**Problem:**
- Too strong (α=2.0 vs VAF's 0.15)
- Only boosts CLIP-selected tokens
- Pre-softmax can be unstable

---

## 💡 Alternative Approaches to Try

### **Approach 1: Post-Softmax Attention Redistribution** ⭐ **TRY FIRST**

Instead of pre-softmax logit addition, redistribute post-softmax attention:

```python
# Get softmax attention
attn_softmax = softmax(attention_logits)

# Calculate visual budget
visual_budget = attn_softmax[:, :, image_tokens].sum(dim=-1, keepdim=True)

# Scale visual tokens by saliency
attn_softmax[:, :, image_tokens] *= (1 + alpha * saliency)

# Renormalize within visual tokens
attn_softmax[:, :, image_tokens] /= attn_softmax[:, :, image_tokens].sum(dim=-1, keepdim=True)

# Scale back to original budget
attn_softmax[:, :, image_tokens] *= visual_budget * (1 + img_scale)

# Downscale text tokens to compensate
attn_softmax[:, :, text_tokens] *= (1 - visual_boost_amount)
```

**Advantages:**
- More stable (respects probability constraints)
- Explicit control over visual vs text balance
- Similar to VAF but keeps CLIP saliency

**Why try first:**
- Easy to implement
- Addresses "boost too strong" problem
- Maintains probability distribution

---

### **Approach 2: Layer-Specific Modulation**

Different layers serve different functions - treat them differently:

```python
for layer in range(n_layers):
    if layer < layer_start:
        # Early layers: Boost visual detection
        attention[layer] += visual_boost_early
    elif layer < layer_end:
        # Middle layers: Enhance visual-language fusion with saliency
        attention[layer][:, :, image_tokens] *= (1 + alpha_mid * saliency)
    else:
        # Late layers: Suppress language priors
        attention[layer][:, :, text_tokens] *= (1 - beta_late)
```

**Parameters:**
- `alpha_early = 0.5` (gentle boost)
- `alpha_mid = 2.0` (current SRF)
- `beta_late = 0.1` (suppress text)

**Advantages:**
- Biologically motivated (early: detect, mid: fuse, late: reason)
- Explains why VAF works on layers 10-15
- Can tune per-layer

---

### **Approach 3: Gradient-Based Intervention**

Use gradients to guide attention in right direction:

```python
# Compute gradient of loss w.r.t. attention weights
grad_attn = torch.autograd.grad(loss, attention_weights, create_graph=True)

# Boost attention in direction that reduces hallucination
attention_weights += lr * grad_attn * saliency_mask
```

**Advantages:**
- Most principled (data-driven)
- Adapts to each sample
- Can find optimal boost direction

**Disadvantages:**
- Computationally expensive
- Requires differentiable forward pass
- May overfit to training distribution

---

### **Approach 4: Causal Intervention (Do-Calculus)**

Treat language bias as confounder, apply causal intervention:

```python
# Identify: P(answer | image, text, do(language_bias=0))
# Intervene on language pathway
text_attn = apply_causal_intervention(text_attn, intervention_strength=beta)
visual_attn = visual_attn / (visual_attn.sum() + epsilon)  # Renormalize
```

**Advantages:**
- Theoretically sound (causal inference)
- Explicitly targets language bias
- Based on causal mediation analysis

**Disadvantages:**
- Requires identifying causal structure
- May need structural assumptions
- Complex to implement correctly

---

### **Approach 5: Attention Entropy Regularization**

Encourage uniform visual attention via entropy:

```python
# During training or inference:
visual_attn = attention[:, :, image_tokens]
visual_entropy = -(visual_attn * log(visual_attn + 1e-8)).sum()

# Encourage spreading attention
loss += lambda * (target_entropy - visual_entropy)^2

# At inference, add entropy-based boost:
attention[:, :, image_tokens] += alpha * (1 - visual_entropy / max_entropy)
```

**Advantages:**
- Simple to implement
- Encourages better visual coverage
- Prevents attention collapse

---

### **Approach 6: Token-Level Contrastive Enhancement**

Compute per-token importance and contrast:

```python
# Compute token importance from hidden states
token_importance = compute_token_importance(hidden_states, image_tokens)

# Contrast: enhance high-importance, suppress low-importance
enhanced_tokens = hidden_states.clone()
enhanced_tokens[:, :, image_tokens] *= (1 + alpha * token_importance)
enhanced_tokens[:, :, text_tokens] *= (1 - beta * language_prior)
```

**Advantages:**
- Works at token level (finer granularity)
- Can use hidden state information
- More flexible than fixed saliency

---

### **Approach 7: Dual-Attention Architecture**

Separate visual and textual attention streams:

```python
# Parallel attention streams (not joint)
visual_attn = softmax(Q_v @ K_v^T)  # Visual-only
text_attn = softmax(Q_t @ K_t^T)    # Text-only

# Late fusion
output = visual_attn @ V_v + lambda * text_attn @ V_t
```

**Advantages:**
- Prevents language dominance
- Forces model to use both modalities
- Inspired by MDA paper

**Disadvantages:**
- Requires architecture change
- Can't be implemented as patch
- May break pre-trained features

---

### **Approach 8: Multi-Scale Saliency**

Use saliency at multiple scales (not just CLIP):

```python
# CLIP saliency (coarse)
clip_saliency = compute_clip_saliency(image, question)

# Hidden state saliency (fine-grained)
hidden_saliency = compute_hidden_saliency(model, image_tokens)

# Combine multi-scale
combined_saliency = clip_saliency * 0.5 + hidden_saliency * 0.5
attention[:, :, image_tokens] *= (1 + alpha * combined_saliency)
```

**Advantages:**
- More robust than single saliency source
- Combines complementary signals
- Can weight different sources

---

## 🎯 Recommended Implementation Order

### **Phase 1: Quick Wins (Easy to Implement)**
1. ⭐ **Post-Softmax Redistribution** (Approach 1)
2. **Layer-Specific Modulation** (Approach 2)
3. **Attention Entropy Regularization** (Approach 5)

### **Phase 2: Intermediate Complexity**
4. **Multi-Scale Saliency** (Approach 8)
5. **Token-Level Contrastive Enhancement** (Approach 6)

### **Phase 3: Advanced (Requires significant changes)**
6. **Gradient-Based Intervention** (Approach 3)
7. **Causal Intervention** (Approach 4)
8. **Dual-Attention Architecture** (Approach 7)

---

## 📊 Experimental Plan

### **Current Baseline:**
- **Current SRF:** Pre-softmax additive logit boost
- **Alpha:** 2.0 (POPE/MMVP), 8.0 (VLM-Bias)
- **Results:**
  - POPE: 0.00% delta
  - MMVP: 0.00% delta
  - MME: 0.00% delta
  - VLM-Bias: ~5% (small sample, likely noise)

### **Proposed Experiments:**

#### **Experiment 1: Post-Softmax on POPE (Small Sample)**
- **Dataset:** POPE adversarial (n=20)
- **Compare:** Current SRF vs Post-Softmax
- **Metrics:** Accuracy, F1, Precision, Recall
- **Goal:** Verify post-softmax is more stable

#### **Experiment 2: Layer-Specific on POPE**
- **Dataset:** POPE adversarial (n=20)
- **Compare:** Current SRF vs Layer-Specific
- **Parameters:**
  - Early (0-7): alpha=0.5
  - Mid (8-15): alpha=2.0
  - Late (16-23): beta=0.1
- **Goal:** Test if layer-specific helps

#### **Experiment 3: Comparison on Full POPE**
- **Dataset:** POPE all splits (n=9000)
- **Compare:** Baseline, Current SRF, Post-Softmax, Layer-Specific
- **Goal:** Find best approach on large dataset

#### **Experiment 4: Best Approach on MMVP**
- **Dataset:** MMVP full (n=300)
- **Method:** Winner from Experiment 3
- **Goal:** See if improvement generalizes

#### **Experiment 5: Best Approach on VLM-Bias**
- **Dataset:** VLM-Bias full (n=~300)
- **Method:** Winner from Experiment 3
- **Goal:** Test on counting/precision tasks

---

## 🔬 Technical Implementation Notes

### **Post-Softmax Implementation Location:**
- File: `my_analysis/qwen_attn_patch.py`
- Function: `patched_softmax()`
- Current lines: ~200-300
- Need to add: Post-softmax redistribution logic

### **Layer-Specific Implementation:**
- File: `srf/srf.py`
- Function: `prepare_sample()`
- Current: Uses single `enh_para` for all layers
- Need: Per-layer `alpha` values based on layer index

### **Testing Framework:**
- Use small samples first (n=20)
- Compare: Baseline vs Current SRF vs New Method
- Metrics: Accuracy, F1, plus debug output
- Save saliency images for qualitative analysis

---

## 🚀 Next Steps

1. ✅ **Create this document** (Done)
2. 🔜 **Implement Post-Softmax Redistribution**
3. ⏳ **Test on POPE (n=20)**
4. ⏳ **Compare with Current SRF**
5. ⏳ **If works, test on full datasets**
6. ⏳ **If fails, try Layer-Specific Modulation**

---

## 📖 Paper References

1. Looking Beyond Text: Reducing Language Bias... [[arXiv](https://arxiv.org/abs/2411.14279)]
2. Eliminating Language Bias in VQA with Causal Inference... [[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425019153)]
3. Understanding Bias via Causal Mediation Analysis [[arXiv](https://arxiv.org/html/2407.02814v2)]
4. Improving VLA models with Active Visual Attention [[arXiv](https://arxiv.org/html/2511.18960v1)]
5. Seeing but Not Believing: Visual Attention Probe [[arXiv](https://arxiv.org/html/2510.17771v1)]
6. ClearSight: Visual Signal Enhancement [[arXiv](https://arxiv.org/pdf/2503.13107)]

---

**Last Updated:** 2026-04-30
**Status:** Ready to implement Post-Softmax approach
