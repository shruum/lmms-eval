# Investigation: Why Good CLIP Saliency ≠ Improved Accuracy

**Date:** 2026-05-02
**Status**: ACTIVE INVESTIGATION

---

## 🔍 The Mystery

**Observation**: CLIP saliency images look good (correctly focus on legs, relevant parts)
**Result**: Accuracy barely improves (+0.68% on VLM Bias)
**Question**: WHY?

---

## 📊 Baseline Failure Analysis

### VLM Bias Category Breakdown (Qwen2.5-VL-3B)

| Category | Accuracy | Task Type | Status |
|----------|----------|-----------|--------|
| **Animals** | **0.00%** (0/546) | Counting | 🔴 CATASTROPHIC |
| **Chess Pieces** | **0.00%** (0/288) | Counting | 🔴 CATASTROPHIC |
| Logos | 7.25% | Recognition | 🟠 POOR |
| Game Boards | 8.93% | Counting | 🟠 POOR |
| Patterned Grid | 13.10% | Counting | 🟠 POOR |
| Flags | 22.50% | Recognition | 🟡 MODERATE |
| **Optical Illusion** | **48.11%** | Binary Choice | 🟢 BEST |

### Key Insight
**Counting tasks = 0-13%**
**Recognition/Classification tasks = 22-48%**

The model **cannot count**, even when looking at the right place!

---

## 🎯 Hypotheses: Why Good Saliency Fails

### **Hypothesis 1: Model Lacks Counting Capability** ⭐ MOST LIKELY

**Problem**: Even with perfect visual attention, Qwen2.5-VL can't count objects

**Evidence**:
- Animals: 0% despite CLIP focusing on legs
- Chess Pieces: 0% despite CLIP focusing on pieces
- Optical Illusion: 48% (binary choice, not counting)

**Test**: Can the model count when explicitly prompted?
```python
# Test if model can count with strong prompting
prompt = "Count the number of [object] in this image. Answer with a single number."
# If this still fails → model can't count
# If this succeeds → model can count but current prompts aren't working
```

**Implication**: SRF can't fix a fundamental model limitation

---

### **Hypothesis 2: Language Prior Dominates Visual Signal**

**Problem**: Strong language priors override visual information

**Evidence**:
- Animals 0%: Model might default to "4" (typical animal leg count)
- Chess Pieces 0%: Model might guess based on board state
- VLM Bias paper specifically shows models are biased to language priors

**Test**: Ablate text input, see if model changes answer
```python
# Test 1: Image only
"Count the [objects]." + image

# Test 2: Counterevidence prompt
"This image may have an unusual number of [objects]. Count carefully."
```

**Implication**: Need stronger text suppression (higher srf_text_beta)

---

### **Hypothesis 3: Wrong Layers for Counting**

**Problem**: Counting might happen at different layers than we're boosting

**Evidence**:
- Current SRF: layers 8-15 (middle fusion)
- Counting might need: layers 20-27 (late reasoning)
- Or layers 0-7 (early visual processing)

**Test**: Sweep layer ranges for counting tasks
```python
# Test different layer ranges
layer_ranges = [
    (0, 7),    # Early visual
    (8, 15),   # Current (middle fusion)
    (16, 23),  # Late reasoning
    (20, 27),  # Final layers
    (0, 27),   # All layers
]
```

**Implication**: Wrong layer range for counting tasks

---

### **Hypothesis 4: Attention Boost ≠ Answer Change**

**Problem**: Changing attention doesn't change the final answer

**Evidence**:
- SRF changes attention (confirmed in debug logs)
- But logits only change minimally
- Final answer remains the same

**Test**: Measure if attention changes propagate to output
```python
# Compare top-10 tokens before/after SRF
baseline_top10 = model.generate_baseline(image, question)[:10]
srf_top10 = model.generate_srf(image, question)[:10]

# If they're identical → attention change doesn't affect output
# If they differ → something else is wrong
```

**Implication**: Need post-softmax or post-processing intervention

---

### **Hypothesis 5: CLIP Saliency Quality Issue**

**Problem**: Saliency looks good to humans but isn't what model needs

**Evidence**:
- CLIP focuses on "legs" (correct)
- But model might need to count:
  - Individual legs vs. groups
  - Partial vs. fully visible legs
  - Foreground vs. background legs

**Test**: Compare CLIP saliency vs. model attention rollout
```python
# Compute attention rollout (deep-layer attention patterns)
attention_rollout = compute_attention_rollout(model, image)

# Compare with CLIP saliency
correlation = compare(CLIP_saliency, attention_rollout)

# If low correlation → CLIP highlights wrong things for the model
```

**Implication**: Need model-specific saliency, not CLIP

---

## 🧪 Diagnostic Experiments

### **Experiment 1: Can Qwen2.5-VL Count at All?**

**Goal**: Test if model has any counting capability

**Method**:
```bash
# Test with explicit counting prompts on 50 samples
python srf/counting_capability_test.py \
  --dataset vlmbias \
  --categories Animals Chess_Pieces \
  --n_samples 50 \
  --prompts explicit counterevidence
```

**Expected Outcomes**:
- Explicit prompt succeeds → Model can count, current prompts just weak
- Explicit prompt fails → Model fundamentally can't count

---

### **Experiment 2: Layer Range Sweep for Counting**

**Goal**: Find if counting happens at different layers

**Method**:
```bash
python srf/eval.py \
  --datasets vlmbias \
  --categories Animals Chess_Pieces \
  --layer_start 0 --layer_end 7    # Early
python srf/eval.py \
  --datasets vlmbias \
  --categories Animals Chess_Pieces \
  --layer_start 8 --layer_end 15   # Middle (current)
python srf/eval.py \
  --datasets vlmbias \
  --categories Animals Chess_Pieces \
  --layer_start 20 --layer_end 27  # Late
```

**Expected Outcomes**:
- Late layers (20-27) work better → Counting is a reasoning task
- Early layers (0-7) work better → Counting needs visual features
- None work → Hypothesis 1 confirmed (model can't count)

---

### **Experiment 3: Text Suppression Sweep**

**Goal**: Test if language priors are dominating

**Method**:
```bash
# Current: text_beta=0.0 (disabled)
# Try stronger text suppression
python srf/eval.py \
  --datasets vlmbias \
  --categories Animals Chess_Pieces \
  --text_beta 0.1 0.3 0.5 0.7
```

**Expected Outcomes**:
- Higher text_beta improves accuracy → Language priors are the problem
- No improvement → Visual signal is the bottleneck

---

### **Experiment 4: Post-Softmax vs Pre-Softmax**

**Goal**: Test if post-softmax redistribution works better

**Method**:
```python
# Implement post-softmax redistribution
# (See Next_steps.md Approach 1)
# Compare with current pre-softmax additive boost
```

**Expected Outcomes**:
- Post-softmax works better → Current mechanism is wrong
- No improvement → Problem is deeper than mechanism

---

### **Experiment 5: Qualitative Error Analysis**

**Goal**: Understand *why* model gets counting wrong

**Method**:
1. Run SRF on 50 Animals samples
2. Save:
   - Image
   - CLIP saliency
   - Model attention (baseline)
   - Model attention (SRF)
   - Baseline answer
   - SRF answer
   - Ground truth

3. Manually analyze:
   - Does SRF change attention to correct regions? (✅ confirmed)
   - Does answer change? (❌ confirmed)
   - What does the model say instead? (🔍 needs investigation)

**Key Questions**:
- Does model answer with language prior ("4 legs")?
- Does model answer with random number?
- Does model refuse to answer?
- Does model answer something unrelated?

---

## 📊 Expected Results Matrix

| Hypothesis | Evidence For | Evidence Against | Fix |
|------------|--------------|------------------|-----|
| **Model can't count** | Animals=0%, Chess=0% | Model succeeds on other tasks | ❌ Can't fix with SRF |
| **Language prior** | Optical Illusion=48% (binary) | SRF doesn't help | ✅ Higher text_beta |
| **Wrong layers** | Current layers tuned for POPE | Different tasks need different layers | ✅ Layer sweep |
| **Attention ≠ Answer** | Attention changes, answer doesn't | Nothing | ✅ Post-softmax |
| **CLIP quality** | Saliency looks good to humans | Model might need different features | ✅ Model-specific saliency |

---

## 🎯 Next Steps (Priority Order)

### **Phase 1: Quick Checks (1-2 hours)**
1. ✅ **Analyze baseline errors** (Done: above)
2. 🔜 **Experiment 5: Qualitative analysis** - Save 50 samples with attention maps
3. 🔜 **Experiment 1: Counting capability test** - Test with explicit prompts

### **Phase 2: Mechanism Tweaks (4-6 hours)**
4. 🔜 **Experiment 3: Text suppression sweep** - Test if language priors are dominating
5. 🔜 **Experiment 2: Layer range sweep** - Test if counting needs different layers

### **Phase 3: Architecture Changes (1-2 days)**
6. 🔜 **Experiment 4: Post-softmax redistribution** - Implement and test
7. 🔜 **Experiment 5: Model-specific saliency** - Compare CLIP vs attention rollout

---

## 💡 Preliminary Recommendations

Based on the analysis, **Hypothesis 1 (Model can't count)** is most likely:

**Evidence**:
- Animals: 0% despite perfect saliency
- Chess Pieces: 0% despite perfect saliency
- Optical Illusion: 48% (binary choice, not counting)
- VLM Bias paper shows most models fail on counting

**Action Items**:
1. **Test if model can count with explicit prompts** - If it can't, SRF can't fix this
2. **Focus on recognition/classification tasks** - Where visual attention actually helps
3. **Consider hybrid approach** - SRF for recognition + separate counting module

**Alternative**: If model CAN count with explicit prompts, then:
- Problem is prompt engineering, not visual attention
- Solution: Better prompts + stronger text suppression

---

## 📚 References

- VLM Bias Paper: Shows most models fail on counting tasks
- ClearSight Paper: Uses α=0.15 (much gentler than our α=6.0)
- Next_steps.md: Post-softmax redistribution approach

---

**Last Updated**: 2026-05-02
**Status**: Awaiting experimental results
