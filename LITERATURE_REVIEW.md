# Literature Review - VLM Hallucination Mitigation Methods

**Last Updated:** 2026-06-11
**Purpose:** Consolidate information on all published methods for VLM hallucination mitigation

---

## 📚 **Overview of Methods**

### **Training-Free Methods (Our Focus)**

| Method | Conference | arXiv | Key Technique | Improvement | Repo Status |
|--------|-----------|-------|----------------|-------------|-------------|
| **VCD** | CVPR 2024 | 2311.16922 | Contrastive decoding with noisy images | +3.5% avg | ✅ Cloned |
| **AIR** | 2026 | 2602.24041 | OT-guided patch selection | +2-8% | ❓ Unknown |
| **MemVR** | ICML 2025 | - | Memory-based vision reasoning | +1-3% | ❓ Unknown |
| **VAF/ClearSight** | CVPR 2025 | - | Visual attention amplification | +1-2% | ❓ Unknown |
| **SRF** | - | - | CLIP-guided spatial reasoning focus | TBD | 🔄 In development |

### **Training-Based Methods** (For Reference)
- Multimodal unlearning
- Negative prompt tuning
- Instruction tuning with hallucination reduction

---

## 🔬 **Detailed Method Analysis**

### **1. VCD (Visual Contrastive Decoding)**

**Paper:** "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding"
**Conference:** CVPR 2024
**arXiv:** https://arxiv.org/pdf/2311.16922
**Repo:** https://github.com/DAMO-NLP-SG/VCD
**Status:** ✅ Cloned to `/home/anna2/shruthi/VCD/`

#### **How VCD Works**

1. **Training-free** - no model weights modification
2. **Adds noise to images** - creates distorted version using diffusion noise
3. **Contrastive decoding** - compares logits from original vs noisy image
4. **Formula:**
   ```
   p_vcd = softmax[(1+α)·logit(original) - α·logit(noisy)]
   ```
5. **Cutoff mechanism** - filters tokens using adaptive plausibility constraints

#### **Key Parameters**

**From code (`vcd_utils/vcd_sample.py` lines 142-143):**
- `cd_alpha`: 0.5 (weight for original image logits)
- `cd_beta`: 0.1 (cutoff threshold for adaptive plausibility)

**From bash script (`llava1.5_pope.bash`):**
- `cd_alpha`: 1.0 (different from code default!)
- `cd_beta`: 0.2 (different from code default!)
- `noise_step`: 500 (diffusion noise strength)

**Note:** Paper likely tuned these differently for POPE!

#### **VCD Results on POPE (LLaVA-1.5-7B)**

| Dataset | Split | Vanilla | VCD | Delta |
|---------|-------|---------|-----|-------|
| **COCO** | Random | 83.7% | 85.4% | +1.7% |
| | Popular | 78.2% | 84.3% | +6.1% |
| | Adversarial | 75.0% | 81.8% | +6.8% |
| **A-OKVQA** | Random | 83.4% | 85.9% | +2.5% |
| | Popular | 79.9% | 81.9% | +2.0% |
| | Adversarial | 74.0% | 76.7% | +2.7% |
| **GQA** | Random | 78.2% | 86.3% | +8.1% |
| | Popular | 75.1% | 78.4% | +3.3% |
| | Adversarial | 75.1% | 76.2% | +1.1% |

**Average improvement:** +3.5% accuracy (all 9 splits improved - 100% success rate)
**Best improvement:** +8.1% (GQA Random)

#### **Implementation Details**

**Key files in VCD repo:**
- `experiments/eval/object_hallucination_vqa_llava.py` - Main evaluation
- `vcd_utils/vcd_sample.py` - VCD sampling logic (lines 120-167)
- `vcd_utils/vcd_add_noise.py` - Diffusion noise addition
- `experiments/eval/eval_pope.py` - POPE evaluation

**Contrastive decoding code (lines 142-153):**
```python
cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

# Adaptive plausibility cutoff
cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

# Contrastive decoding
diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```

#### **Reproduction Plan**

1. ✅ VCD repo cloned to `/home/anna2/shruthi/VCD/`
2. ⏳ Setup VCD environment (conda, dependencies)
3. ⏳ Test VCD on one POPE split
4. ⏳ Run VCD on all 9 configurations
5. ⏳ Verify results match within ±0.5% of paper

---

### **2. AIR (Look Carefully)**

**Paper:** "Look Carefully: Training-Free Inference-Time Intervention for Mitigating Hallucinations in LVLMs"
**arXiv:** https://arxiv.org/pdf/2602.24041
**Status:** ❓ Repo not yet located

#### **How AIR Works**

1. **Optimal Transport (OT)-guided patch selection**
2. **Prune non-salient patches** from image
3. **Focus on relevant regions** for question answering
4. **Training-free** - no model modification

#### **AIR Results on POPE (LLaVA-1.5-7B)**

| Dataset | Split | Vanilla | AIR | Delta |
|---------|-------|---------|-----|-------|
| **COCO** | Random | 83.70% | 83.70% | 0.0% |
| | Popular | 78.20% | 78.20% | 0.0% |
| | Adversarial | 75.00% | 75.00% | 0.0% |
| **A-OKVQA** | Random | 83.40% | 83.40% | 0.0% |
| | Popular | 79.90% | 79.90% | 0.0% |
| | Adversarial | 74.00% | 74.00% | 0.0% |
| **GQA** | Random | 83.70% | 86.30% | +2.6% |
| | Popular | 78.20% | 79.40% | +1.2% |
| | Adversarial | 75.10% | 77.80% | +2.7% |

**Note:** Paper table shows different numbers than VCD paper - need to verify which is correct

---

### **3. MemVR (Memory-based Vision Reasoning)**

**Conference:** ICML 2025
**Status:** ❓ Repo not yet located

#### **How MemVR Works**

1. **Memory-based approach** - stores visual features
2. **Reranking mechanism** - improves answer selection
3. **Training-free** - no model modification
4. **Improvement:** +1-3% over vanilla

#### **Key Features**
- Uses memory to store visual reasoning patterns
- Reranks answers based on memory retrieval
- Reduces hallucination by checking against stored knowledge

**Status:** Need to locate paper and repo

---

### **4. VAF/ClearSight (Visual Amplification Fusion)**

**Conference:** CVPR 2025
**Status:** ❓ Repo not yet located

#### **How VAF Works**

1. **Boost visual attention** - amplify relevant visual features
2. **Suppress background** - reduce irrelevant information
3. **Training-free** - no model modification
4. **Similar to SRF** - but without CLIP guidance

#### **VAF Parameters** (From Literature)
- **Alpha (α):** 0.15 (boosting strength)
- **Layers:** 10-15 (fusion layers)
- **Head selection:** 50% (conservative)
- **Eps:** 0.2 (suppression strength)

#### **VAF Results**
- Improvement: +1-2% over vanilla
- Similar to SRF approach
- BUT: Our tests with these parameters showed -1.69% degradation

**Issue:** VAF parameters might be tuned for different model/architecture

---

### **5. Other Related Work**

#### **LLMind (CVPR 2026)**
- **Technique:** Möbius warp + SPSA (simultaneous perturbation stochastic approximation)
- **Idea:** Foveation via adaptive sampling
- **Improvement:** +2-4% reported
- **Status:** Not yet investigated

#### **Multimodal Unlearning**
- **Technique:** Negative prompts
- **Idea:** Steer away from hallucinated objects
- **Requires:** Training/fine-tuning
- **Status:** Training-based, not our focus

#### **Instruction Tuning**
- **Technique:** Fine-tune on hallucination-aware data
- **Requires:** Training
- **Status:** Training-based, not our focus

---

## 🔍 **Comparison with SRF**

### **SRF vs VCD**

| Aspect | SRF | VCD |
|--------|-----|-----|
| **Guidance** | CLIP saliency (external) | Noisy image contrast (internal) |
| **Intervention** | Attention manipulation | Logit manipulation |
| **Target** | Cross-modal fusion layers | Output logits |
| **Parameters** | α, ε, layers, heads | cd_α, cd_β, noise_step |
| **Results (LLaVA)** | -0.53% (current best) | +3.5% (paper) |

**Key Difference:** VCD uses contrastive decoding at output level, SRF manipulates attention at intermediate layers

### **SRF vs VAF**

| Aspect | SRF | VAF |
|--------|-----|-----|
| **Guidance** | CLIP saliency (external) | None (internal) |
| **Intervention** | Attention boost/suppress | Attention boost only |
| **Parameters** | α, ε, layers, heads | α=0.15, layers 10-15 |
| **Results (LLaVA)** | -0.53% (current best) | +1-2% (paper) |

**Key Difference:** VAF doesn't use external CLIP guidance, but also doesn't work in our tests

### **SRF vs MemVR**

| Aspect | SRF | MemVR |
|--------|-----|-------|
| **Guidance** | CLIP saliency (external) | Memory retrieval (internal) |
| **Intervention** | Single-pass attention | Multi-pass reranking |
| **Parameters** | α, ε, layers, heads | Memory size, rerank K |
| **Results (LLaVA)** | -0.53% (current best) | +1-3% (paper) |

**Key Difference:** MemVR uses memory to store and retrieve patterns, SRF uses CLIP for spatial reasoning

---

## 📊 **Summary Table**

| Method | Type | Guidance | Improvement | Status |
|--------|------|----------|-------------|--------|
| **VCD** | Training-free | Noisy image contrast | +3.5% | ✅ Reproducing |
| **AIR** | Training-free | OT patch selection | +2-8% | ❓ Not started |
| **MemVR** | Training-free | Memory retrieval | +1-3% | ❓ Not started |
| **VAF** | Training-free | None (internal) | +1-2% | ❓ Not started |
| **SRF** | Training-free | CLIP saliency | -0.53% | 🔄 Developing |

---

## 🎯 **Research Questions**

### **Why Does VCD Work Better Than SRF?**

**Hypotheses:**
1. **Output-level intervention** might be more effective than attention-level
2. **Contrastive approach** (original vs noisy) better than absolute guidance
3. **Cutoff mechanism** filters implausible tokens effectively
4. **Different layer targets** - VCD doesn't specify layers

**Test:** Reproduce VCD and compare intervention strategies

### **Why Does VAF Work in Paper But Not For Us?**

**Hypotheses:**
1. **Different model** - VAF might be tuned for different architecture
2. **Different dataset** - might work better on other benchmarks
3. **Parameters too weak** - α=0.15 might not be enough for LLaVA
4. **Layer mismatch** - layers 10-15 might not be optimal

**Test:** Try stronger parameters (α=1.0, 2.0), different layers

### **What Makes MemVR Effective?**

**Hypotheses:**
1. **Memory mechanism** captures patterns CLIP misses
2. **Reranking** improves answer selection
3. **Multi-pass** allows refinement
4. **Different guidance** - memory-based vs saliency-based

**Test:** Reproduce MemVR and understand mechanism

---

## 📋 **Reproduction Priority Order**

1. **VCD** (HIGH PRIORITY - June 2026)
   - ✅ Repo cloned
   - ⏳ Setup environment
   - ⏳ Test on one split
   - ⏳ Run all 9 configurations
   - ⏳ Compare with paper

2. **MemVR** (AFTER VCD)
   - ❓ Locate paper/repo
   - 📋 Setup environment
   - 📋 Reproduce results

3. **VAF** (AFTER MemVR)
   - ❓ Locate paper/repo
   - 📋 Setup environment
   - 📋 Reproduce results

4. **AIR** (LOW PRIORITY)
   - ❓ Locate paper/repo
   - 📋 Reproduce if time permits

---

## 🔗 **Resources**

### **Papers**
- VCD: https://arxiv.org/pdf/2311.16922
- AIR: https://arxiv.org/pdf/2602.24041
- MemVR: ICML 2025 (need to locate)
- VAF: CVPR 2025 (need to locate)

### **Repos**
- VCD: https://github.com/DAMO-NLP-SG/VCD
- AIR: (need to locate)
- MemVR: (need to locate)
- VAF: (need to locate)

### **Our Documentation**
- PROJECT_OVERVIEW.md - Project goals
- CODE_GUIDE.md - Code architecture
- RESULTS_COMPENDIUM.md - All results
- NEXT_STEPS.md - Current priorities

---

*For project goals: see PROJECT_OVERVIEW.md*
*For code details: see CODE_GUIDE.md*
*For all results: see RESULTS_COMPENDIUM.md*
*For current work: see NEXT_STEPS.md*
