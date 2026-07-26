# Literature Review & Method Ideas

## Foveation & Adaptive Visual Processing

### LLMind (CVPR 2026) - [arXiv:2603.14882](https://arxiv.org/abs/2603.14882)
**Core Idea**: Bio-inspired training-free adaptive visual representations using Möbius transform for non-uniform sampling.
- **BASS**: Möbius-parameterized warp that zooms into task-relevant regions, compresses background
- **CSF**: Closed-loop semantic feedback via SPSA optimization (gradient-free) to align saliency with task
- **Results**: +20% VQAv2, +38% Seed-Bench, +37% A-OKVQA at 1% pixel budget
- **Gap**: Requires ground-truth for optimization; ~4.75s/iteration overhead

**Our Opportunity**: Use CLIP to estimate fixation point instead of SPSA optimization → zero-shot, no GT needed.

---

### Foveated Reasoner (arXiv:2604.21079) - [arXiv:2604.21079](https://arxiv.org/abs/2604.21079)
**Core Idea**: Autoregressive VLM with foveation during decoding. Starts low-res, triggers high-res crop only when needed.
- **Training**: Coldstart supervision + RL to discourage "see-everything" trivial solutions
- **Key**: Foveation decisions are stateful and action-based within single decoding trajectory
- **Gap**: Requires training; uses reinforcement learning

**Our Opportunity**: Training-free approximation using CLIP similarity to trigger "foveation" (i.e., boost salient tokens).

---

### Foveated Diffusion (arXiv:2603.23491) - [arXiv:2603.23491](https://arxiv.org/abs/2603.23491)
**Core Idea**: Mixed-resolution token allocation for diffusion models. High token density in foveal regions, low in periphery.
- **Mechanism**: Post-training adaptation from existing base model
- **Gap**: Requires eye-tracking or gaze estimation; diffusion-specific

**Our Opportunity**: Use CLIP to predict "gaze" region based on query text instead of eye-tracking.

---

## Token Pruning & Compression

### ADSC - Attention-Driven Self-Compression (arXiv:2602.12618) - [arXiv:2602.12618](https://arxiv.org/abs/2602.12618)
**Core Idea**: LLM itself drives compression of vision tokens.
- **Results**: LLaVA-1.5: -53.7% FLOPs, -56.7% KV-cache, 98.2% accuracy preserved
- **Method**: Self-compression based on attention scores

**Our Opportunity**: Combine ADSC-style pruning with CLIP-guided selection → prune non-salient + boost salient.

---

### HiPrune, TopV, ForestPrune (ICCV/CVPR 2025)
**Core Idea**: Hierarchical attention pruning for visual token reduction.
- **HiPrune**: Training-free hierarchical pruning
- **TopV**: Compatible token pruning for low-memory VLMs
- **ForestPrune**: High-ratio compression for video VLMs

**Our Opportunity**: Adaptive pruning based on CLIP saliency rather than fixed thresholds.

---

## Negative Prompting & Unlearning

### Multimodal Unlearning (arXiv:2603.26316) - [arXiv:2603.26316](https://arxiv.org/html/2603.26316v1)
**Core Idea**: Sensitive association-level multimodal unlearning benchmark.
- **Method**: Sampling-time steering with negative prompts to deflect unwanted associations
- **Key**: No weight changes, pure inference-time intervention

**Our Opportunity**: Use negative prompts like "ignore background objects, focus only on [query object]" to suppress bias.

---

### SAUCE (ICCV 2025) - Selective Concept Unlearning
**Core Idea**: Unlearn specific concepts in VLMs using sparse autoencoders.
- **Method**: Weight-space intervention (requires training)
- **Gap**: Not training-free

**Our Opportunity**: Inference-time approximation using attention biasing instead of weight updates.

---

## Physics-Aware Guidance

### PhysVid (CVPR 2026) - [arXiv:2603.26285](https://arxiv.org/abs/2603.26285)
**Core Idea**: Physics-aware local conditioning for video generation.
- **Negative Physics Prompts**: Descriptions of law violations to steer generation away from implausible trajectories
- **Results**: +33% on VideoPhy physical commonsense

**Our Opportunity**: "Negative bias prompts" - inject descriptions of biased reasoning patterns to steer away from them.

---

## Novel Method Ideas

### Idea 1: CLIP-Guided Negative Prompting
```
1. Extract query object from question ("Is there a cat?")
2. Generate CLIP saliency map for "cat"
3. If max saliency < threshold (object absent):
   - Add negative prompt: "ignore objects, answer based on text only"
   - Suppress all image tokens (α=-5.0)
4. If max saliency ≥ threshold (object present):
   - Boost salient tokens (α=2.0)
   - Suppress non-salient tokens (α=-2.0)
```

### Idea 2: Contrastive Saliency Decoding
```
1. CLIP saliency for query object: S_query
2. CLIP saliency for negative concepts (e.g., "background", "distractor"): S_neg
3. Boost tokens where: S_query - S_neg > threshold
4. This naturally suppresses biasing background objects
```

### Idea 3: Möbius Warp + Attention Injection
```
1. Use CLIP to find fixation point (max saliency region)
2. Apply Möbius transform to warp image (zoom into relevant region)
3. Process warped image through VLM
4. Inject warped attention into original decoding via residual connection
```

### Idea 4: Layer-Adaptive Absence Detection
```
1. Different layers have different "absence" thresholds
2. Early layers: low threshold (more permissive)
3. Middle layers (fusion zone): high threshold (stricter)
4. Late layers: medium threshold
5. Calibrate thresholds per layer on validation set
```

### Idea 5: Ensemble Saliency (CLIP + Internal)
```
1. CLIP saliency: external, query-conditioned
2. Internal attention rollout: model's current focus
3. Combine: S_final = α·S_CLIP + β·S_internal - γ·S_background
4. Where S_background = attention to system prompt tokens
```

---

## Key Design Principles from Literature

| Principle | Source | Application |
|-----------|--------|-------------|
| **Absence-aware processing** | Multimodal Unlearning | Different strategies for present vs absent |
| **Query-conditioned selection** | Foveated Reasoner | Different tokens for different questions |
| **Negative prompting** | PhysVid | Steer away from biased patterns |
| **Training-free optimization** | LLMind (CSF) | Gradient-free feedback loops |
| **Adaptive thresholds** | AdaptVis (ICML 2025) | Layer/confidence-gated interventions |
| **Token selectivity** | AIR (ICLR 2026) | Prune non-salient, boost salient |

---

## Experimental Directions

1. **Baseline**: CLIP saliency + threshold (absence detection)
2. **+ Negative prompts**: Add bias-suppressing prompts for absent objects
3. **+ Möbius warp**: Zoom into salient regions before feeding to VLM
4. **+ Layer-adaptive**: Different boost/suppress per layer
5. **+ Contrastive**: Subtract background saliency from query saliency

---

## Sources

- [LLMind](https://arxiv.org/abs/2603.14882) - CVPR 2026
- [Foveated Reasoner](https://arxiv.org/abs/2604.21079) - arXiv 2026
- [Foveated Diffusion](https://arxiv.org/abs/2603.23491) - arXiv 2026
- [PhysVid](https://arxiv.org/abs/2603.26285) - CVPR 2026
- [ADSC](https://arxiv.org/abs/2602.12618) - arXiv 2025
- [Multimodal Unlearning](https://arxiv.org/html/2603.26316v1) - arXiv 2026
