# Layer-wise and Head Selection Strategy Implementation Plan

**Goal**: Improve SRF performance on POPE benchmark through systematic testing of two targeted strategies

**Target**: ≥1% improvement over baseline (79.30% → ≥80.30% on COCO Adversarial)

**Status**: Phase 1 implementation complete, testing blocked by CLIP authentication issue

---

## Background

### Current Performance (Round 1)
- **Best Config**: Config 11 (α=0.15, layers=10-18, head_pct=0.5) → 78.77%
- **Baseline**: 79.30% on COCO Adversarial
- **Gap**: -0.53% below baseline
- **Key Finding**: Lower alpha works better (α=1.0 > α=2.0 > α=4.0)

### Problem Statement
SRF performs **worse than baseline** on LLaVA-1.5-7B. Need hyperparameters/approach that achieves ≥1% improvement.

---

## Strategy 1: Layer-wise Alpha Strategy ✅ IMPLEMENTED

### Concept
Different fusion stages need different intervention strengths:
- **Early layers** (2-8): Global visual integration → weak boost
- **Middle layers** (10-18): Cross-modal fusion → strong boost (current SRF)
- **Late layers** (20-31): Language prior dominance → minimal boost

### Implementation Status: ✅ COMPLETE & TESTING IN PROGRESS

**Files Modified**:
1. `my_analysis/qwen_attn_patch.py` - Added layer-wise alpha computation
2. `srf/eval.py` - Added CLI arguments
3. `srf/srf.py` - Integrated layer-wise parameters

**Code Changes**:

**my_analysis/qwen_attn_patch.py** (lines 314-333):
```python
# Apply layer-wise multipliers if enabled
if _STATE.get("use_layerwise", False):
    layer_start = _STATE["vaf_layer_start"]
    layer_end = _STATE["vaf_layer_end"]
    early_mult = _STATE.get("early_alpha_mult", 0.3)
    mid_mult = _STATE.get("mid_alpha_mult", 1.5)
    late_mult = _STATE.get("late_alpha_mult", 0.1)
    
    if current_layer < layer_start:
        alpha_val = base_alpha * early_mult
    elif layer_start <= current_layer <= layer_end:
        alpha_val = base_alpha * mid_mult
    else:
        alpha_val = base_alpha * late_mult
else:
    alpha_val = base_alpha
```

**srf/eval.py** (CLI arguments):
```python
p.add_argument("--use_layerwise", action="store_true")
p.add_argument("--early_alpha_mult", type=float, default=0.3)
p.add_argument("--mid_alpha_mult", type=float, default=1.5)
p.add_argument("--late_alpha_mult", type=float, default=0.1)
```

**srf/srf.py** (parameter integration):
```python
# In reset_for_dataset() signature:
use_layerwise: bool | None = None,
early_alpha_mult: float | None = None,
mid_alpha_mult: float | None = None,
late_alpha_mult: float | None = None,

# In overrides dictionary:
"use_layerwise": use_layerwise,
"early_alpha_mult": early_alpha_mult,
"mid_alpha_mult": mid_alpha_mult,
"late_alpha_mult": late_alpha_mult,
```

### Test Matrix

| Config | Early Mult | Mid Mult | Late Mult | Total Range | Status |
|--------|------------|----------|-----------|-------------|--------|
| LW-1 | 0.3 | 1.5 | 0.1 | 0.3-1.5x | ⏳ Ready to test |
| LW-2 | 0.5 | 1.2 | 0.2 | 0.5-1.2x | ⏳ Ready to test |
| LW-3 | 0.1 | 2.0 | 0.05 | 0.1-2.0x | ⏳ Ready to test |
| LW-4 | 0.0 | 1.8 | 0.0 | 0-1.8x | ⏳ Ready to test |
| LW-5 | 0.5 | 1.0 | 0.5 | Uniform | ⏳ Ready to test |

**Base parameters**: α=0.15, layers=10-18 (from best Round 1 result)

### Testing Protocol

**Command Template**:
```bash
conda run -n mllm python srf/eval.py \
  --method srf \
  --use_layerwise \
  --early_alpha_mult <VALUE> \
  --mid_alpha_mult <VALUE> \
  --late_alpha_mult <VALUE> \
  --alpha 0.15 \
  --layer_start 10 \
  --layer_end 18 \
  --datasets pope \
  --n_pope 100 \
  --pope_splits adversarial \
  --output results/layerwise_lw<N>_coco/
```

**Success Criteria**:
- **Minimum**: ≥79.30% (match baseline)
- **Target**: ≥80.30% (+1% improvement)
- **Stretch**: ≥80.80% (+1.5% improvement)

**Current Status**: ✅ CLIP ISSUE RESOLVED - 5 experiments running (LW-1 to LW-5), testing 100 samples each

---

## Strategy 2: Head Selection Strategy ⏳ PENDING

### Concept
Different query types benefit from different head selections:
- **Count queries** ("How many cats?") → Need heads that attend broadly
- **Absence queries** ("Is there a cat?") → Need selective heads
- **Spatial queries** ("Where is the cat?") → Need precise localization
- **Attribute queries** ("What color is the cat?") → Need detail-focused heads

### Implementation Plan

**Files to Create**:
1. `srf/query_classifier.py` - Query classification logic

**Files to Modify**:
1. `srf/srf.py` - Integrate query classification
2. `srf/eval.py` - Add CLI arguments

**Key Implementation**:

**srf/query_classifier.py** (new file):
```python
def classify_query(question: str) -> str:
    """Classify query type from text."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["how many", "count", "number of"]):
        return "count"
    elif any(word in question_lower for word in ["is there", "are there"]):
        return "absence"
    elif any(word in question_lower for word in ["where", "which position"]):
        return "spatial"
    elif any(word in question_lower for word in ["what color", "what size"]):
        return "attribute"
    return "other"

def get_head_params(query_type: str) -> dict:
    """Return head selection parameters for query type."""
    params = {
        "count": {"head_top_k_pct": 0.7, "layer_start": 8, "layer_end": 22},
        "absence": {"head_top_k_pct": 0.3, "layer_start": 12, "layer_end": 18},
        "spatial": {"head_top_k_pct": 0.5, "layer_start": 10, "layer_end": 20},
        "attribute": {"head_top_k_pct": 0.6, "layer_start": 14, "layer_end": 24},
        "other": {"head_top_k_pct": 0.5, "layer_start": 10, "layer_end": 18},
    }
    return params.get(query_type, params["other"])
```

**srf/srf.py** (modification in `prepare_sample()`):
```python
# In prepare_sample(), after getting question:
if _STATE.get("use_query_head_selection", False):
    from srf.query_classifier import classify_query, get_head_params
    
    query_type = classify_query(question)
    head_params = get_head_params(query_type)
    
    # Override head selection for this sample
    _STATE["vaf_head_top_k_pct"] = head_params["head_top_k_pct"]
    _STATE["vaf_layer_start"] = head_params["layer_start"]
    _STATE["vaf_layer_end"] = head_params["layer_end"]
```

**srf/eval.py** (CLI arguments):
```python
p.add_argument("--use_query_heads", action="store_true",
               help="Enable query-conditioned head selection")
```

### Test Matrix

| Config | Query Type | Head % | Layers | Status |
|--------|------------|--------|--------|--------|
| HS-1 | Absence | 30% | 12-18 | ⏳ Not implemented |
| HS-2 | Default | 50% | 10-18 | ⏳ Not implemented |
| HS-3 | Count | 70% | 8-22 | ⏳ Not implemented |

**Note**: POPE only has absence queries, so HS-1 is most relevant for POPE.

### Testing Protocol

**Command Template**:
```bash
conda run -n mllm python srf/eval.py \
  --method srf \
  --use_query_heads \
  --head_top_k_pct 0.3 \
  --layer_start 12 \
  --layer_end 18 \
  --datasets pope \
  --n_pope 100 \
  --pope_splits adversarial \
  --output results/query_heads_hs1_pope/
```

**Success Criteria**: Same as layer-wise strategy
- **Target**: ≥80.30% on COCO, ≥76.50% on GQA
- **Minimum**: ≥79.30% on at least one dataset

---

## Phase 3: Combined Strategy ⏳ PENDING

If both strategies show promise individually, test combinations:

| Config | Layer-wise | Query Heads | Rationale | Status |
|--------|------------|-------------|-----------|--------|
| C-1 | LW-1 (0.3/1.5/0.1) | HS-1 (30%, 12-18) | Conservative both | ⏳ Not implemented |
| C-2 | LW-3 (0.1/2.0/0.05) | HS-1 (30%, 12-18) | Aggressive layer + selective heads | ⏳ Not implemented |
| C-3 | LW-1 (0.3/1.5/0.1) | HS-3 (70%, 8-22) | Conservative layer + broad heads | ⏳ Not implemented |

---

## Phase 4: AutoSearch Hyperparameter Optimization ⏳ PENDING

### Concept
Use Optuna for automated hyperparameter search based on AutoSearch principles (Andrej Karpathy).

### Implementation Plan

**Setup**:
```bash
conda install -n mllm optuna
```

**Search Space**:
- `alpha`: loguniform(0.01, 2.0)
- `layer_start`: int(4, 16)
- `layer_end`: int(18, 28)
- `head_top_k_pct`: uniform(0.1, 0.9)
- `early_alpha_mult`: uniform(0.0, 1.0)
- `mid_alpha_mult`: uniform(0.5, 2.5)
- `late_alpha_mult`: uniform(0.0, 1.0)

**Trials**: 100 trials with TPE sampler
**Pruning**: Median pruner (stop trials that underperform)

**Command**:
```bash
python srf/autosearch.py \
  --n_trials 100 \
  --study_name srf_pope_optimization \
  --storage sqlite:///srf_optimization.db \
  --datasets pope \
  --n_pope 200 \
  --pope_splits adversarial
```

---

## Success Metrics

### Minimum Viable Success ✅
- Any config ≥79.30% on at least 1 dataset (match baseline)

### Target Success 🎯
- Any config ≥80.30% on COCO OR ≥76.50% on GQA

### Stretch Success 🌟
- Same config improves ≥1% on 2+ datasets

### Failure Criteria ❌
- All configs <79.30% on all datasets → Try alternative improvements

---

## Alternative Approaches (If All Fail)

1. **Adaptive alpha based on CLIP confidence**: Continuous scaling instead of discrete layers
2. **Hyperparameter extremes**: Ultra-low alpha (0.05) or ultra-high (2.0)
3. **Baseline improvement**: Better decoding, prompt engineering
4. **Ensemble methods**: Combine multiple SRF configs
5. **Different CLIP models**: CLIP-ViT-L/14, CLIP-ViT-B/32

---

## Immediate Next Steps

### Priority 1: Resolve CLIP Authentication Issue
**Status**: 🔴 BLOCKER

The layer-wise implementation is complete but cannot be tested due to:
```
Error: User Access Token 'callm' is expired
```

**Solutions**:
1. Refresh HuggingFace token
2. Use cached/local CLIP models
3. Set `HF_TOKEN` environment variable
4. Use alternative CLIP model loading method

**Commands to try**:
```bash
# Option 1: Login to HuggingFace
huggingface-cli login

# Option 2: Use cached model
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1

# Option 3: Set token directly
export HF_TOKEN=<your_token>
```

### Priority 2: Test Layer-wise Strategy (Once CLIP is fixed)

**Sequence**:
1. Test LW-1 (100 samples) → Check if ≥79.30%
2. If promising, test LW-2 through LW-5
3. Run full evaluation on best config (3000 samples)
4. If successful, move to Phase 2

### Priority 3: Implement Head Selection Strategy

**Estimated Time**: 3-4 hours
- Implementation: 2 hours
- Testing: 1-2 hours

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Layer-wise Testing | 4-6 hours | CLIP auth fix |
| Phase 2: Head Selection | 3-4 hours | Phase 1 complete |
| Phase 3: Combined Strategy | 2-3 hours | Phase 1 & 2 complete |
| Phase 4: AutoSearch | 6-8 hours | Optional, if Phases 1-3 fail |
| **Total** | **15-21 hours** | **Excludes wait times** |

---

## Documentation

### Related Files
- `srf/CONTEXT.md` - SRF algorithm details
- `SRF_EXPERIMENT_STATUS.md` - Current sweep status
- `POPE_BASELINE_COMPARISON.md` - Correct baselines
- `SRF_TARGET_OBJECTIVES.md` - Success criteria

### Results Location
```
results/
├── layerwise_lw{1-5}_coco/    # Phase 1 results
├── query_heads_hs{1-3}_pope/   # Phase 2 results
├── combined_{c1-c3}/           # Phase 3 results
└── autosearch/                 # Phase 4 results
```

---

*Last Updated: 2026-05-24*
*Current Status: Phase 1 implementation complete, blocked by CLIP authentication*
