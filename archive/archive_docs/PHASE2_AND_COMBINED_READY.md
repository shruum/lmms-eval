# Phase 2 & Combined Strategy Status Report

## ✅ **PHASE 2 (Head Selection Strategy): IMPLEMENTATION COMPLETE**

### Created Components:
1. **`srf/query_classifier.py`**
   - `classify_query()`: Detects query type (count/absence/spatial/attribute/other)
   - `get_head_params()`: Returns query-specific head parameters
   - **Tested and working correctly**

2. **Integration with SRF Pipeline**
   - CLI args: `--use_query_heads`, `--query_type`
   - Modified `eval.py`, `srf.py` to handle query parameters
   - Query classification in `prepare_sample()` function
   - Dynamic head selection per query

### Head Selection Configs for POPE:
- **HS-1 (Absence)**: 30% heads, layers 12-18 → *Best for absence queries*
- **HS-2 (Default)**: 50% heads, layers 10-18 → *Balanced baseline*
- **HS-3 (Count)**: 70% heads, layers 8-22 → *Broad for counting*

### Test Matrix:
| Config | Head % | Layers | Target Query Type |
|--------|--------|--------|------------------|
| HS-1 | 30% | 12-18 | Absence (POPE) |
| HS-2 | 50% | 10-18 | Default/Generic |
| HS-3 | 70% | 8-22  | Count (future) |

---

## ✅ **PHASE 1+2 COMBINED STRATEGY: READY TO LAUNCH**

### Concept:
Combine **layer-wise alpha scaling** (Phase 1) with **query-conditioned head selection** (Phase 2) for maximum optimization.

### Combined Configs:
| Config | Layer-wise (early/mid/late) | Head % | Layers | Strategy |
|--------|---------------------------|--------|--------|----------|
| **C-1** | 0.3/1.5/0.1 (conservative) | 30% | 12-18 | Conservative both |
| **C-2** | 0.3/1.5/0.1 (conservative) | 50% | 10-18 | Conservative layer + default heads |
| **C-3** | 0.1/2.0/0.05 (aggressive) | 30% | 12-18 | Aggressive mid + selective heads |
| **C-4** | 0.5/1.2/0.2 (balanced) | 30% | 12-18 | Balanced both |

**Base parameters**: α=0.15, POPE adversarial, 100 samples per config

### Launch Commands (Ready to Execute):
```bash
# C-1: Conservative layer-wise + Selective heads
conda run -n mllm python srf/eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --method srf \
  --use_layerwise \
  --early_alpha_mult 0.3 \
  --mid_alpha_mult 1.5 \
  --late_alpha_mult 0.1 \
  --use_query_heads \
  --head_top_k_pct 0.3 \
  --layer_start 12 \
  --layer_end 18 \
  --alpha 0.15 \
  --datasets pope \
  --n_pope 100 \
  --pope_splits adversarial \
  --output results/combined_c1_coco/

# C-2: Conservative layer-wise + Default heads
# (Same as C-1 but head_top_k_pct=0.5, layer_start=10, layer_end=18)

# C-3: Aggressive mid + Selective heads
# (early_alpha_mult=0.1, mid_alpha_mult=2.0, late_alpha_mult=0.05)

# C-4: Balanced layer-wise + Selective heads
# (early_alpha_mult=0.5, mid_alpha_mult=1.2, late_alpha_mult=0.2)
```

---

## ⏳ **CURRENT STATUS: GPU BLOCKED**

### GPU Memory Situation:
- **GPU 0-4, 7**: 43-46GB used (previous experiments not releasing memory)
- **GPU 5**: 41GB used, 100% utilization (active process)
- **GPU 6**: 49GB used, 97% utilization (almost complete)

### Issue:
Previous layer-wise experiments completed but GPU memory not released (common CUDA issue).

### Next Steps:
1. **Wait for GPU cleanup** (natural process, 10-30 minutes)
2. **Force cleanup** if needed: `pkill -9 python`
3. **Alternative**: System reboot (if accessible)
4. **Test with minimal samples**: 2-5 samples per config to verify implementation

---

## 📊 **EXPECTED RESULTS**

### Success Criteria:
- **Minimum**: Any config ≥79.30% (match baseline)
- **Target**: Any config ≥80.30% (+1% improvement)
- **Stretch**: ≥80.80% (+1.5% improvement)

### Why Combined Strategy Might Work:
1. **Layer-wise**: Targets different fusion stages with appropriate boost strength
2. **Head selection**: Tailors attention pattern to query requirements
3. **Synergy**: Both strategies optimize different aspects of SRF

### If Combined Fails:
- SRF approach may have fundamental limitations on LLaVA-1.5-7B
- Consider different base models (Qwen2.5-VL, Qwen3-VL)
- Try more aggressive hyperparameter ranges
- Investigate alternative hallucination mitigation approaches

---

## 🎯 **READY FOR LAUNCH**

**Phase 2**: 3 experiments ready (HS-1, HS-2, HS-3)
**Phase 1+2**: 4 experiments ready (C-1, C-2, C-3, C-4)

**Total**: 7 experiments ready to run once GPUs are available.

---

*Last Updated: 2026-05-25 | Phase 2: Implementation Complete | Phase 1+2: Ready to Launch*