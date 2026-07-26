# SRF AutoResearch - Quick Start Guide

## Inspired by Karpathy's AutoResearch

**Key concept:** AI agents modify code and run experiments autonomously while you sleep.

---

## Files (Karpathy-style Structure)

```
srf/
├── prepare.py           # [FIXED] Utilities for loading, evaluation - DO NOT MODIFY
├── srf.py               # [AGENT MODIFIES] Core SRF algorithm - agent edits this
├── program.md           # [YOU EDIT] Instructions for the AI agent
└── run_baseline.sh      # Start baseline experiment
```

---

## How to Use

### Option A: Manual AutoResearch (You control the loop)

**Step 1: Run baseline**
```bash
export CUDA_VISIBLE_DEVICES=1
python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output results/baseline_llava7b/
```

**Step 2: Test one variation**
```bash
# Example: Try 5×5 CLIP grid
python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --clip_coarse_grid 5 \
    --output results/test_5x5/
```

**Step 3: Compare results**
- If improvement > 0.5%, keep the change
- If worse, revert and try next variation

### Option B: Autonomous AutoResearch (Claude Code controls the loop)

**Give Claude Code these instructions:**

```
Hi! Please read srf/program.md and help me run the SRF autoresearch loop.

We're testing improvements on LLaVA-7B + POPE adversarial:
- Baseline: ~84% accuracy
- Target: >85% accuracy
- GPU: 1 or 2

Please:
1. Start with the baseline experiment
2. Try variations from program.md (Option 1 and Option 2)
3. Run 3-5 experiments
4. Report which configuration works best
```

---

## Experimental Variations

### Quick Wins (Easy to test)

| # | Variation | Command |
|---|-----------|---------|
| 1 | **5×5 CLIP grid** | `--clip_coarse_grid 5` |
| 2 | **9×9 CLIP grid** | `--clip_coarse_grid 9` |
| 3 | **Stronger boost** | `--alpha 4.0 --eps 0.2` |
| 4 | **Weaker boost** | `--alpha 1.5 --eps 0.1` |
| 5 | **No absence-aware** | `--clip_suppress_thresh 0.0` |

### Advanced (Need code changes)

| # | Variation | What to modify |
|---|-----------|-----------------|
| 6 | **Multi-scale ensemble** | `srf/saliency/clip_salience.py` |
| 7 | **Graduated boost** | `srf/srf.py` - `prepare_sample()` |
| 8 | **Better CLIP model** | `srf/saliency/clip_salience.py` - CLIP model |
| 9 | **Text token suppression** | `srf/srf.py` - add text suppression |

---

## Tracking Results

Each experiment outputs to `results/[experiment_name]/`

Check accuracy:
```bash
# Find accuracy in output
grep -i "accuracy" results/test_5x5/output.txt

# Compare across experiments
ls -la results/*/
```

---

## Expected Timeline

**Today:**
- ✓ Run baseline (10 min)
- ✓ Test 2-3 variations (30 min)
- ✓ Identify promising directions

**Tonight:**
- 🚀 Let Claude run 10-20 experiments autonomously
- 📊 Check results in morning

**Tomorrow:**
- 🏆 Keep winners, discard losers
- 📈 Final results + write-up

---

## Troubleshooting

**CUDA error:**
```bash
export CUDA_VISIBLE_DEVICES=1  # or 2
```

**Model download taking too long:**
```bash
# Set HF cache
export HF_HOME=/path/to/huggingface_cache
```

**Out of memory:**
```bash
# Use smaller batch size or different GPU
export CUDA_VISIBLE_DEVICES=2
```

---

## Credits

- Based on [@karpathy](https://github.com/karpathy)'s [autoresearch](https://github.com/karpathy/autoresearch)
- Adapted for SRF hallucination mitigation research
