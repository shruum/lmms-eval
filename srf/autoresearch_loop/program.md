# SRF AutoResearch Program

**Inspired by:** https://github.com/karpathy/autoresearch

---

## Concept

AI agents autonomously run SRF research experiments on LLaVA-7B + POPE while you sleep.

**Key insight from Karpathy:** You're not editing Python files directly. Instead, you edit `program.md` which instructs the AI agent what to research. The agent modifies `train.py` (in our case `srf.py`), runs experiments, and keeps/discards based on results.

---

## File Structure

```
srf/
├── prepare.py           # [FIXED] Dataset loading, evaluation utilities (DO NOT MODIFY)
├── srf.py               # [AGENT MODIFIES] SRF algorithm - agent edits this file
├── program.md           # [YOU EDIT] Instructions for the AI agent
└── autoresearch.sh      # Run automated experiments
```

---

## How It Works

### The Loop (Runs Automatically)

1. **Agent reads** `program.md` for research instructions
2. **Agent modifies** `srf.py` (algorithm, hyperparameters, saliency method)
3. **Run experiment** for fixed budget (n=100 samples, ~5-10 min)
4. **Check metric** (accuracy on POPE adversarial)
5. **Keep or discard** - if accuracy improves >0.5%, save change; else revert
6. **Repeat** - run next experiment, iterate overnight

### Key Design Choices (from Karpathy)

- **Single file to modify** (`srf.py`) - keeps scope manageable, diffs reviewable
- **Fixed budget** (n=100 samples) - experiments are directly comparable
- **Simple metric** (accuracy) - easy to understand if improvement happened
- **Self-contained** - no external dependencies beyond PyTorch/transformers

---

## Quick Start

### Step 1: Run Baseline

```bash
# Establish baseline accuracy
export CUDA_VISIBLE_DEVICES=1
python srf/eval.py \
    --method srf \
    --model llava-hf/llava-1.5-7b-hf \
    --datasets pope \
    --pope_splits adversarial \
    --n_pope 100 \
    --output results/baseline_llava7b/
```

**Expected:** ~84% accuracy

### Step 2: Start AutoResearch

Give this to Claude/Claude Code:

```
Hi! Please read srf/program.md and let's start the SRF autoresearch loop.
We're testing improvements on LLaVA-7B + POPE adversarial.
```

The agent will:
1. Read `program.md` for instructions
2. Suggest modifications to `srf.py`
3. Run experiments
4. Track results
5. Keep winners

---

## Experimental Variations to Test

### Option 1: Better Salience Detection

**Problem:** CLIP ViT-B/32 with 7×7 grid is too coarse

**Variations:**
- `1A`: Use 5×5 CLIP grid (finer granularity)
- `1B`: Use 9×9 CLIP grid (coarser, more context)
- `1C`: Ensemble of 5×5, 7×7, 9×9 (multi-scale)
- `1D`: Use ViT-L/14 instead of ViT-B/32 (stronger model)

**Implementation:** Modify `clip_salience.py` in `srf/saliency/`

### Option 2: Graduated Multi-Stage Attention

**Problem:** Single α boost is too crude

**Variations:**
- `2A`: 3-stage boost (α_high=4.0, α_mid=2.0, α_low=1.0, ε=0.2)
- `2B`: Stronger simple boost (α=4.0, ε=0.2) - matches working mmvp-srf config
- `2C`: Weaker boost (α=1.5, ε=0.1) - gentler intervention
- `2D`: Text token suppression (reduce language prior)

**Implementation:** Modify `prepare_sample()` in `srf/srf.py`

---

## Tracking Results

Each experiment logs:

```json
{
  "experiment_id": "1A_multiscale_5x5",
  "timestamp": "2026-04-28T20:15:30",
  "accuracy": 85.2,
  "delta_vs_baseline": "+1.2",
  "git_commit": "abc123",
  "status": "winner"
}
```

**Winner criteria:**
- Δ > +0.5% → Keep as new baseline
- Δ > +1.0% → Major discovery
- Δ < +0.5% → Discard

---

## Safety & Constraints

**Agent can ONLY modify:**
- `srf/srf.py` (algorithm)
- `srf/saliency/clip_salience.py` (saliency computation)

**Agent CANNOT modify:**
- `srf/prepare.py` (fixed utilities)
- `srf/eval.py` (evaluation logic)
- `srf/config.py` (use CLI overrides instead)
- Dataset loading
- Model architecture

**Each experiment must:**
- Complete in < 10 minutes
- Use only n=100 samples (quick iteration)
- Run on GPU:1 or GPU:2
- Log results to `results/autoresearch_llava/`

---

## Expected Timeline

**Day 1:** Establish baseline, test 2-3 variations manually
**Night 1:** Let agent run 10-20 experiments
**Day 2:** Analyze results, keep winners, test new variations
**Night 2:** Another 10-20 experiments
**Day 3:** Final results, write-up findings

**Total:** ~50 experiments, target +1-3% improvement

---

## Troubleshooting

**If CUDA error:**
```bash
export CUDA_VISIBLE_DEVICES=1  # or 2
```

**If model loading error:**
```bash
# Check model is downloaded
python -c "from transformers import LlavaForConditionalGeneration; print('OK')"
```

**If accuracy parsing fails:**
- Check `results/[experiment_name]/output.txt`
- Manually find accuracy in output

---

## Success Criteria

✓ Find configuration achieving > 85% accuracy (≥ +1% over baseline)
✓ At least one variation shows consistent improvement
✓ Results are reproducible (same config = same accuracy)
✓ Code changes are minimal and reviewable

---

## Credits

- Inspired by [@karpathy](https://github.com/karpathy)'s [autoresearch](https://github.com/karpathy/autoresearch)
- Adapted for SRF hallucination mitigation research
- Target: NeurIPS 2026 submission
