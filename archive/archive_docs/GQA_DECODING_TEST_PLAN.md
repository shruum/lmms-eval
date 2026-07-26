# GQA Adversarial Decoding Test Plan

**Date:** 2026-05-14
**Goal:** Test if sampling decoding (do_sample=True) fixes GQA Adversarial baseline discrepancy

---

## Current Status

### Baseline Discrepancy
- **Our baseline (greedy):** 69.63% accuracy, 75.83% yes ratio (overpredicts "yes")
- **VCD paper baseline (sampling):** 75.08% accuracy, ~50% yes ratio (balanced)
- **Gap:** -5.45% accuracy, +25.83% yes ratio

### Root Cause Hypothesis
VCD paper uses **sampling decoding** (`do_sample=True`), while our evaluation uses **greedy decoding** (`do_sample=False`).

From VCD GitHub:
```python
model.generate(
    do_sample=True  # ← VCD uses sampling!
)
```

From our code:
```python
model.generate(
    do_sample=False  # ← We use greedy
)
```

---

## Code Changes Made

### 1. Modified `srf/eval_pope_vcd_fixed.py`
Added command-line arguments for sampling decoding:
- `--do_sample`: Enable sampling decoding
- `--temperature`: Temperature for sampling (default: 0.7)
- `--top_p`: Top-p for sampling (default: 0.9)

**Usage:**
```bash
# Greedy decoding (our current baseline)
conda run -n mllm python srf/eval_pope_vcd_fixed.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
  --pope_vcd_name gqa_adversarial_greedy \
  --image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --output results/gqa_greedy \
  --device cuda:0

# Sampling decoding (VCD paper method)
conda run -n mllm python srf/eval_pope_vcd_fixed.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
  --pope_vcd_name gqa_adversarial_sampling \
  --image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --output results/gqa_sampling \
  --device cuda:0 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9
```

### 2. Created `test_gqa_decoding.py`
Comprehensive test script that compares 3 decoding methods:
1. Greedy (do_sample=False) - our current baseline
2. Sampling with temp=0.7 (VCD paper default)
3. Sampling with temp=0.5 (test if lower temp helps)

**Features:**
- Tests on subset or full dataset (--n_samples 100 or --full)
- Shows sample responses for qualitative analysis
- Outputs comparison table with all metrics

---

## Test Plan

### Step 1: Quick Test (100 samples)
Compare greedy vs sampling on 100 samples to see if decoding method matters.

```bash
conda run -n mllm python test_gqa_decoding.py --n_samples 100 --device cuda:X
```

**Expected outcomes:**
- If sampling fixes overprediction: yes ratio should drop from 75% to ~50%
- If sampling doesn't matter: yes ratio stays high regardless of decoding method
- If sampling helps but not enough: yes ratio drops but not to 50%

### Step 2: Full Test (3000 samples)
If quick test shows improvement, run full GQA Adversarial dataset with both methods.

**Greedy (baseline):**
```bash
conda run -n mllm python srf/eval_pope_vcd_fixed.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
  --pope_vcd_name gqa_adversarial_greedy \
  --image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --output results/gqa_adversarial_greedy \
  --device cuda:X
```

**Sampling (VCD method):**
```bash
conda run -n mllm python srf/eval_pope_vcd_fixed.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
  --pope_vcd_name gqa_adversarial_sampling \
  --image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --output results/gqa_adversarial_sampling \
  --device cuda:X \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9
```

### Step 3: Compare Results
Compare metrics with VCD paper baseline:
- Target: 75.08% accuracy, 73.19% precision, 79.16% recall, 76.06% F1
- Success criterion: Match within ±0.5%

---

## Current Blocker

**GPU Memory Issue:**
- All GPUs (0-7) have VLLM workers using ~44GB memory each
- Only ~4.8GB free per GPU
- LLaVA-1.5-7B requires ~14GB GPU memory to load
- Model loading fails with CUDA OOM error

**Options:**
1. **Stop one VLLM worker temporarily** (recommended)
   ```bash
   # Kill one VLLM worker to free up a GPU
   kill <PID>
   ```

2. **Use CPU offloading** (very slow, not recommended for 3000 samples)
   ```bash
   # Would require code changes to use device_map="auto" with CPU offloading
   ```

3. **Wait for VLLM workers to finish** (if they're temporary jobs)

4. **Use a different model checkpoint** (smaller model like LLaVA-1.5-3B)

---

## Success Criteria

### If Sampling Fixes Overprediction:
- Yes ratio drops to ~50% (from 75.83%)
- Accuracy increases to ~75% (from 69.63%)
- Metrics match VCD paper within ±0.5%
- **Conclusion:** Decoding method was the issue

### If Sampling Doesn't Fix Overprediction:
- Yes ratio remains high (>70%)
- Accuracy remains low (<72%)
- **Conclusion:** Issue is elsewhere (dataset version, prompt format, answer parsing, etc.)

### If Sampling Partially Helps:
- Yes ratio drops but not to 50% (e.g., 60-65%)
- Accuracy increases but not to 75% (e.g., 72-73%)
- **Conclusion:** Decoding method contributes but isn't the only factor

---

## Next Steps

1. **Resolve GPU memory issue** (stop VLLM worker or wait for availability)
2. **Run quick test** (100 samples) to validate hypothesis
3. **Run full test** (3000 samples) if quick test shows improvement
4. **Document results** and update investigation documents
5. **If sampling fixes it:** Re-run all POPE baselines with sampling decoding
6. **If sampling doesn't fix it:** Investigate other potential causes (dataset version, prompt, parsing)

---

## Files Modified

1. **`/home/anna2/shruthi/lmms-eval/srf/eval_pope_vcd_fixed.py`**
   - Added `--do_sample`, `--temperature`, `--top_p` arguments
   - Modified `evaluate_pope_vcd()` to accept decoding parameters
   - Updated generation logic to support both greedy and sampling

2. **`/home/anna2/shruthi/lmms-eval/test_gqa_decoding.py`**
   - New comprehensive test script for decoding method comparison
   - Tests 3 decoding methods on same data
   - Outputs detailed comparison table and sample responses

3. **`/home/anna2/shruthi/pope.md`**
   - Added "Running Scripts" section documenting all POPE evaluation scripts
   - Included usage instructions and script details

---

## Reference Documents

- **VCD Paper:** `/home/anna2/shruthi/2311.16922v1.pdf` (Table 1: GQA Adversarial baseline)
- **GQA Investigation:** `/home/anna2/shruthi/lmms-eval/GQA_ADVERSARIAL_INVESTIGATION.md`
- **Baseline Comparison:** `/home/anna2/shruthi/lmms-eval/POPE_BASELINE_COMPARISON.md`
- **Discrepancy Investigation:** `/home/anna2/shruthi/lmms-eval/BASELINE_DISCREPANCY_INVESTIGATION.md`

---

*Last updated: 2026-05-14 — Waiting for GPU availability to test decoding methods.*
