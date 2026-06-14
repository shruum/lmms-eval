# Code Guide - Architecture, Important Files, and How to Modify

**Last Updated:** 2026-06-11

---

## 🏗️ **Code Architecture Overview**

### **High-Level Pipeline**
```
Input (dataset + question) → SRF Intervention → Model Inference → Evaluation → Results
                              ↓
                         CLIP Saliency → Attention Manipulation
```

### **Directory Structure**
```
lmms-eval/
├── srf/                          # SRF method implementation
│   ├── eval.py                   # Main evaluation script (STABLE)
│   ├── srf.py                    # Core SRF function (STABLE)
│   ├── srf_e.py                  # Enhanced SRF (contrastive) (STABLE)
│   ├── saliency/                 # CLIP saliency computation
│   │   ├── clip_salience.py     # CLIP saliency (FIXED June 2026)
│   │   └── noun_extract.py      # Query noun extraction
│   ├── config.py                 # Default parameters
│   └── investigations/          # Analysis reports
├── info/                         # Dataset guides
├── results/                      # All experiment results
└── dataset/POPE_images/          # POPE benchmark data
```

---

## 📂 **Important Files - What They Do**

### **Core Evaluation Scripts**

#### **`srf/eval.py`** - Main Evaluation Entry Point
- **Purpose:** Run SRF/baseline experiments on POPE
- **Status:** STABLE - don't modify without discussion
- **Key Functions:**
  - `run_pope_vcd()` - POPE evaluation
  - `run_vlind_bench()` - VLind-Bench evaluation
  - `run_whatsup()` - WhatsUp evaluation

**Usage:**
```bash
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_name coco_adversarial \
  --alpha 0.5 --eps 0.1 \
  --layer_start 10 --layer_end 18 \
  --do_sample --temperature 0.7 --top_p 0.9
```

#### **`srf/srf.py`** - Core SRF Function
- **Purpose:** Apply SRF intervention to model
- **Status:** STABLE - don't modify without discussion
- **Key Function:** `apply_srf(model, image, attention_mask, alpha, eps, layer_start, layer_end, head_top_k_pct)`

**What it does:**
1. Compute CLIP saliency from image
2. Extract query nouns from prompt
3. For each layer in range:
   - Select top-k attention heads
   - Boost attention to salient tokens by α
   - Suppress attention to background by ε

#### **`srf/srf_e.py`** - Enhanced SRF (Contrastive)
- **Purpose:** Two-pass contrastive SRF (like VCD)
- **Status:** STABLE - not currently used
- **Difference:** Uses original + noisy images for contrastive decoding

### **Saliency Computation**

#### **`srf/saliency/clip_salience.py`** - CLIP Saliency
- **Purpose:** Compute CLIP-based saliency maps
- **Status:** FIXED (June 2026) - handles tuples, long strings
- **Key Function:** `compute_clip_salience(image, noun_or_text, ...)`

**Recent Fixes:**
- ✅ Handles tuple inputs (noun_A, noun_B) from WhatsUp
- ✅ Handles pre-extracted nouns (doesn't re-extract)
- ✅ Fixed entropy calculation

**Metrics Implemented:**
- Entropy, peak-to-mean, coverage
- Gini, KL divergence, top-K concentration

#### **`srf/saliency/noun_extract.py`** - Query Noun Extraction
- **Purpose:** Extract nouns from questions for CLIP queries
- **Status:** STABLE
- **Dataset-specific functions:**
  - `_pope()` - POPE noun extraction
  - `_vlind()` - VLind-Bench noun extraction
  - `_whatsup()` - WhatsUp spatial relation extraction (returns tuple)

### **Configuration**

#### **`srf/config.py`** - Default Parameters
- **Purpose:** Store default hyperparameters
- **Status:** STABLE
- **Defaults:**
  ```python
  ALPHA = 0.5          # Boosting strength
  EPS = 0.1            # Suppression strength
  LAYER_START = 10     # First fusion layer
  LAYER_END = 18       # Last fusion layer
  HEAD_TOP_K_PCT = 0.5 # Head selection (50%)
  ```

---

## 🛠️ **How to Run Experiments**

### **Baseline Experiments**
```bash
# POPE - Single split
python srf/eval.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_file /path/to/coco_adversarial.json \
  --pope_vcd_name coco_adversarial \
  --pope_image_dir /path/to/images \
  --do_sample --temperature 0.7 --top_p 0.9 \
  --output results/baseline_test/

# All 9 splits (use script)
bash run_all_pope_baselines.sh
```

### **SRF Experiments**
```bash
# Single config
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope_vcd \
  --pope_vcd_name coco_adversarial \
  --alpha 0.5 --eps 0.1 \
  --layer_start 10 --layer_end 18 \
  --head_top_k_pct 0.5 \
  --do_sample --temperature 0.7 --top_p 0.9 \
  --output results/srf_test/

# Parameter sweep
bash launch_srf_sweep.sh  # Custom script for multiple configs
```

### **Monitoring Experiments**
```bash
# Check GPU usage
nvidia-smi

# Check experiment progress
tail -f results/srf_test/run.log

# Parse results
python srf/parse_results.py results/srf_test/
```

---

## ⚠️ **CRITICAL RULES - Don't Break These**

### **🔴 CODE MODIFICATION POLICY**

1. **DON'T MODIFY STABLE FILES**
   - `srf/srf.py` - Core SRF (tested and working)
   - `srf/srf_e.py` - Enhanced SRF (not used but stable)
   - `srf/eval.py` - Main evaluation (entry point)

2. **DO NOT HARD-CODE HYPER-PARAMETERS**
   - Use CLI arguments for all parameter testing
   - Default values from `srf/config.py`
   - Never hard-code alpha, layers, etc. in code

3. **WHEN TESTING IDEAS**
   - Keep core files unchanged
   - Create wrapper functions/files
   - Pass parameters via CLI
   - Name experimental files clearly: `srf_experimental_v2.py`

4. **ALWAYS TEST ON SMALL DATA FIRST**
   - Use `--limit 10` to test on 10 samples
   - Verify output format
   - Check for errors before full run

**Rationale:** Ensures reproducibility and prevents experimental code from polluting stable codebase.

---

## 🐛 **Debugging Guide**

### **Common Issues & Solutions**

#### **Issue 1: SRF has no effect (identical to baseline)**
**Symptoms:** SRF accuracy = baseline accuracy
**Possible Causes:**
1. Alpha too low (try α=1.0, 2.0)
2. Wrong layer range (try 5-10, 12-18)
3. CLIP saliency not aligned with model attention
4. Head selection too conservative (try 0.7, 0.9)

**Debug Steps:**
```bash
# Check saliency quality
python srf/debug_saliency.py --image path/to/image.jpg --noun "dog"

# Check attention manipulation
python srf/debug_attention.py --layer 15 --head 5

# Visualize saliency
python srf/visualize_saliency.py --image path/to/image.jpg --noun "dog"
```

#### **Issue 2: SRF degrades performance**
**Symptoms:** SRF accuracy < baseline accuracy
**Possible Causes:**
1. Alpha too high (over-boosting)
2. Eps too high (over-suppressing)
3. Wrong layer range (disrupting fusion)
4. Head selection too aggressive

**Solutions:**
- Reduce alpha (try 0.1-0.5)
- Reduce eps (try 0.0-0.05)
- Narrower layer range (try 10-15, 12-16)
- More conservative head selection (try 0.3-0.5)

#### **Issue 3: CLIP saliency crashes**
**Symptoms:** TypeError in clip_salience.py
**Recent Fix:** June 2026 - now handles tuples, long strings
**If still crashes:**
```bash
# Test saliency directly
python -c "
from srf.saliency.clip_salience import compute_clip_salience
from PIL import Image
img = Image.open('path/to/image.jpg')
result = compute_clip_salience(img, 'dog')
print(result)
"
```

#### **Issue 4: Noun extraction fails**
**Symptoms:** Wrong noun extracted, crashes on special datasets
**Check:**
```bash
# Test noun extraction
python -c "
from srf.saliency.noun_extract import extract_query_noun
print(extract_query_noun('Is there a dog in the image?', 'pope'))
print(extract_query_noun('Which is larger, dog or cat?', 'whatsup'))
"
```

---

## 📊 **Results & Output**

### **Result File Format**
```json
{
  "method": {
    "0.0": {
      "accuracy": 0.7930,
      "precision": 0.8234,
      "recall": 0.8123,
      "f1": 0.8178,
      "yes_ratio": 0.5023
    }
  },
  "baseline": {
    "0.0": {
      "accuracy": 0.7930,
      ...
    }
  }
}
```

### **Reading Results**
```bash
# Quick accuracy check
cat results/experiment/pope_coco_adversarial.json | jq '.method."0.0".accuracy'

# Full results table
python srf/generate_results_table.py results/experiment/

# Compare multiple experiments
python srf/compare_experiments.py results/exp1/ results/exp2/
```

---

## 🔬 **Experiment Scripts**

### **Monitoring Scripts**
```bash
# Check sweep progress
bash check_sweep_detailed.sh

# Monitor GPUs
watch -n 10 nvidia-smi

# Find top results
python find_top_results.py results/srf_focused_sweep/
```

### **Launch Scripts**
```bash
# Launch diagnostic sweep
bash launch_diagnostic_round.sh

# Launch GQA sweep
bash launch_gqa_sweep.sh

# Custom sweep (modify this)
bash launch_custom_sweep.sh
```

---

## 🎯 **Best Practices**

### **For Running Experiments**
1. **Always use --limit first** to test on small data
2. **Monitor GPU usage** - don't overload
3. **Check output format** before full run
4. **Save run logs** for debugging
5. **Document parameters** in filename/folder

### **For Modifying Code**
1. **Copy stable file** to experimental version
2. **Modify copy**, not original
3. **Test thoroughly** before committing
4. **Document changes** in commit message
5. **Revert if doesn't work**

### **For Debugging**
1. **Start with small data** (limit=10)
2. **Check each component** separately
3. **Visualize saliency** to verify quality
4. **Compare with baseline** to isolate issue
5. **Use print statements** to trace execution

---

## 📚 **Related Documentation**

- **PROJECT_OVERVIEW.md** - Project goals and context
- **RESULTS_COMPENDIUM.md** - All results in one place
- **LITERATURE_REVIEW.md** - Other methods and papers
- **NEXT_STEPS.md** - Current priorities
- `info/POPE_LLAVA_GUIDE.md` - POPE-specific guide

---

## 🔑 **Key Takeaways**

1. **Core files are STABLE** - don't modify without discussion
2. **Use CLI arguments** - never hard-code parameters
3. **Test on small data first** - use --limit
4. **Debug systematically** - check each component
5. **Document everything** - filenames, logs, parameters

---

*For project goals: see PROJECT_OVERVIEW.md*
*For all results: see RESULTS_COMPENDIUM.md*
*For current priorities: see NEXT_STEPS.md*
