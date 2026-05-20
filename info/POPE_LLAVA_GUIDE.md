# POPE & LLaVA Evaluation — Complete Setup & Running Guide

## Quick Start Summary

This guide covers **POPE (Polling-based Object Probing Evaluation)** on **LLaVA-1.5-7B** using SRF hallucination mitigation.

### What is POPE?
- **Purpose**: Evaluates object hallucination in VLMs
- **Method**: Yes/No questions about objects that may or may not be present
- **Datasets**: MSCOCO, A-OKVQA, GQA (3 splits each: Random, Popular, Adversarial)
- **Metric**: Accuracy, F1-score, Precision, Recall

### What is LLaVA?
- **Model**: Large Language and Vision Assistant
- **Version**: LLaVA-1.5-7B-HF (HuggingFace format)
- **Size**: 7 billion parameters
- **Architecture**: Vision encoder + Language model

---

## 📦 Complete Setup Guide

### Step 1: Environment Setup

```bash
# Activate conda environment
conda activate mllm

# Set up environment variables
export HF_HOME=/home/anna2/shruthi/hf_cache  # HuggingFace cache
export CUDA_VISIBLE_DEVICES=0  # GPU to use

# Navigate to project
cd /home/anna2/shruthi/lmms-eval
```

### Step 2: Download POPE Dataset Files

```bash
# Create dataset directory
mkdir -p /home/anna2/shruthi/dataset/POPE_images
cd /home/anna2/shruthi/dataset/POPE_images

# Download POPE annotation files
git clone https://github.com/AoiDragon/POPE.git temp_pope

# Organize annotation files
mkdir -p coco aokvqa gqa
mv temp_pope/output/coco/*.json coco/
mv temp_pope/output/seem/aokvqa/*.json aokvqa/
mv temp_pope/output/seem/gqa/*.json gqa/

# Clean up
rm -rf temp_poe
```

### Step 3: Download Images

#### Option A: Full COCO Dataset (Recommended)
```bash
# Download full COCO val2014 (~6.3GB)
cd /home/anna2/shruthi/dataset/POPE_images
mkdir -p images/val2014

wget http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
unzip /tmp/val2014.zip -d images/
# Results in: images/val2014/COCO_val2014_000000XXXXXX.jpg
```

#### Option B: Minimal Download (Faster)
```bash
# Download only required images (~150MB)
cd /home/anna2/shruthi/dataset/POPE_images

python3 << 'EOF'
import json, os, urllib.request

# Extract image filenames from annotation files
images_needed = set()

for dataset in ["coco", "aokvqa"]:
    for split in ["adversarial", "popular", "random"]:
        file_path = f"{dataset}/coco_pope_{split}.json" if dataset == "coco" else f"{dataset}/aokvqa_pope_seem_{split}.json"
        
        with open(file_path) as f:
            if dataset == "coco":
                for line in f:
                    data = json.loads(line)
                    images_needed.add(data["image"])
            else:
                data = json.load(f)
                for item in data:
                    images_needed.add(item["image"])

print(f"Found {len(images_needed)} unique images needed")

# Download images (requires COCO images to be accessible)
# This would need access to COCO server or local copy
print("Image filenames extracted. Download from COCO server or use full val2014.zip")
EOF
```

#### Step 4: Download GQA Images
```bash
# GQA uses different images
cd /home/anna2/shruthi/dataset/POPE_images
mkdir -p images/gqa

# Download GQA images
wget https://storage.googleapis.com/gqa/images.zip -O /tmp/gqa_images.zip
unzip /tmp/gqa_images.zip -d images/gqa/
```

### Step 5: Verify Dataset Structure
```bash
# Expected structure:
/home/anna2/shruthi/dataset/POPE_images/
├── coco/
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── aokvqa/
│   ├── aokvqa_pope_seem_adversarial.json
│   ├── aokvqa_pope_seem_popular.json
│   └── aokvqa_pope_seem_random.json
├── gqa/
│   ├── gqa_pope_seem_adversarial.json
│   ├── gqa_pope_seem_popular.json
│   └── gqa_pope_seem_random.json
└── images/
    ├── val2014/          # COCO images (for COCO + A-OKVQA)
    │   └── COCO_val2014_000000XXXXXX.jpg
    └── gqa/              # GQA images
        └── XXXXXXX.jpg
```

---

## 🚀 Running POPE on LLaVA

### Quick Test (5 samples)
```bash
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco/coco_pope_adversarial.json \
  --pope_vcd_name coco_adversarial \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --calib_dataset pope \
  --eval_method generation \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/quick_test/
```

### Full COCO Evaluation (All 3 splits)
```bash
# COCO Random split
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco/coco_pope_random.json \
  --pope_vcd_name coco_random \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --calib_dataset pope \
  --eval_method generation \
  --alpha 1.0 \
  --eps 0.1 \
  --phase both \
  --layer_start 10 \
  --layer_end 18 \
  --head_top_k_pct 0.5 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/llava_pope_coco/

# COCO Popular split
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco/coco_pope_popular.json \
  --pope_vcd_name coco_popular \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --calib_dataset pope \
  --eval_method generation \
  --alpha 1.0 \
  --eps 0.1 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/llava_pope_coco/

# COCO Adversarial split
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco/coco_pope_adversarial.json \
  --pope_vcd_name coco_adversarial \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --calib_dataset pope \
  --eval_method generation \
  --alpha 1.0 \
  --eps 0.1 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/llava_pope_coco/
```

### A-OKVQA Evaluation
```bash
# A-OKVQA uses same images as COCO
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/aokvqa/aokvqa_pope_seem_adversarial.json \
  --pope_vcd_name aokvqa_adversarial \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --calib_dataset pope \
  --eval_method generation \
  --alpha 1.0 \
  --eps 0.1 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/llava_pope_aokvqa/
```

### GQA Evaluation
```bash
# GQA uses different images
python srf/eval.py \
  --method srf \
  --model llava-hf/llava-1.5-7b-hf \
  --datasets pope \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa/gqa_pope_seem_adversarial.json \
  --pope_vcd_name gqa_adversarial \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --calib_dataset pope \
  --eval_method generation \
  --alpha 1.0 \
  --eps 0.1 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --output results/llava_pope_gqa/
```

---

## 🎯 Running Scripts (Convenient)

### Use existing scripts
```bash
# COCO evaluation
bash run_llava_pope_coco.sh

# A-OKVQA evaluation  
bash run_llava_pope_aokvqa.sh

# GQA evaluation
bash run_llava_pope_gqa.sh

# All POPE datasets (full sweep)
bash run_pope_all_sampling_srf.sh
```

---

## 📊 Understanding Results

### Result Files
After running, check results in:
```bash
# Results directory
ls -la results/llava_pope_coco/

# Key files:
# - pope_coco_adversarial.json  # Final results with accuracy, F1, etc.
# - run.log                      # Execution log
# - samples.json                 # Individual predictions
```

### Metrics Explained
```json
{
  "accuracy": 79.30,           // % correct yes/no answers
  "precision": 75.65,          // % "yes" answers that were correct
  "recall": 75.20,             // % actual "yes" cases found
  "f1": 75.43,                 // Harmonic mean of precision & recall
  "yes_ratio": 0.497,          // % of "yes" answers (should be ~50%)
  "n_samples": 3000            // Total questions evaluated
}
```

---

## 🔧 Important Parameters

### Decoding Parameters (CRITICAL for correct baselines)
```bash
--do_sample          # Use sampling (not greedy) - REQUIRED for paper comparison
--temperature 0.7    # Sampling temperature
--top_p 0.9          # Nucleus sampling parameter
```

### SRF Parameters
```bash
--alpha 1.0           # Attention boost strength (0.5-2.0 tested)
--eps 0.1             # Background suppression (0.0-0.2 tested)
--layer_start 10      # First layer to modify
--layer_end 18        # Last layer to modify
--head_top_k_pct 0.5  # % of heads to modify (0.3-0.9 tested)
--phase both          # Apply to both calibration and evaluation
```

### Dataset Parameters
```bash
--pope_vcd_file       # Path to POPE annotation file
--pope_vcd_name       # Dataset name for results
--pope_image_dir      # Path to images
--calib_dataset pope  # Use POPE for calibration
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "File not found" errors
```bash
# Check dataset paths
ls /home/anna2/shruthi/dataset/POPE_images/coco/
ls /home/anna2/shruthi/dataset/POPE_images/images/val2014/ | head -5

# Verify paths in command
--pope_vcd_file /full/path/to/file.json
--pope_image_dir /full/path/to/images/
```

#### 2. Out of memory errors
```bash
# Use smaller batch size or fewer samples
--n_pope 10  # Test with only 10 samples first

# Or use a different GPU
export CUDA_VISIBLE_DEVICES=1
```

#### 3. Wrong baseline accuracy
```bash
# Make sure you're using sampling decoding
--do_sample --temperature 0.7 --top_p 0.9

# Expected baselines (LLaVA-1.5-7B):
# COCO Random: 83.43%
# COCO Popular: 81.23%  
# COCO Adversarial: 79.30%
```

#### 4. Model loading issues
```bash
# Set HuggingFace cache
export HF_HOME=/home/anna2/shruthi/hf_cache

# Test model loading first
python -c "from transformers import AutoModel; AutoModel.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

---

## 📈 Current Best Results

### LLaVA-1.5-7B + SRF (May 2026)
| Dataset | Split | Baseline | Best SRF | Delta |
|---------|-------|----------|----------|-------|
| COCO    | Adversarial | 79.30% | 78.77% | -0.53% ❌ |
| GQA     | Adversarial | 75.50% | 69.43% | -6.07% ❌ |

**Status**: SRF failed to beat baseline on LLaVA-1.5-7B after 31+ hyperparameter configurations.

---

## 🔍 Debug Commands

### Test individual components
```bash
# Test CLIP saliency
python -c "from srf.saliency.clip_salience import compute_clip_saliency; print('CLIP OK')"

# Test attention patching  
python -c "from my_analysis.qwen_attn_patch import patch_model; print('Patching OK')"

# Test dataset loading
python -c "from srf.eval_datasets import load_pope_dataset; print('Dataset loading OK')"

# Quick sanity test
bash test_pope_sanity.sh
```

---

## 📚 Additional Resources

### Documentation Files
- **`info/CONTEXT.md`** - Algorithm details & parameters
- **`info/pope.md`** - POPE dataset structure & paper comparison
- **`info/POPE_BASELINE_COMPARISON.md`** - Baseline vs paper results
- **`srf/CONTEXT.md`** - SRF implementation details

### Key Scripts
- **`run_llava_pope_*.sh`** - Individual dataset scripts
- **`check_sweep_detailed.sh`** - Status monitoring
- **`final_results_analysis.py`** - Results analysis

---

*Last updated: May 2026 - LLaVA-1.5-7B + POPE Evaluation Guide*
*For questions: Check `info/` folder documentation files*