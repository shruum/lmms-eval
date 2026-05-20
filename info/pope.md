---
type: research
tags: [pope, srf, eval, hallucination, qwen, results]
---

# POPE Evaluation — Full Benchmark Plan

## What papers actually report

Papers (ClearSight 2503.13107, AIR 2602.24041, VCD 2311.16922) evaluate POPE on
**3 separate image datasets**: MSCOCO, A-OKVQA, and GQA.

For each dataset, 3 question splits (sampling strategies for absent objects):
- **Random** — randomly sampled objects not in the image
- **Popular** — frequently occurring objects that are absent
- **Adversarial** — semantically related objects that are absent (hardest)

→ **9 combinations total** (3 datasets × 3 splits), each reporting **Accuracy + F1**.

---

## Table structure from papers

From ClearSight (2503.13107) Table 2 — LLaVA-1.5-7B:

| | MSCOCO | | A-OKVQA | | GQA | |
|--|--|--|--|--|--|--|
| | Acc | F1 | Acc | F1 | Acc | F1 |
| Random baseline | 88.2 | 87.4 | 87.6 | 87.6 | 88.0 | 88.2 |
| VCD | 88.5 | 87.6 | 87.7 | 87.8 | 88.6 | 88.8 |
| **VAF** | **89.8** | **89.4** | **89.4** | **89.1** | **89.5** | **89.4** |
| Popular baseline | 86.1 | 85.5 | 81.9 | 82.8 | 79.4 | 81.1 |
| VCD | 86.3 | 85.8 | 82.1 | 83.1 | 79.9 | 81.7 |
| **VAF** | **87.5** | **87.4** | **84.2** | **84.6** | **81.8** | **82.9** |
| Adversarial baseline | 82.3 | 82.1 | 74.3 | 77.1 | 76.3 | 78.9 |
| VCD | 82.3 | 82.4 | 72.4 | 76.7 | 75.2 | 78.3 |
| **VAF** | **83.4** | **82.6** | **77.2** | **79.2** | **79.7** | **81.2** |

VAF is the closest method to SRF (both do attention logit boosting, same F.softmax patch approach).

---

## Annotation files (question JSONs)

Source: **AoiDragon/POPE** GitHub repo (the original POPE paper repo).

```
https://github.com/AoiDragon/POPE/tree/main/output/
├── coco/
│   ├── coco_pope_adversarial.json   (JSONL, one object per line)
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
└── seem/
    ├── aokvqa/
    │   ├── aokvqa_pope_seem_adversarial.json  (JSON array)
    │   ├── aokvqa_pope_seem_popular.json
    │   └── aokvqa_pope_seem_random.json
    └── gqa/
        ├── gqa_pope_seem_adversarial.json     (JSON array)
        ├── gqa_pope_seem_popular.json
        └── gqa_pope_seem_random.json
```

Each entry: `{"question_id": int, "image": "<filename>", "text": "Is there a X?", "label": "yes"|"no"}`

Note: COCO files are JSONL (one JSON per line); AOKVQA/GQA files are JSON arrays.

---

## Image sources — verified

Confirmed by checking all 9 annotation files:

| Dataset | Image source | # unique images | Image filename format |
|---------|-------------|-----------------|----------------------|
| COCO    | COCO val2014 | 500 | `COCO_val2014_000000XXXXXX.jpg` |
| A-OKVQA | COCO val2014 | 500 | `COCO_val2014_000000XXXXXX.jpg` |
| GQA     | GQA images  | 500 | `XXXXXXX.jpg` (e.g. `2405722.jpg`) |

- COCO and A-OKVQA use the **same image format** (COCO val2014) but **different images** — only 11 of 500 overlap
- To run both COCO + A-OKVQA: need **989 unique COCO val2014 images** (not the full 40k dataset)
- GQA uses its own separate image set (Visual Genome / GQA)

---

## How other papers run all 9

They have the COCO val2014 folder downloaded locally as part of standard LLaVA eval setup:
```
playground/data/eval/pope/val2014/   # full COCO val2014 (~40k images, ~6GB)
```
This covers both COCO and A-OKVQA splits. GQA images are a separate download.

---

## Download instructions (run on remote server)

### Step 1 — Download the 9 POPE annotation JSON files

```bash
mkdir -p ~/pope_data/coco ~/pope_data/aokvqa ~/pope_data/gqa

# COCO (JSONL format)
for split in adversarial popular random; do
  wget -q "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_${split}.json" \
       -O ~/pope_data/coco/coco_pope_${split}.json
done

# A-OKVQA (JSON array format)
for split in adversarial popular random; do
  wget -q "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/seem/aokvqa/aokvqa_pope_seem_${split}.json" \
       -O ~/pope_data/aokvqa/aokvqa_pope_seem_${split}.json
done

# GQA (JSON array format)
for split in adversarial popular random; do
  wget -q "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/seem/gqa/gqa_pope_seem_${split}.json" \
       -O ~/pope_data/gqa/gqa_pope_seem_${split}.json
done
```

### Step 2 — Download COCO val2014 images (covers COCO + A-OKVQA, 6/9 combos)

**Option A — Full download (~6.3 GB, recommended if disk space available):**
```bash
mkdir -p ~/pope_data/images/val2014
wget http://images.cocodataset.org/zips/val2014.zip -O /tmp/val2014.zip
unzip /tmp/val2014.zip -d ~/pope_data/images/
# → ~/pope_data/images/val2014/COCO_val2014_000000XXXXXX.jpg
```

**Option B — Download only the 989 images needed for POPE (much faster, ~150 MB):**
```bash
mkdir -p ~/pope_data/images/val2014

# Extract image list from the 6 COCO+AOKVQA JSON files
python3 - << 'PYEOF'
import json, os, urllib.request

data_dir = os.path.expanduser("~/pope_data")
imgs = set()

# COCO files are JSONL
for split in ["adversarial", "popular", "random"]:
    path = f"{data_dir}/coco/coco_pope_{split}.json"
    for line in open(path):
        imgs.add(json.loads(line)["image"])

# AOKVQA files are JSON arrays
for split in ["adversarial", "popular", "random"]:
    path = f"{data_dir}/aokvqa/aokvqa_pope_seem_{split}.json"
    for r in json.load(open(path)):
        imgs.add(r["image"])

print(f"Downloading {len(imgs)} COCO val2014 images...")
out_dir = f"{data_dir}/images/val2014"
os.makedirs(out_dir, exist_ok=True)
base = "http://images.cocodataset.org/val2014/{}"
for i, img in enumerate(sorted(imgs)):
    dst = f"{out_dir}/{img}"
    if not os.path.exists(dst):
        urllib.request.urlretrieve(base.format(img), dst)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(imgs)}")
print("Done.")
PYEOF
```

### Step 3 — Download GQA images (GQA split, 3/9 combos, ~20 GB)

```bash
# Full GQA image archive (only option — no per-image URLs available)
mkdir -p ~/pope_data/images/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O /tmp/gqa_images.zip
unzip /tmp/gqa_images.zip -d ~/pope_data/images/gqa/
# → ~/pope_data/images/gqa/images/XXXXXXX.jpg

# OR: extract only the 500 images needed after downloading
python3 -c "
import json, zipfile, os
data_dir = os.path.expanduser('~/pope_data')
imgs = set()
for split in ['adversarial', 'popular', 'random']:
    for r in json.load(open(f'{data_dir}/gqa/gqa_pope_seem_{split}.json')):
        imgs.add(r['image'])
print(f'Need {len(imgs)} GQA images:', sorted(imgs)[:5])
# Then extract only those from the zip
"
```

### Expected final layout

```
~/pope_data/
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
    ├── val2014/          ← COCO val2014 (989 or 40k images)
    │   └── COCO_val2014_000000XXXXXX.jpg
    └── gqa/              ← GQA images (500 or full set)
        └── images/
            └── XXXXXXX.jpg
```

---

## Current status

| Dataset | Images | Annotation JSONs | Status |
|---------|--------|-----------------|--------|
| COCO    | In `lmms-lab/POPE` HF (embedded PIL) | Not needed (HF has Q+A+image) | **Ready** |
| A-OKVQA | Need COCO val2014 download | On AoiDragon/POPE GitHub | **Needs 989 COCO images** |
| GQA     | Need GQA images download | On AoiDragon/POPE GitHub | **Needs 500 GQA images** |

---

## ✅ Actual Implementation (May 2026)

**Location:** `/home/anna2/shruthi/dataset/POPE_images/`

**What was downloaded:**
- 9 JSON files (COCO, A-OKVQA, GQA × 3 splits each)
- 989 COCO val2014 images (covers COCO + A-OKVQA)
- 500 GQA images

**Scripts:**
- `extract_gqa.py` - Extract 500 GQA images from full zip
- `monitor_and_extract.sh` - Monitor download + auto-extract
- `coco_aokvqa_images.txt` - 989 COCO image filenames
- `gqa_images.txt` - 500 GQA image filenames

**Followed pope.md plan (Option B - selective download)** ✓

---

## What we can run now

`lmms-lab/POPE` (already cached) gives **COCO × 3 splits = 9,000 questions**.
Fields: `question_id`, `question`, `answer`, `image` (PIL), `category` (adversarial/popular/random).

This is the most important table column — all compared papers lead with MSCOCO.

---

## How evaluation works

### Answer extraction (from VCD / LLaVA eval)

```python
text = text.split('.')[0]        # keep first sentence
text = text.replace(',', '')
words = text.split(' ')
if 'No' in words or 'not' in words or 'no' in words:
    pred = 'no'
else:
    pred = 'yes'
```

### Metrics

```python
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
accuracy  = (TP + TN) / total
yes_ratio = pred_yes / total    # balanced model → ~0.5
```

---

## Our current eval vs paper format

| | Current autoresearch | Paper-ready |
|--|--|--|
| Dataset | COCO adversarial only | COCO × 3 splits (+ AOKVQA/GQA if images downloaded) |
| n | 200 (subsampled) | 3,000 per split (full) |
| Answer method | logit(yes) vs logit(no) | `generate()` → text parse |
| Metrics | Accuracy only | Acc + F1 + Yes% |

**Current numbers** (200 adversarial, logit method):
- Baseline: 89.50%
- SRF: 90.50% (+1.00pp)

---

## File locations

### POPE Dataset (Downloaded - May 2026)
```
/home/anna2/shruthi/dataset/POPE_images/
├── coco_{adversarial,popular,random}.json       (COCO annotations, JSONL)
├── aokvqa_{adversarial,popular,random}.json     (A-OKVQA annotations, JSON array)
├── gqa_{adversarial,popular,random}.json        (GQA annotations, JSON array)
├── coco_aokvqa_images.txt                       (989 COCO image list)
├── gqa_images.txt                               (500 GQA image list)
├── extract_gqa.py                               (GQA extraction script)
├── monitor_and_extract.sh                       (GQA download monitor)
└── images/
    ├── val2014/                                 (989 COCO val2014 images)
    └── gqa/                                     (500 GQA images)
```

### Evaluation Scripts
| File | Purpose |
|------|---------|
| `lmms_eval/tasks/pope/pope_{adv,pop,random}.yaml` | lmms-eval task YAMLs |
| `lmms_eval/tasks/pope/utils.py` | metrics: Acc, Precision, Recall, F1, Yes% |
| `LLaVA/llava/eval/eval_pope.py` | LLaVA-style eval (VCD format, local images) |
| `my_analysis/autoresearch/pope_eval_all.py` | Existing harness: COCO×3 splits, generate() |
| `srf/eval_pope_vcd_fixed.py` | SRF POPE evaluation (VCD format, local images) |
| `test_pope_quick.py` | Quick sanity test (2 samples per dataset) |

### Running Scripts (Shell Scripts - Production)
All scripts located in `/home/anna2/shruthi/lmms-eval/`:

| Script | Dataset | Method | Splits | GPU | Purpose |
|--------|---------|--------|--------|-----|---------|
| `run_pope_coco.sh` | COCO | Baseline | adversarial, popular, random | 3 | Run all 3 COCO splits baseline |
| `run_pope_coco_srf.sh` | COCO | SRF | adversarial, popular, random | 3 | Run all 3 COCO splits SRF (VAF-like params) |
| `run_pope_aokvqa.sh` | A-OKVQA | Baseline | adversarial, popular, random | 3 | Run all 3 A-OKVQA splits baseline |
| `run_pope_aokvqa_srf.sh` | A-OKVQA | SRF | adversarial, popular, random | 3 | Run all 3 A-OKVQA splits SRF (VAF-like params) |
| `run_pope_gqa.sh` | GQA | Baseline | adversarial, popular, random | 3 | Run all 3 GQA splits baseline |
| `run_pope_gqa_srf.sh` | GQA | SRF | adversarial, popular, random | 2 | Run all 3 GQA splits SRF (VAF-like params) |

**Usage:**
```bash
# Navigate to repo
cd /home/anna2/shruthi/lmms-eval

# Run baseline for all GQA splits
bash run_pope_gqa.sh

# Run SRF for all GQA splits
bash run_pope_gqa_srf.sh

# Check GPU usage first
nvidia-smi
```

**Script Details:**
- **Environment:** Uses conda environment `mllm`
- **Model:** `llava-hf/llava-1.5-7b-hf`
- **Dataset location:** `/home/anna2/shruthi/dataset/POPE_images/`
- **Output location:** `/home/anna2/shruthi/lmms-eval/results/llava_pope_{dataset}{,_srf}/`
- **Log files:** Each split logs to `{output_dir}/{split}/srf.log` or `baseline.log`
- **SRF parameters (VAF-like):** α=0.15, layers 10-15, 50% heads (from ClearSight paper)

### Usage with SRF Evaluation

**Run all 3 datasets (COCO, A-OKVQA, GQA) × 3 splits:**
```bash
# Baseline
python srf/eval.py \
  --method baseline \
  --datasets pope_vcd \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco_adversarial.json \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --output results/pope_baseline/

# SRF
python srf/eval.py \
  --method srf \
  --datasets pope_vcd \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco_adversarial.json \
  --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
  --alpha 0.15 \
  --layer_start 10 \
  --layer_end 15 \
  --output results/pope_srf/
```

**Run all 9 combinations (3 datasets × 3 splits):**
```bash
# COCO
for split in adversarial popular random; do
  python srf/eval.py \
    --method srf \
    --datasets pope_vcd \
    --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/coco_${split}.json \
    --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
    --output results/pope_coco_${split}_srf/
done

# A-OKVQA (uses same COCO images)
for split in adversarial popular random; do
  python srf/eval.py \
    --method srf \
    --datasets pope_vcd \
    --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/aokvqa_${split}.json \
    --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/val2014 \
    --output results/pope_aokvqa_${split}_srf/
done

# GQA (uses GQA images)
for split in adversarial popular random; do
  python srf/eval.py \
    --method srf \
    --datasets pope_vcd \
    --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_${split}.json \
    --pope_image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
    --output results/pope_gqa_${split}_srf/
done
```

**Quick sanity test (2 samples per dataset):**
```bash
python test_pope_quick.py
```

---

## Implementation Summary (May 2026)

### ✅ Completed
- [x] Downloaded all 9 POPE annotation JSON files (COCO, A-OKVQA, GQA × 3 splits)
- [x] Downloaded 989 COCO val2014 images (covers COCO + A-OKVQA)
- [x] Downloaded and extracted 500 GQA images
- [x] Created extraction scripts for GQA (automated workflow)
- [x] Verified JSON formats and image counts
- [x] Documented all script locations and usage

### 📊 Dataset Coverage
| Dataset | Questions | Images | Splits | Status |
|---------|-----------|--------|--------|--------|
| COCO    | 9,000    | 500    | 3      | ✅ Ready |
| A-OKVQA | 9,000    | 500    | 3      | ✅ Ready |
| GQA     | 9,000    | 500    | 3      | ✅ Ready |
| **Total** | **27,000** | **1,489** | **9** | ✅ **Complete** |

### 🎯 Ready for Evaluation
All 9 POPE combinations (3 datasets × 3 splits) are ready for SRF evaluation:
- **27,000 questions** total
- **1,489 unique images** (989 COCO + 500 GQA)
- **JSON formats verified** (JSONL for COCO, JSON array for A-OKVQA/GQA)
- **Scripts documented** and ready to use

### 📝 Notes
- Followed the "Option B" selective download approach from pope.md
- GQA extraction automated for efficiency (extract only 500 needed images from 20GB zip)
- Image lists generated for verification and batch processing
- Compatible with both LLaVA-style eval and SRF evaluation scripts
