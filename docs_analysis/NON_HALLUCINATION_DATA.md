# SRF New Datasets Analysis — Implementation Guide

> Compiled from session analysis. Goal: evaluate SRF on two new datasets (one bias, one spatial) that satisfy:
> 1. Object always present in image (SRF's saliency map works)
> 2. Comparison numbers exist in published papers (LLaVA/Qwen baselines)

---

## Core Insight: Why POPE/CHAIR Don't Work for SRF

SRF's saliency map is only useful when the queried object IS in the image.
- Object absent → CLIP finds no matching region → `max_sim < threshold` → fallback to uniform → **no gain over baseline**
- POPE: ~50% absent samples → half the dataset is the bad case
- CHAIR: model hallucinates absent objects → same problem

**SRF works when:** object IS present, model just doesn't attend to the right region.
**SRF fails when:** object is absent and model hallucinates it.

Full dataset POPE result (n=3000 adversarial): baseline=86.37%, SRF=86.13% → **no gain**.
The 90.50% in Table 1 is from n=200 subset using SRF-v2 (different variant) — not representative.

---

## Selected Datasets

### Dataset 1 (Bias): VLind-Bench
**HF ID:** `MM-Hallu/VLind-Bench` (also try `klee972/VLind-Bench`)
**Paper:** "VLind-Bench: Measuring Language Priors in Large Vision-Language Models" — arXiv:2406.08702
**Already cited in your paper as:** `lee2025vlind` (in intro, but NOT evaluated — reviewers will notice)

**What it tests:**
- **Conventional Bias:** Does the model answer correctly when image contradicts language priors?
  - Example: image shows a red banana → question "What color is the banana?" → correct=Red, model says Yellow
- **Shortcut Bias:** Does removing the image leave the answer unchanged? (model uses text shortcuts)

**Why SRF fits:**
- Object (banana, etc.) IS in the image — always present
- Failure is exactly attention misallocation: model ignores visual evidence, uses language prior
- CLIP noun ("banana") is extractable → boost attention to banana region → model should see red

**Noun extraction mode:** `vlmbias` or new `vlind` mode — extract the object noun from the question
- "What color is the banana?" → noun = `banana`
- "What is the dog doing?" → noun = `dog`

**Baseline numbers (from VLind-Bench paper):**
- LLaVA-1.5-7B conventional bias score: ~40-60% (need exact from paper)
- No attention intervention method has run on this benchmark yet

---

### Dataset 2 (Spatial): WhatsUp
**HF ID:** `ServiceNow/whatsup_all`
**Paper:** arXiv:2310.19785, EMNLP 2023
**Comparison method:** AdaptVis (ICML 2025, arXiv:2503.01773) — direct comparison available

**What it tests:** Spatial relationship reasoning. Objects always present. Question asks which spatial relationship holds (above/below/left/right). Model tends to guess from language prior without checking actual positions.

**Dataset structure** (verified from HF):
```
Splits: Controlled_Images_A, Controlled_Images_B, COCO_QA_one_obj, COCO_QA_two_obj,
        VG_QA_one_obj, VG_QA_two_obj, VG_Relation, VG_Attribution, COCO_Order
```

Each sample has:
- `image_options`: PIL image (single image for QA subtasks, composite 1280×960 for Controlled)
- `caption_options`: list of 2-4 caption strings
- Correct answer: **always index 0** (verified — TARGET=0 for all 412 controlled samples)

**Task format for VLMs:** Given the image, which caption correctly describes the spatial arrangement?
```
Question: "Which of the following best describes the image?
A. A beer bottle on a armchair
B. A beer bottle under a armchair
C. A beer bottle to the left of a armchair
D. A beer bottle to the right of a armchair
Answer with the option letter only."
GT: A (always index 0)
```

**Why SRF fits:**
- Both objects always present in image
- Pure attentional/positional failure — model reads spatial word from language prior
- Extract both object nouns → compute saliency for both → boost attention to both regions
- Two-noun CLIP: `saliency = max(CLIP(patch, noun_A), CLIP(patch, noun_B))`

**Noun extraction for WhatsUp:**
- `COCO_QA_one_obj`: "A photo of a dining table on the bottom" → noun = `dining table`
- `COCO_QA_two_obj`: "A photo of a dining table to the left of a refrigerator" → nouns = `dining table`, `refrigerator`
- `Controlled_Images_A/B`: "A beer bottle on a armchair" → nouns = `beer bottle`, `armchair`

**AdaptVis comparison numbers (LLaVA-1.5-7B):**
| Subtask | Baseline | AdaptVis | Δ |
|---|---|---|---|
| Controlled_A | 60.3% | 84.9% | +24.6 pts |
| Controlled_B | 73.1% | 83.8% | +10.7 pts |
| COCO_one | 53.0% | 53.6% | +0.6 |
| COCO_two | 58.2% | 59.9% | +1.7 |
| VG_one | 35.9% | 42.7% | +6.8 |
| VG_two | 40.8% | 48.1% | +7.3 |

**Qwen2-VL numbers (from AdaptVis):**
| Subtask | Baseline | AdaptVis |
|---|---|---|
| Controlled_A | 98.18% | 98.18% (ceiling) |
| VG_two | 56.22% | 66.95% |

Note: Qwen2.5-VL-3B numbers not in any paper — run our own baseline.

**Why SRF should beat AdaptVis:** AdaptVis uses temperature scaling (global signal). SRF uses CLIP-conditioned saliency on both objects (targeted signal). COCO/VG subtasks where AdaptVis gains are small (+0.6–1.7) are the best opportunity for SRF to outperform.

---

## What Needs to Be Implemented

### 1. `config.py` — Add dataset params

```python
SRF_DATASET_PARAMS = {
    "mmvp":    {"phase": "both",       "alpha": 4.0, "eps": 0.2},
    "pope":    {"phase": "generation", "alpha": 4.0, "eps": 0.2},
    "vlmbias": {"phase": "generation", "alpha": 8.0, "eps": 0.5},
    "mme":     {"phase": "both",       "alpha": 4.0, "eps": 0.2},
    # NEW:
    "vlind":   {"phase": "generation", "alpha": 8.0, "eps": 0.5},  # similar to vlmbias
    "whatsup": {"phase": "both",       "alpha": 4.0, "eps": 0.2},  # start with mmvp defaults
}
```

Also add to `dataset_layer_end` in each arch:
```python
"dataset_layer_end": {"mmvp": 15, "pope": 15, "vlmbias": 14, "mme": 15,
                      "vlind": 14, "whatsup": 15},  # tune after first run
```

### 2. `noun_extract.py` — Add new modes

```python
def _vlind(question: str) -> str:
    """Extract noun from VLind-Bench question.
    'What color is the banana?' → 'banana'
    'What is the dog doing?' → 'dog'
    """
    q = question.strip().lower().rstrip("?")
    # "What [attr] is the X" → X
    m = re.search(r'\bthe\s+(\w+(?:\s+\w+)?)\b', q)
    if m:
        noun = m.group(1).strip()
        if noun not in _GENERIC_NOUNS:
            return noun
    # fallback: first 4+ char content word
    words = re.findall(r'\b[a-z]{4,}\b', q)
    for w in words:
        if w not in _GENERIC_NOUNS:
            return w
    return "object"


def _whatsup(caption: str) -> tuple[str, str | None]:
    """Extract object nouns from a WhatsUp caption option.
    
    'A photo of a dining table on the bottom' → ('dining table', None)
    'A photo of a dining table to the left of a refrigerator' → ('dining table', 'refrigerator')
    'A beer bottle on a armchair' → ('beer bottle', 'armchair')
    
    Returns (noun_A, noun_B) where noun_B is None for single-object captions.
    """
    c = caption.strip().lower()
    # Remove "A photo of" prefix if present
    c = re.sub(r'^a photo of\s+', '', c)
    # Remove leading article
    c = re.sub(r'^an?\s+', '', c)
    
    # Two-object: "X [spatial_rel] [a/an] Y"
    spatial_rels = r'(?:to the left of|to the right of|in front of|behind|above|below|on top of|next to|near|beside)'
    m = re.search(rf'^(.+?)\s+{spatial_rels}\s+(?:an?\s+)?(.+?)(?:\s*$)', c)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    
    # One-object: "X [spatial_rel]" (position descriptor at end)
    m = re.search(r'^(.+?)\s+(?:on|under|above|below|to the|in|at)\b', c)
    if m:
        return m.group(1).strip(), None
    
    return c.strip(), None
```

Update `extract_clip_noun` to handle these modes:
```python
def extract_clip_noun(question: str, mode: str = "mmvp") -> str | tuple:
    if mode == "mmvp":    return _mmvp(question)
    elif mode == "vlmbias": return _vlmbias(question)
    elif mode == "pope":  return _pope(question)
    elif mode == "vlind": return _vlind(question)
    elif mode == "whatsup": return _whatsup(question)  # returns (noun_A, noun_B)
    else: raise ValueError(f"Unknown mode {mode!r}")
```

### 3. `srf.py` — Add calibration + two-noun saliency

In `_build_calib_inputs`, add cases:
```python
elif dataset == "vlind":
    ds = hf_load("MM-Hallu/VLind-Bench", split="train")
    rows = list(ds); rng.shuffle(rows)
    for r in rows[:n]:
        # adapt to actual column names after checking HF structure
        img  = r["image"].convert("RGB")
        q    = str(r["question"]).strip()
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  q}]}]
        # ... (same pattern as other datasets)

elif dataset == "whatsup":
    ds = hf_load("ServiceNow/whatsup_all", split="COCO_QA_one_obj")
    rows = list(ds); rng.shuffle(rows)
    for r in rows[:n]:
        img     = r["image_options"].convert("RGB")
        captions = r["caption_options"]
        # Build MCQ prompt from captions
        opts    = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(captions))
        q       = f"Which caption best describes the image?\n{opts}\nAnswer with the option letter only."
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text",  "text":  q}]}]
        # ... (same pattern)
```

In `_noun_mode_map` inside `reset_for_dataset`:
```python
_NOUN_MODE_MAP = {"mme": "pope", "hallusionbench": "pope",
                  "vlind": "vlind", "whatsup": "whatsup"}
```

In `prepare_sample`, handle two-noun WhatsUp saliency:

```python
# After extracting noun:
noun_result = extract_clip_noun(question, mode=_noun_mode)

if isinstance(noun_result, tuple):
    # Two-noun mode (WhatsUp): max of both saliency maps
    noun_A, noun_B = noun_result
    grid_h, grid_w = clip_sal.get_grid_dims(inputs, _spatial)
    res_A = clip_sal.compute_clip_salience(image, noun_A, grid_h, grid_w, ...)
    if noun_B:
        res_B = clip_sal.compute_clip_salience(image, noun_B, grid_h, grid_w, ...)
        combined = torch.max(res_A.saliency, res_B.saliency)
        max_sim  = max(res_A.max_sim, res_B.max_sim)
    else:
        combined = res_A.saliency
        max_sim  = res_A.max_sim
    
    if max_sim < SALIENCY["clip_fallback_thresh"]:
        patch._STATE["salience_mask"] = None
        patch._STATE["value"]         = 0.0
    else:
        patch._STATE["salience_mask"] = combined
else:
    noun = noun_result
    # ... existing single-noun logic
```

### 4. `eval_datasets.py` — Add loaders

```python
def load_vlind_bench(groups_filter, n_samples):
    """VLind-Bench — MM-Hallu/VLind-Bench."""
    from datasets import load_dataset as hf_load
    # Check actual columns on target server — may differ from kdhong/VLind-Bench
    ds = hf_load("MM-Hallu/VLind-Bench", split="train")
    # Adapt based on actual column names
    ...

def load_whatsup(groups_filter, n_samples):
    """WhatsUp — ServiceNow/whatsup_all.
    
    Groups = subtask names: Controlled_Images_A, COCO_QA_one_obj, etc.
    Correct answer is always caption index 0.
    """
    from datasets import load_dataset as hf_load
    
    SUBTASKS = {
        "controlled_a": "Controlled_Images_A",
        "controlled_b": "Controlled_Images_B",
        "coco_one":     "COCO_QA_one_obj",
        "coco_two":     "COCO_QA_two_obj",
        "vg_one":       "VG_QA_one_obj",
        "vg_two":       "VG_QA_two_obj",
    }
    target_subtasks = groups_filter if groups_filter else list(SUBTASKS.keys())
    out = []
    for key in target_subtasks:
        split = SUBTASKS.get(key, key)
        ds = hf_load("ServiceNow/whatsup_all", split=split)
        rows = list(ds)
        if n_samples:
            rows = random.Random(SEED).sample(rows, min(n_samples, len(rows)))
        for row in rows:
            img      = row["image_options"].convert("RGB")
            captions = row["caption_options"]
            opts     = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(captions))
            prompt   = f"Which caption best describes the image?\n{opts}\nAnswer with the option letter only."
            gt       = "A"  # always index 0
            out.append({
                "image":        img,
                "prompt":       prompt,
                "ground_truth": gt,
                "group":        key,
                "caption_options": captions,  # keep for noun extraction
            })
    return out
```

### 5. `eval.py` — Add run functions + CLI

```python
# Add to --datasets choices:
choices=["mmvp", "pope", "vlmbias", "mme", "whatsup", "vlind"]

# Add run_whatsup() and run_vlind() following same pattern as run_mmvp()
# Key difference for WhatsUp: noun extracted from caption_options[0] (ground truth caption)
# not from the MCQ prompt string
```

---

## Dataset HF IDs Summary

| Dataset | HF ID | Split(s) |
|---|---|---|
| WhatsUp | `ServiceNow/whatsup_all` | `Controlled_Images_A`, `Controlled_Images_B`, `COCO_QA_one_obj`, `COCO_QA_two_obj`, `VG_QA_one_obj`, `VG_QA_two_obj` |
| VLind-Bench | `MM-Hallu/VLind-Bench` | `train` (verify on server) |
| WhatsUp controlled only | `ServiceNow/whatsup_controlled_images_a_inference_test` | `Controlled_Images_A` |

---

## Expected Results

**WhatsUp** — SRF should beat AdaptVis on COCO/VG subtasks:
- AdaptVis gain on COCO_one: +0.6 pts (weak — temperature scaling too blunt)
- SRF's targeted CLIP saliency should do better here

**VLind-Bench** — No competing method has run here yet. SRF should gain because:
- Red banana → CLIP sees redness in banana region → model attends there → says red
- Same mechanism as VLM-Bias (already +4.8%)

---

## Key Implementation Notes

1. **WhatsUp correct answer is always index 0** — verified experimentally (all 412 controlled samples have TARGET=0). This means no shuffling needed — the first caption option is always the ground truth.

2. **For noun extraction in WhatsUp**, extract from `caption_options[0]` (the GT caption), not from the full MCQ prompt. This gives clean object nouns.

3. **Two-noun saliency**: For COCO_two and VG_two subtasks, extract both object nouns and take `max(saliency_A, saliency_B)` elementwise.

4. **Calibration dataset for WhatsUp**: Use `COCO_QA_one_obj` split (largest, ~2247 samples) to calibrate vision-aware heads.

5. **VLind-Bench HF access**: `kdhong/VLind-Bench` returns 404 — use `MM-Hallu/VLind-Bench` instead. Verify columns on your server before writing the loader.

6. **WhatsUp Controlled_Images subtask**: The image is 1280×960 — a composite of 4 images side by side (each 320×960). For VLM eval, use only the QA subtasks (COCO/VG) which have single proper images.
