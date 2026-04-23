#!/usr/bin/env python3
"""
Qwen2.5-VL-3B inference-time attention intervention evaluation.

Loads the model once, runs sanity checks, then evaluates every
(dataset, method, value) combination and saves:
  <output_dir>/results.json     — per-sample records
  <output_dir>/summary.csv      — accuracy per (dataset, group, method, value)
  <output_dir>/plots/           — accuracy-vs-value line plots

Usage
-----
# Baseline on all datasets (5 samples per category)
python run_qwen_eval.py --dataset all --method baseline --n_samples 5

# Temperature sweep on POPE
python run_qwen_eval.py --dataset pope --method temperature \\
    --sweep 0.5 0.75 1.0 1.5 2.0 --n_samples 10

# Vision-boost sweep on MMBench, specific categories
python run_qwen_eval.py --dataset mmbench --method vision_boost \\
    --sweep 1.0 2.0 4.0 8.0 --groups spatial_relationship ocr --n_samples 5

# Save per-layer attention .npy maps alongside results
python run_qwen_eval.py --dataset vlm_bias --method vision_boost \\
    --sweep 1.0 2.0 --save_attn --n_samples 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# patches live in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_attn_patch  as patch
import qwen_decode_patch as decode_patch
import clip_salience
import hssa_salience

os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

MODEL       = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"
SEED        = 42


# ---------------------------------------------------------------------------
# Dataset loaders — each returns List[Dict] with keys:
#   image, prompt, ground_truth, group
# ---------------------------------------------------------------------------

def load_vlm_bias(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    from datasets import load_dataset as hf_load

    print("  Loading VLM Bias (anvo25/vlms-are-biased)…")
    ds   = hf_load("anvo25/vlms-are-biased", split="main")
    seen: Dict[str, int] = {}
    out  = []
    for row in ds:
        group = str(row.get("topic", "unknown")).strip().replace(" ", "_")
        if groups_filter and group not in groups_filter:
            continue
        seen.setdefault(group, 0)
        if seen[group] >= n_samples:
            continue
        seen[group] += 1
        out.append({
            "image":        row["image"].convert("RGB"),
            "prompt":       str(row["prompt"]).strip(),
            "ground_truth": str(row["ground_truth"]).strip(),
            "group":        group,
        })
    print(f"  → {len(out)} samples, {len(seen)} groups")
    return out


def load_pope(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    from datasets import load_dataset as hf_load

    print("  Loading POPE (lmms-lab/POPE)…")
    ds      = hf_load("lmms-lab/POPE", split="test")
    targets = {g.lower() for g in groups_filter} if groups_filter \
              else {"adversarial", "popular", "random"}

    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for row in ds:
        cat = str(row.get("category", "unknown")).strip().lower()
        if cat not in targets:
            continue
        gt     = "Yes" if str(row.get("answer", "")).strip().lower() == "yes" else "No"
        prompt = str(row.get("question", "")).strip() + "\nAnswer with Yes or No only."
        by_group[cat].append({
            "image":        row["image"].convert("RGB"),
            "prompt":       prompt,
            "ground_truth": gt,
            "group":        cat,
        })

    rng = random.Random(SEED)
    out = []
    for cat, items in sorted(by_group.items()):
        out.extend(rng.sample(items, min(n_samples, len(items))))
    print(f"  → {len(out)} samples, {len(by_group)} groups")
    return out


def load_mmbench(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    from datasets import load_dataset as hf_load

    print("  Loading MMBench (lmms-lab/MMBench en dev)…")
    ds       = hf_load("lmms-lab/MMBench", "en", split="dev")
    by_group: Dict[str, List[Dict]] = defaultdict(list)

    for row in ds:
        cat = str(row.get("category", "unknown")).strip()
        if groups_filter and cat not in groups_filter:
            continue
        opts = {
            lbl: str(row.get(lbl, "") or "").strip()
            for lbl in ["A", "B", "C", "D"]
            if str(row.get(lbl, "") or "").strip().lower() not in ("", "nan")
        }
        if not opts:
            continue
        gt       = str(row.get("answer", "")).strip().upper()
        question = str(row.get("question", "")).strip()
        hint     = str(row.get("hint", "") or "").strip()
        if hint and hint.lower() != "nan":
            question = f"{hint}\n{question}"
        opt_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
        prompt   = (f"{question}\n{opt_text}\n"
                    "Answer with the option's letter from the given choices directly.")
        by_group[cat].append({
            "image":        row["image"].convert("RGB"),
            "prompt":       prompt,
            "ground_truth": gt,
            "group":        cat,
        })

    rng = random.Random(SEED)
    out = []
    for cat, items in sorted(by_group.items()):
        out.extend(rng.sample(items, min(n_samples, len(items))))
    print(f"  → {len(out)} samples, {len(by_group)} groups")
    return out


def load_cv_bench(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    from datasets import load_dataset as hf_load

    print("  Loading CV-Bench Relation (nyu-visionx/CV-Bench)…")
    ds  = hf_load("nyu-visionx/CV-Bench", split="test")
    rel = [r for r in ds if r["task"] == "Relation"]

    # Group by spatial direction type
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for row in rel:
        group = "_".join(sorted(row["choices"]))   # "above_below" or "left_right"
        if groups_filter and group not in groups_filter:
            continue
        # Answer: "(A)" → "A"
        gt      = row["answer"].strip().strip("()")
        choices = row["choices"]   # e.g. ['above', 'below']
        opt_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt  = (f"{row['question']}\n{opt_text}\n"
                   "Answer with the option's letter from the given choices directly.")
        by_group[group].append({
            "image":        row["image"].convert("RGB"),
            "prompt":       prompt,
            "ground_truth": gt,
            "group":        group,
        })

    rng = random.Random(SEED)
    out = []
    for grp, items in sorted(by_group.items()):
        out.extend(rng.sample(items, min(n_samples, len(items))))
    print(f"  → {len(out)} samples, {len(by_group)} groups")
    return out


def load_mmvp(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    """MMVP: 300 image pairs, each with an MCQ question testing fine-grained visual patterns.
    Images are indexed 1-300; Questions.csv has matching Index column.
    Groups = pair index (1-150), since each pair shares a question topic.
    We cap at n_samples pairs (2 samples each) unless groups_filter specifies pair IDs.
    """
    import pandas as pd
    from datasets import load_dataset as hf_load

    QUESTIONS_CSV = "/volumes2/hugging_face_cache/mmvp_questions/Questions.csv"
    if not os.path.exists(QUESTIONS_CSV):
        from huggingface_hub import hf_hub_download
        QUESTIONS_CSV = hf_hub_download(
            "MMVP/MMVP", "Questions.csv", repo_type="dataset",
            local_dir="/volumes2/hugging_face_cache/mmvp_questions",
        )

    print("  Loading MMVP (MMVP/MMVP)…")
    df  = pd.read_csv(QUESTIONS_CSV)   # columns: Index, Question, Options, Correct Answer
    img_ds = hf_load("MMVP/MMVP", split="train")   # 300 PIL images

    # HuggingFace imagefolder loads files in lexicographic order (1, 10, 100, 101, ... 2, 20 ...)
    # but the CSV uses numeric order (1, 2, 3, ..., 300).
    # Build mapping: CSV 1-indexed value → HF dataset index.
    lex_sorted = sorted(range(1, 301), key=str)          # lexicographic sort of 1..300
    csv_to_hf  = {csv_1idx: hf_idx for hf_idx, csv_1idx in enumerate(lex_sorted)}

    rng  = random.Random(SEED)
    # Pair IDs: 1..150 (pair i = CSV rows 2i-1 and 2i, 1-indexed)
    pair_ids = list(range(1, 151))
    if groups_filter:
        pair_ids = [int(g) for g in groups_filter if g.isdigit()]
    rng.shuffle(pair_ids)
    selected_pairs = pair_ids[:n_samples]   # n_samples pairs = up to 2*n_samples images

    out = []
    for pair_id in selected_pairs:
        for offset in (0, 1):   # two images per pair
            csv_1idx = (pair_id - 1) * 2 + offset + 1   # 1-indexed CSV Index
            row_idx  = csv_1idx - 1                      # 0-based df.iloc row
            if row_idx >= len(df):
                continue
            row     = df.iloc[row_idx]
            img_idx = csv_to_hf[csv_1idx]               # correct HF dataset index
            # Parse options: "(a) Open (b) Closed" → choices list
            opts_raw = str(row["Options"])
            # Extract option letters and text
            import re
            opt_matches = re.findall(r'\(([ab])\)\s*([^(]+)', opts_raw, re.IGNORECASE)
            if not opt_matches:
                continue
            # Map (a)→A, (b)→B for standard MCQ format the model was trained on
            choices  = {m[0].upper(): m[1].strip() for m in opt_matches}
            opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
            gt_raw   = str(row["Correct Answer"]).strip().strip("()").upper()   # "A" or "B"

            prompt = (f"{row['Question']}\n{opt_text}\n"
                      "Answer with the option letter only.")

            out.append({
                "image":        img_ds[img_idx]["image"].convert("RGB"),  # img_idx = csv_to_hf[csv_1idx]
                "prompt":       prompt,
                "ground_truth": gt_raw,   # "A" or "B"
                "group":        f"pair_{pair_id:03d}",
                "dataset":      "mmvp",
            })

    print(f"  → {len(out)} samples, {len(selected_pairs)} pairs")
    return out


def load_hallusionbench(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    """HallusionBench: visual hallucination benchmark with Yes/No questions.
    Uses the 'image' split (951 samples with images).
    Categories: VS (Visual Supplement = prior-vision conflict), FS, VD, etc.
    gt_answer: '1' = Yes, '0' = No.
    Groups = subcategory (chart, figure, illusion, ocr, etc.).
    """
    from datasets import load_dataset as hf_load

    print("  Loading HallusionBench (lmms-lab/HallusionBench)…")
    ds = hf_load("lmms-lab/HallusionBench", split="image")

    # Default: use VS category (Visual Supplement = prior-vision conflict, best fit)
    # groups_filter can override with specific subcategories
    target_cats  = {"VS"}   # expand later if needed
    by_group: Dict[str, List[Dict]] = defaultdict(list)

    for row in ds:
        cat    = str(row.get("category", "")).strip()
        subcat = str(row.get("subcategory", "unknown")).strip()
        if cat not in target_cats:
            continue
        gt_raw = str(row.get("gt_answer", "0")).strip()
        gt     = "Yes" if gt_raw == "1" else "No"
        prompt = str(row.get("question", "")).strip() + "\nAnswer with Yes or No only."
        img    = row.get("image")
        if img is None:
            continue

        group = f"{cat}_{subcat}"
        if groups_filter and group not in groups_filter and subcat not in groups_filter:
            continue

        by_group[group].append({
            "image":        img.convert("RGB"),
            "prompt":       prompt,
            "ground_truth": gt,
            "group":        group,
            "dataset":      "hallusionbench",
        })

    rng = random.Random(SEED)
    out = []
    for grp, items in sorted(by_group.items()):
        out.extend(rng.sample(items, min(n_samples, len(items))))
    print(f"  → {len(out)} samples, {len(by_group)} groups")
    return out


LOADERS: Dict[str, Any] = {
    "vlm_bias":        load_vlm_bias,
    "pope":            load_pope,
    "mmbench":         load_mmbench,
    "cv_bench":        load_cv_bench,
    "mmvp":            load_mmvp,
    "hallusionbench":  load_hallusionbench,
}


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def is_correct(pred: str, gt: str) -> bool:
    """Flexible answer matching:
    - Strips curly braces: '{Yes}' → 'Yes'  (VLM Bias)
    - Prefix match: 'A. explanation' → 'A'  (MMBench, MMVP)
    - Case-insensitive
    """
    pred_clean = pred.strip().strip("{}").strip().lower()
    gt_clean   = gt.strip().lower()
    return pred_clean.startswith(gt_clean)


# ---------------------------------------------------------------------------
# Optional attention-map saving (mirrors run_qwen_attn_adaptvis.py)
# ---------------------------------------------------------------------------

def _save_attn_npy(
    model: Any,
    inputs: Any,
    processor: Any,
    save_dir: str,
    spatial_merge_size: int,
    first_gen_token_id: int,
) -> None:
    """Capture per-layer attention at the answer token and save as .npy log-probs."""
    image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    image_mask     = inputs["input_ids"][0].cpu() == image_token_id
    positions      = image_mask.nonzero(as_tuple=True)[0]
    img_start      = int(positions[0].item())
    img_end        = int(positions[-1].item())

    thw    = inputs["image_grid_thw"][0]
    grid_h = int(thw[1].item()) // spatial_merge_size
    grid_w = int(thw[2].item()) // spatial_merge_size

    extra = torch.tensor([[first_gen_token_id]],
                         dtype=inputs["input_ids"].dtype,
                         device=inputs["input_ids"].device)
    ext                     = dict(inputs)
    ext["input_ids"]        = torch.cat([inputs["input_ids"], extra], dim=1)
    ext["attention_mask"]   = torch.cat(
        [inputs["attention_mask"],
         torch.ones((1, 1), dtype=inputs["attention_mask"].dtype,
                    device=inputs["attention_mask"].device)],
        dim=1,
    )

    captured: List[Any] = []

    def _hook(module: Any, _inp: Any, output: Any) -> Any:
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            row = output[1][0, :, -1, :]           # (n_heads, seq_len)
            captured.append(row.detach().cpu().float())
            return (output[0], None) + output[2:]
        return output

    layers = model.language_model.layers
    hooks  = [lay.self_attn.register_forward_hook(_hook) for lay in layers]
    try:
        with torch.inference_mode():
            model(**ext, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    os.makedirs(save_dir, exist_ok=True)
    for layer_idx, attn_row in enumerate(captured):
        arr     = attn_row.unsqueeze(0).numpy()     # (1, n_heads, seq_len)
        log_arr = np.log(arr + 1e-10)
        np.save(
            os.path.join(save_dir,
                         f"diff_{layer_idx}_start{img_start}_end{img_end}.npy"),
            log_arr,
        )

    grid_info = {"grid_h": grid_h, "grid_w": grid_w,
                 "img_start": img_start, "img_end": img_end,
                 "n_layers": len(captured)}
    with open(os.path.join(save_dir, "grid_info.json"), "w") as f:
        json.dump(grid_info, f, indent=2)


# ---------------------------------------------------------------------------
# Per-value evaluation pass
# ---------------------------------------------------------------------------

ATTN_METHODS     = ("baseline", "temperature", "vision_boost", "vhr_boost", "vaf",
                    "attn_salience", "clip_salience", "srf_clip", "srf_hssa")
CONTRAST_METHODS = ("icd", "vcd")
_GEN_KWARGS = dict(max_new_tokens=16, do_sample=False,
                   temperature=None, top_p=None, num_beams=1)


def evaluate(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    samples: List[Dict],
    method: str,
    value: float,
    spatial_merge_size: int,
    save_attn: bool,
    attn_dir: str,
) -> List[Dict]:
    """
    Run `samples` under (method, value), return list of per-sample result dicts.

    Attention methods (baseline/temperature/vision_boost/vhr_boost):
        Uses qwen_attn_patch — modifies softmax logits inside the decoder.

    Contrastive methods (icd/vcd):
        Uses qwen_decode_patch — subtracts contrast logits at the first token.
        The attention patch is left unpatched for these methods.
    """
    if method in ATTN_METHODS:
        patch.patch_model(model, method, value)
    else:
        patch.unpatch_model()   # contrastive methods don't touch attention

    records: List[Dict] = []

    for i, sample in enumerate(samples):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text",  "text":  sample["prompt"]},
        ]}]
        text       = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, _  = process_vision_info(messages)
        inputs     = processor(text=[text], images=img_in,
                               padding=True, return_tensors="pt")
        inputs     = inputs.to(next(model.parameters()).device)

        # --- generate ---
        if method in ATTN_METHODS:
            img_start, img_end = patch.get_image_token_range(inputs, processor)

            # Salience methods: compute per-token mask before the intervention pass
            sal_top_k = patch._STATE.get("_sal_top_k", 0.3)
            if method == "attn_salience":
                sal = patch.compute_attention_salience(
                    model, inputs, img_start, img_end,
                    top_k_pct=sal_top_k, normalize=True,
                )
                patch._STATE["salience_mask"] = sal
            elif method == "clip_salience":
                grid_h, grid_w = clip_salience.get_grid_dims(inputs, spatial_merge_size)
                clip_result = clip_salience.compute_clip_salience(
                    sample["image"], sample["prompt"], grid_h, grid_w,
                    top_k_pct=sal_top_k,
                )
                # If object is absent (low CLIP similarity), clip_result.mask is
                # uniform (ones) so we fall back to standard visboost_heads behaviour.
                patch._STATE["salience_mask"] = clip_result.mask
            elif method == "srf_clip":
                # SRF-CLIP: use soft continuous saliency (not binary mask).
                # Reset then flip to "srf" so generate uses the SRF intervention.
                patch._STATE["method"] = "srf_clip"
                grid_h, grid_w = clip_salience.get_grid_dims(inputs, spatial_merge_size)
                clip_result = clip_salience.compute_clip_salience(
                    sample["image"], sample["prompt"], grid_h, grid_w,
                    top_k_pct=sal_top_k,
                )
                patch._STATE["salience_mask"] = clip_result.saliency   # soft [0,1]
                patch._STATE["method"] = "srf"
            elif method == "srf_hssa":
                # SRF-HSSA: one forward pass for HSSA saliency, then generate with srf.
                # Reset method to "srf_hssa" (no-op in _patched_softmax) before the
                # HSSA forward pass to avoid using stale img_start/img_end from the
                # previous sample. Flip to "srf" after saliency is ready.
                patch._STATE["method"] = "srf_hssa"
                hssa_layer = patch._STATE.get("_hssa_layer", 16)
                hssa_result = hssa_salience.compute_hssa_salience(
                    model, inputs, img_start, img_end,
                    layer_idx=hssa_layer, top_k_pct=sal_top_k,
                )
                patch._STATE["salience_mask"] = hssa_result.saliency   # soft [0,1]
                patch._STATE["method"] = "srf"
            else:
                patch._STATE["salience_mask"] = None

            patch.update_sample(img_start, img_end)
            with torch.inference_mode():
                out = model.generate(**inputs, **_GEN_KWARGS)

        elif method == "icd":
            out = decode_patch.generate_with_icd(
                model, inputs, sample, processor, process_vision_info,
                alpha=value, **_GEN_KWARGS,
            )

        elif method == "vcd":
            out = decode_patch.generate_with_vcd(
                model, inputs, alpha=value, **_GEN_KWARGS,
            )

        pred    = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        correct = is_correct(pred, sample["ground_truth"])

        if save_attn and method in ATTN_METHODS:
            first_token_id  = int(out[0, inputs["input_ids"].shape[1]].item())
            sample_attn_dir = os.path.join(attn_dir, f"{sample['group']}_{i:03d}")
            _save_attn_npy(model, inputs, processor, sample_attn_dir,
                           spatial_merge_size, first_token_id)
            with open(os.path.join(sample_attn_dir, "sample_info.json"), "w") as f:
                json.dump({"Prompt": sample["prompt"], "Generation": pred,
                           "Golden": sample["ground_truth"]}, f, indent=2)
        elif save_attn and method in CONTRAST_METHODS:
            print(f"  Note: --save_attn not supported for {method}, skipping attn maps.")

        torch.cuda.empty_cache()

        records.append({
            "group":        sample["group"],
            "prompt":       sample["prompt"][:120],
            "ground_truth": sample["ground_truth"],
            "prediction":   pred,
            "correct":      correct,
            "method":       method,
            "value":        value,
        })

        status = "✓" if correct else "✗"
        print(f"    [{i + 1:03d}/{len(samples):03d}] {sample['group']:30s} "
              f"GT={sample['ground_truth']:6s}  Pred={pred[:14]:14s}  {status}")

    return records


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def append_results(records: List[Dict], path: str) -> None:
    """Append records to a JSON array file (creates if missing)."""
    existing: List[Dict] = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    with open(path, "w") as f:
        json.dump(existing + records, f, indent=2)


def append_summary_csv(records: List[Dict], path: str) -> None:
    """Aggregate records by (dataset, group, method, value) and append to CSV."""
    key_records: Dict[Tuple, List[bool]] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["group"], r["method"], r["value"])
        key_records[key].append(r["correct"])

    rows = [
        {
            "dataset":   ds,
            "group":     grp,
            "method":    meth,
            "value":     val,
            "accuracy":  round(sum(corrects) / len(corrects), 4),
            "n_samples": len(corrects),
        }
        for (ds, grp, meth, val), corrects in sorted(key_records.items())
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "group", "method", "value", "accuracy", "n_samples"]
        )
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar(all_records: List[Dict], output_dir: str) -> None:
    """
    Bar chart of per-group accuracy for each (dataset, method, value) combo
    that has only a single value (e.g. baseline).  One PNG per dataset.
    """
    by_ds_method: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in all_records:
        by_ds_method[(r["dataset"], r["method"])].append(r)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (ds, method), records in by_ds_method.items():
        values = sorted({r["value"] for r in records})
        if len(values) != 1:
            continue    # sweep results → handled by plot_accuracy

        val    = values[0]
        groups = sorted({r["group"] for r in records})
        accs   = []
        for grp in groups:
            subset = [r for r in records if r["group"] == grp]
            accs.append(sum(r["correct"] for r in subset) / len(subset))
        overall = sum(r["correct"] for r in records) / len(records)

        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.7), 5))
        colors  = plt.cm.tab20.colors  # type: ignore[attr-defined]
        bars    = ax.bar(groups, accs,
                         color=[colors[i % len(colors)] for i in range(len(groups))],
                         edgecolor="white", linewidth=0.5)
        ax.axhline(overall, color="black", linewidth=1.8, linestyle="--",
                   label=f"overall={overall:.2f}")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds}  —  {method}  (value={val})")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=45)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        save_path = os.path.join(plot_dir, f"{ds}_{method}_bar.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {save_path}")


def plot_accuracy(all_records: List[Dict], output_dir: str) -> None:
    """
    For each (dataset, method) with ≥2 distinct values, plot accuracy vs. value
    with one line per group and a thick "overall" line.
    """
    by_ds_method: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in all_records:
        by_ds_method[(r["dataset"], r["method"])].append(r)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for (ds, method), records in by_ds_method.items():
        values = sorted({r["value"] for r in records})
        if len(values) < 2:
            continue                # single value → bar chart instead

        groups = sorted({r["group"] for r in records})
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.tab20.colors  # type: ignore[attr-defined]

        # Per-group lines (light)
        for gi, grp in enumerate(groups):
            grp_recs = [r for r in records if r["group"] == grp]
            xs, ys   = [], []
            for v in values:
                subset = [r for r in grp_recs if r["value"] == v]
                if subset:
                    xs.append(v)
                    ys.append(sum(r["correct"] for r in subset) / len(subset))
            ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=4,
                    color=colors[gi % len(colors)], alpha=0.6, label=grp)

        # Overall line (thick + black)
        xs, ys = [], []
        for v in values:
            subset = [r for r in records if r["value"] == v]
            if subset:
                xs.append(v)
                ys.append(sum(r["correct"] for r in subset) / len(subset))
        ax.plot(xs, ys, marker="D", linewidth=2.5, markersize=7,
                color="black", label="overall", zorder=5)

        ax.set_xlabel("Intervention value")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds}  —  {method}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(plot_dir, f"{ds}_{method}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    # Resolve datasets and sweep values
    datasets = list(LOADERS.keys()) if args.dataset == "all" else [args.dataset]

    if args.method == "baseline":
        if args.sweep and args.sweep != [1.0]:
            print("Note: --sweep ignored for baseline (always value=1.0)")
        values = [1.0]
    elif args.method in CONTRAST_METHODS and not args.sweep and args.value == 1.0:
        # Default sweep for contrastive methods if user didn't specify
        values = [0.5, 1.0, 1.5, 2.0]
        print(f"Note: no --sweep given for {args.method}, using default {values}")
    else:
        values = args.sweep if args.sweep else [args.value]

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    summary_path = os.path.join(args.output_dir, "summary.csv")

    # -----------------------------------------------------------------------
    # --plot_only: regenerate plots from existing results.json, then exit
    # -----------------------------------------------------------------------
    if args.plot_only:
        if not os.path.exists(results_path):
            print(f"ERROR: {results_path} not found. Run without --plot_only first.")
            sys.exit(1)
        with open(results_path) as f:
            all_records = json.load(f)
        plot_bar(all_records, args.output_dir)
        plot_accuracy(all_records, args.output_dir)
        print("Plots regenerated.")
        return

    # -----------------------------------------------------------------------
    # Load model once
    # -----------------------------------------------------------------------
    print(f"\nLoading {MODEL} (eager attention)…")
    # bfloat16 requires Ampere+ (RTX 30xx). On older GPUs (e.g. GTX 1080 Ti / Pascal)
    # use float16 instead: TORCH_DTYPE=float16 python run_qwen_eval.py ...
    _dtype_str = os.environ.get("TORCH_DTYPE", "bfloat16")
    _dtype = torch.float16 if _dtype_str == "float16" else torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="auto", torch_dtype=_dtype,
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=400 * 28 * 28)

    vision_cfg         = getattr(model.config, "vision_config", None)
    spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
    n_layers           = len(model.language_model.layers)
    print(f"  spatial_merge_size={spatial_merge_size}, decoder layers={n_layers}")

    # -----------------------------------------------------------------------
    # Sanity checks — run once on a single sample from the first dataset
    # -----------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    n_calib  = max(1, min(20, args.n_samples))   # up to 20 samples for VHR calibration
    pre_samples = LOADERS[datasets[0]](args.groups, n_samples=n_calib)
    if not pre_samples:
        print("ERROR: no samples for sanity check — check --groups.")
        sys.exit(1)

    # Pre-process all calibration samples (needed for VHR)
    calib_inputs_list: List[Any] = []
    calib_img_ranges: List[Any]  = []
    device = next(model.parameters()).device

    for s in pre_samples:
        sc_msgs = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text",  "text":  s["prompt"]},
        ]}]
        sc_text   = processor.apply_chat_template(sc_msgs, tokenize=False,
                                                  add_generation_prompt=True)
        sc_img, _ = process_vision_info(sc_msgs)
        sc_inp    = processor(text=[sc_text], images=sc_img,
                              padding=True, return_tensors="pt")
        sc_inp    = {k: v.to(device) for k, v in sc_inp.items()}
        calib_inputs_list.append(sc_inp)
        calib_img_ranges.append(patch.get_image_token_range(sc_inp, processor))

    sc_inputs = calib_inputs_list[0]   # single sample for non-VHR checks
    sc_sample = pre_samples[0]

    if args.method in ATTN_METHODS:
        # For attn-level methods: run standard CHECK 1-4
        # Use temperature=1.5 as the non-trivial method for baseline runs
        if args.method == "baseline":
            sc_method, sc_value = "temperature", 1.5
        elif args.method in ("vhr_boost", "vaf", "attn_salience", "clip_salience",
                             "srf_clip", "srf_hssa"):
            # Standard checks use vision_boost as proxy (simpler, same principle)
            sc_method = "vision_boost"
            sc_value  = 2.0
        else:
            sc_method = args.method
            sc_value  = next((v for v in values if v != 1.0), None)
            if sc_value is None:
                sc_value = 1.5

        patch.run_sanity_checks(model, sc_inputs, processor, sc_method, sc_value)
        patch.unpatch_model()

        # VHR / VAF / salience methods: identify vision-aware heads
        if args.method in ("vhr_boost", "vaf", "attn_salience", "clip_salience",
                           "srf_clip", "srf_hssa"):
            torch.cuda.empty_cache()   # reclaim fragmented memory before calibration
            print(f"\n--- VHR head identification (used by {args.method}) ---")
            patch.run_vhr_sanity_checks(
                model,
                calib_inputs_list,
                calib_img_ranges,
                sc_inputs,
                processor,
                top_k_pct=args.vhr_top_k,
            )
            # head_mask is now stored in patch._STATE["head_mask"]
            n_heads    = len(patch._STATE["head_mask"])
            n_selected = int(patch._STATE["head_mask"].sum().item())
            print(f"  Using {n_selected}/{n_heads} vision-aware heads "
                  f"(top {args.vhr_top_k*100:.0f}%)")

        # VAF / SRF: propagate layer range + beta into _STATE
        if args.method in ("vaf", "srf_clip", "srf_hssa"):
            patch._STATE["vaf_beta"]        = args.vaf_beta
            patch._STATE["vaf_layer_start"] = args.vaf_layer_start
            patch._STATE["vaf_layer_end"]   = args.vaf_layer_end
            if args.method == "vaf":
                print(f"  VAF: alpha=value (sweep), beta={args.vaf_beta}, "
                      f"layers={args.vaf_layer_start}-{args.vaf_layer_end}")

        # Salience methods: store top_k in _STATE for per-sample computation
        if args.method in ("attn_salience", "clip_salience", "srf_clip", "srf_hssa"):
            patch._STATE["_sal_top_k"] = args.sal_top_k
            print(f"  {args.method}: sal_top_k={args.sal_top_k:.0%}, "
                  f"using {'vision-aware heads' if patch._STATE['head_mask'] is not None else 'all heads'}")

        # SRF: propagate SRF-specific params
        if args.method in ("srf_clip", "srf_hssa"):
            patch._STATE["srf_background_eps"] = args.srf_background_eps
            patch._STATE["_hssa_layer"]        = args.hssa_layer
            print(f"  SRF: boost=log(value), background_eps={args.srf_background_eps}, "
                  f"beta={args.vaf_beta}, layers={args.vaf_layer_start}-{args.vaf_layer_end}"
                  + (f", hssa_layer={args.hssa_layer}" if args.method == "srf_hssa" else ""))

        # Free calibration GPU tensors — calib_inputs_list holds pixel_values
        # for n_calib samples on GPU. With n_calib=20 this can consume ~1-2 GB
        # and cause OOM in the vision encoder during evaluation.
        del calib_inputs_list, calib_img_ranges
        torch.cuda.empty_cache()

    elif args.method in CONTRAST_METHODS:
        # For contrastive methods: run attn checks with temperature (to verify patch),
        # then run contrastive-specific checks
        patch.run_sanity_checks(model, sc_inputs, processor, "temperature", 1.5)
        patch.unpatch_model()
        decode_patch.run_decode_sanity_checks(
            model, sc_inputs, sc_sample, processor, process_vision_info, alpha=1.0
        )
        del calib_inputs_list, calib_img_ranges
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Main evaluation loop
    # -----------------------------------------------------------------------
    all_records: List[Dict] = []

    for ds_name in datasets:
        print(f"\n{'='*65}")
        print(f"  Dataset : {ds_name}")
        print(f"{'='*65}")

        samples = LOADERS[ds_name](args.groups, args.n_samples)
        if not samples:
            print(f"  No samples found for {ds_name}. Skipping.")
            continue

        for val in values:
            print(f"\n  method={args.method}  value={val}")
            attn_dir = os.path.join(
                args.output_dir, "attn_maps", ds_name, f"{args.method}_{val}")

            records = evaluate(
                model, processor, process_vision_info,
                samples, args.method, val,
                spatial_merge_size,
                save_attn=args.save_attn,
                attn_dir=attn_dir,
            )

            for r in records:
                r["dataset"] = ds_name

            n_correct = sum(r["correct"] for r in records)
            print(f"\n  → Overall: {n_correct}/{len(records)} "
                  f"= {n_correct / len(records):.3f}")

            all_records.extend(records)
            append_results(records, results_path)
            append_summary_csv(records, summary_path)

    # -----------------------------------------------------------------------
    # Plots and cleanup
    # -----------------------------------------------------------------------
    patch.unpatch_model()
    plot_bar(all_records, args.output_dir)
    plot_accuracy(all_records, args.output_dir)

    print(f"\n{'='*65}")
    print(f"Done.")
    print(f"  results  → {results_path}")
    print(f"  summary  → {summary_path}")
    print(f"  plots    → {os.path.join(args.output_dir, 'plots')}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Qwen2.5-VL attention intervention evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", default="all",
        choices=["all"] + list(LOADERS.keys()),
        help="Dataset(s) to evaluate (default: all). "
             "Options: " + ", ".join(LOADERS.keys()),
    )
    p.add_argument(
        "--method", default="baseline",
        choices=["baseline", "temperature", "vision_boost", "vhr_boost", "vaf",
                 "icd", "vcd", "attn_salience", "clip_salience",
                 "srf_clip", "srf_hssa"],
        help="Intervention method. Attn-level: baseline/temperature/vision_boost/vhr_boost/vaf/"
             "attn_salience/clip_salience/srf_clip/srf_hssa. Logit-level: icd/vcd.",
    )
    p.add_argument(
        "--vhr_top_k", type=float, default=0.50,
        help="Fraction of heads to select as vision-aware for vhr_boost/vaf (default: 0.50).",
    )
    # Salience-method hyperparameters
    p.add_argument(
        "--sal_top_k", type=float, default=0.3,
        help="Fraction of image tokens selected as salient for attn_salience/clip_salience/srf_* (default: 0.3).",
    )
    # SRF-specific hyperparameters
    p.add_argument(
        "--srf_background_eps", type=float, default=0.5,
        help="SRF: suppression amount for non-salient image tokens (default: 0.5).",
    )
    p.add_argument(
        "--hssa_layer", type=int, default=16,
        help="SRF-HSSA: decoder layer index for hidden-state saliency (default: 16).",
    )
    # VAF-specific hyperparameters
    p.add_argument(
        "--vaf_beta", type=float, default=0.1,
        help="VAF suppression coefficient β for system-prompt attention (default: 0.1).",
    )
    p.add_argument(
        "--vaf_layer_start", type=int, default=8,
        help="VAF: first decoder layer to apply (default: 8).",
    )
    p.add_argument(
        "--vaf_layer_end", type=int, default=15,
        help="VAF: last decoder layer to apply inclusive (default: 15).",
    )
    p.add_argument(
        "--value", type=float, default=1.0,
        help="Single intervention value (use --sweep for multiple).",
    )
    p.add_argument(
        "--sweep", type=float, nargs="+", default=None,
        help="Sweep over multiple values, e.g. --sweep 0.5 1.0 1.5 2.0",
    )
    p.add_argument(
        "--n_samples", type=int, default=5,
        help="Samples per group/category (default: 5).",
    )
    p.add_argument(
        "--groups", nargs="*", default=None,
        help="Groups/categories to include (default: all).",
    )
    p.add_argument(
        "--save_attn", action="store_true",
        help="Save per-layer attention .npy maps alongside results (disk-heavy).",
    )
    p.add_argument(
        "--output_dir", default="output/qwen_eval",
        help="Root output directory (default: output/qwen_eval).",
    )
    p.add_argument(
        "--plot_only", action="store_true",
        help="Skip evaluation; regenerate plots from existing results.json.",
    )
    main(p.parse_args())
