"""
Shared dataset loaders for VLM attention-intervention evaluation.

Each loader returns List[Dict] with keys:
    image         : PIL.Image (RGB)
    prompt        : str — question / instruction sent to the model
    ground_truth  : str — expected answer
    group         : str — category / subcategory label (used for per-group accuracy)

Import in eval scripts:
    from eval_datasets import LOADERS, is_correct, SEED
"""
from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

SEED: int = 42


# ---------------------------------------------------------------------------
# Answer-matching helper
# ---------------------------------------------------------------------------

def is_correct(pred: str, gt: str) -> bool:
    """Flexible answer matching.

    - Strips curly braces: ``{Yes}`` → ``Yes``  (VLM Bias)
    - Prefix match: ``A. explanation`` → ``A``   (MMBench, MMVP)
    - Case-insensitive
    """
    pred_clean = pred.strip().strip("{}").strip().lower()
    gt_clean   = gt.strip().lower()
    return pred_clean.startswith(gt_clean)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_vlm_bias(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    """VLM Bias — anvo25/vlms-are-biased.

    Groups = topic labels (e.g. chess_pieces, logos, animals, …).
    n_samples is capped per group.
    """
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
    """POPE — lmms-lab/POPE, adversarial / popular / random splits.

    n_samples is the max per category.  Default = all 3 categories.
    """
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
    """MMBench — lmms-lab/MMBench English dev split.

    Groups = category labels (spatial_relationship, attribute, …).
    """
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
    """CV-Bench spatial-relation subset — nyu-visionx/CV-Bench.

    Groups = sorted choice labels, e.g. "above_below" or "left_right".
    """
    from datasets import load_dataset as hf_load

    print("  Loading CV-Bench Relation (nyu-visionx/CV-Bench)…")
    ds  = hf_load("nyu-visionx/CV-Bench", split="test")
    rel = [r for r in ds if r["task"] == "Relation"]

    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for row in rel:
        group = "_".join(sorted(row["choices"]))
        if groups_filter and group not in groups_filter:
            continue
        gt       = row["answer"].strip().strip("()")
        choices  = row["choices"]
        opt_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt   = (f"{row['question']}\n{opt_text}\n"
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
    """MMVP — MMVP/MMVP.  300 images = 150 pairs; n_samples = number of pairs.

    Groups = "pair_NNN".  groups_filter accepts numeric pair IDs as strings.
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
    df     = pd.read_csv(QUESTIONS_CSV)
    img_ds = hf_load("MMVP/MMVP", split="train")

    # HuggingFace imagefolder loads in lexicographic order (1, 10, 100, …, 2, 20, …)
    # but CSV uses numeric order (1, 2, 3, …).  Build the mapping once.
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {csv_1idx: hf_idx for hf_idx, csv_1idx in enumerate(lex_sorted)}

    rng      = random.Random(SEED)
    pair_ids = list(range(1, 151))
    if groups_filter:
        pair_ids = [int(g) for g in groups_filter if g.isdigit()]
    rng.shuffle(pair_ids)
    selected_pairs = pair_ids[:n_samples]

    out = []
    for pair_id in selected_pairs:
        for offset in (0, 1):
            csv_1idx = (pair_id - 1) * 2 + offset + 1
            row_idx  = csv_1idx - 1
            if row_idx >= len(df):
                continue
            row         = df.iloc[row_idx]
            img_idx     = csv_to_hf[csv_1idx]
            opts_raw    = str(row["Options"])
            opt_matches = re.findall(r"\(([ab])\)\s*([^(]+)", opts_raw, re.IGNORECASE)
            if not opt_matches:
                continue
            choices  = {m[0].upper(): m[1].strip() for m in opt_matches}
            opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
            gt_raw   = str(row["Correct Answer"]).strip().strip("()").upper()
            prompt   = (f"{row['Question']}\n{opt_text}\n"
                        "Answer with the option letter only.")
            out.append({
                "image":        img_ds[img_idx]["image"].convert("RGB"),
                "prompt":       prompt,
                "ground_truth": gt_raw,
                "group":        f"pair_{pair_id:03d}",
            })

    print(f"  → {len(out)} samples, {len(selected_pairs)} pairs")
    return out


def load_hallusionbench(groups_filter: Optional[List[str]], n_samples: int) -> List[Dict]:
    """HallusionBench — lmms-lab/HallusionBench, VS (Visual Supplement) category.

    Groups = "{category}_{subcategory}" (e.g. VS_chart, VS_figure).
    """
    from datasets import load_dataset as hf_load

    print("  Loading HallusionBench (lmms-lab/HallusionBench)…")
    ds           = hf_load("lmms-lab/HallusionBench", split="image")
    target_cats  = {"VS"}
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
        })

    rng = random.Random(SEED)
    out = []
    for grp, items in sorted(by_group.items()):
        out.extend(rng.sample(items, min(n_samples, len(items))))
    print(f"  → {len(out)} samples, {len(by_group)} groups")
    return out


# ---------------------------------------------------------------------------
# Registry — maps dataset name → loader function
# ---------------------------------------------------------------------------

LOADERS: Dict[str, Any] = {
    "vlm_bias":        load_vlm_bias,
    "pope":            load_pope,
    "mmbench":         load_mmbench,
    "cv_bench":        load_cv_bench,
    "mmvp":            load_mmvp,
    "hallusionbench":  load_hallusionbench,
}
