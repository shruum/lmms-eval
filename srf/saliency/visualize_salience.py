#!/usr/bin/env python3
"""
Salience visualizer: CLIP vs HSSA side-by-side on POPE, MMVP, VLM Bias.

Layout per sample (3 columns):
  Col 0: original image + question + GT
  Col 1: CLIP saliency (ViT-L/14, 7×7 grid, soft jet heatmap)
  Col 2: HSSA saliency (Qwen hidden state cosine sim at layer 12, soft jet heatmap)

Datasets:
  pope     — POPE adversarial, object hallucination yes/no
  mmvp     — MMVP visual pattern VQA (image-question mapping fixed)
  vlm_bias — VLM Bias, Chess Pieces + Logos topics only

Usage:
  python visualize_salience.py --datasets pope mmvp vlm_bias --n 5
      --output_dir /volumes2/mllm/lmms-eval/results/salience_vis
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import random
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from srf.saliency import clip_salience as clip_sal
from hssa_salience import compute_hssa_salience
import qwen_attn_patch as patch

SEED = 42

# VLM Bias: only Chess Pieces + Logos (no animal images)
VLM_BIAS_TOPICS = {"chess pieces", "logos"}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_pope(n: int) -> List[dict]:
    from datasets import load_dataset
    ds  = load_dataset("lmms-lab/POPE", split="test")
    rng = random.Random(SEED)
    rows = rng.sample([r for r in ds if r.get("category") == "adversarial"], n)
    return [{"image":   r["image"].convert("RGB"),
             "prompt":  r["question"] + "\nAnswer with Yes or No only.",
             "gt":      r["answer"],
             "dataset": "pope",
             "topic":   ""} for r in rows]


def load_mmvp(n: int) -> List[dict]:
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    QUESTIONS_CSV = "/volumes2/hugging_face_cache/mmvp_questions/Questions.csv"
    if not os.path.exists(QUESTIONS_CSV):
        QUESTIONS_CSV = hf_hub_download(
            "MMVP/MMVP", "Questions.csv", repo_type="dataset",
            local_dir="/volumes2/hugging_face_cache/mmvp_questions",
        )
    df     = pd.read_csv(QUESTIONS_CSV)
    img_ds = load_dataset("MMVP/MMVP", split="train")

    # Fix: HuggingFace imagefolder loads in lexicographic order (1, 10, 100, ..., 2, 20 ...)
    # Build mapping: CSV 1-indexed value → HF dataset index
    lex_sorted = sorted(range(1, 301), key=str)
    csv_to_hf  = {csv_idx: hf_idx for hf_idx, csv_idx in enumerate(lex_sorted)}

    rng      = random.Random(SEED)
    pair_ids = list(range(1, 151))
    rng.shuffle(pair_ids)
    selected = pair_ids[:n]

    out = []
    for pair_id in selected:
        for offset in (0, 1):
            csv_1idx = (pair_id - 1) * 2 + offset + 1
            row      = df.iloc[csv_1idx - 1]
            opts_raw = str(row["Options"])
            opt_matches = re.findall(r'\(([ab])\)\s*([^(]+)', opts_raw, re.IGNORECASE)
            if not opt_matches:
                continue
            choices  = {m[0].upper(): m[1].strip() for m in opt_matches}
            opt_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
            gt       = str(row["Correct Answer"]).strip().strip("()").upper()
            prompt   = f"{row['Question']}\n{opt_text}\nAnswer with the option letter only."
            out.append({
                "image":   img_ds[csv_to_hf[csv_1idx]]["image"].convert("RGB"),
                "prompt":  prompt,
                "gt":      gt,
                "dataset": "mmvp",
                "topic":   f"pair_{pair_id:03d}",
            })
    return out


def load_vlm_bias(n: int) -> List[dict]:
    from datasets import load_dataset
    ds  = load_dataset("anvo25/vlms-are-biased", split="main")
    rng = random.Random(SEED)
    # Chess Pieces + Logos only
    filtered = [r for r in ds if str(r.get("topic", "")).lower() in VLM_BIAS_TOPICS]
    rows = rng.sample(filtered, min(n, len(filtered)))
    return [{"image":   r["image"].convert("RGB"),
             "prompt":  r["prompt"],
             "gt":      r["ground_truth"],
             "dataset": "vlm_bias",
             "topic":   str(r["topic"]).lower()} for r in rows]


LOADERS = {
    "pope":     load_pope,
    "mmvp":     load_mmvp,
    "vlm_bias": load_vlm_bias,
}


# ---------------------------------------------------------------------------
# Noun extraction (for CLIP query)
# ---------------------------------------------------------------------------

def extract_noun(prompt: str, dataset: str, topic: str = "") -> str:
    """Return the best single noun for CLIP similarity query."""

    # VLM Bias: use topic directly — more reliable than regex
    if dataset == "vlm_bias":
        return topic if topic else "object"

    if dataset == "pope":
        m = re.search(r"is there an?\s+(\w+)", prompt, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    if dataset == "mmvp":
        # Drop possessives: "butterfly's wings" → "butterfly"
        clean = re.sub(r"'s\s+\w+", "", prompt)
        m = re.search(
            r"(?:the|a|an)\s+([\w]+(?:\s+[\w]+)?)",
            clean, re.IGNORECASE
        )
        if m:
            noun = m.group(1).strip()
            skip = {"image", "camera", "more", "front", "color", "shadow",
                    "left", "right", "single", "same", "following", "statement"}
            if noun.lower() not in skip:
                return noun.lower()

    # Fallback: first long content word
    skip = {"does", "this", "have", "that", "what", "which", "where", "there",
            "with", "answer", "image", "person", "people", "using", "about",
            "from", "into", "over", "only", "according", "chart", "table",
            "based", "following", "statement", "correct"}
    words = re.findall(r"\b[a-zA-Z]{4,}\b", prompt)
    nouns = [w for w in words if w.lower() not in skip]
    return nouns[0].lower() if nouns else "object"


def extract_fallback_nouns(prompt: str, primary: str) -> List[str]:
    """Return content words from the prompt to try if primary noun is absent."""
    skip = {"does", "this", "have", "that", "what", "which", "where", "there",
            "with", "answer", "image", "person", "people", "using", "about",
            "from", "into", "over", "only", "according", "chart", "table",
            "based", "following", "statement", "correct", "answer", "many",
            "number", "your", "please", "letter", "option", "curly", "bracket"}
    words = re.findall(r"\b[a-zA-Z]{3,}\b", prompt)
    seen  = {primary.lower()}
    out   = []
    for w in words:
        wl = w.lower()
        if wl not in skip and wl not in seen:
            seen.add(wl)
            out.append(wl)
    return out[:5]   # at most 5 candidates


# ---------------------------------------------------------------------------
# Overlay helper — jet heatmap over image
# ---------------------------------------------------------------------------

def saliency_to_overlay(
    sal:    torch.Tensor,
    grid_h: int,
    grid_w: int,
    image:  Image.Image,
    alpha:  float = 0.55,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Upsample saliency tensor (n_img_tokens,) to image size and blend with
    the image using a colormap. Alpha=0 → only image; alpha=1 → only heat.
    """
    sal_np  = sal.float().numpy().reshape(grid_h, grid_w)
    # Normalise to [0,1] in case not already
    sal_np  = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
    # Bilinear upsample to image size
    sal_img = Image.fromarray((sal_np * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)
    sal_arr = np.array(sal_img) / 255.0
    # Apply colormap
    heat    = cm.get_cmap(cmap_name)(sal_arr)[..., :3]   # (H, W, 3), RGB
    img_arr = np.array(image.convert("RGB")) / 255.0
    overlay = (1 - alpha) * img_arr + alpha * heat
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model + processor loading (for HSSA)
# ---------------------------------------------------------------------------

def load_model_and_processor():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
    from qwen_vl_utils import process_vision_info

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    print("Loading Qwen2.5-VL-3B-Instruct processor…")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=12845056)
    cfg       = Qwen2_5_VLConfig.from_pretrained(MODEL_ID)
    sms       = int(getattr(cfg.vision_config, "spatial_merge_size", 2))

    print("Loading Qwen2.5-VL-3B-Instruct model (for HSSA)…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    return model, processor, process_vision_info, sms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["pope", "mmvp", "vlm_bias"],
                   choices=list(LOADERS.keys()))
    p.add_argument("--n", type=int, default=5,
                   help="Samples per dataset (mmvp: n pairs = 2n images)")
    p.add_argument("--top_k", type=float, default=0.3)
    p.add_argument("--hssa_layer", type=int, default=16,
                   help="Decoder layer index for HSSA (0-indexed)")
    p.add_argument("--output_dir",
                   default="/volumes2/mllm/lmms-eval/results/salience_vis")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, processor, process_vision_info, sms = load_model_and_processor()

    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        samples = LOADERS[ds_name](args.n)
        ds_dir  = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        for idx, s in enumerate(samples):
            # ---- Build processor inputs ----
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text",  "text":  s["prompt"]},
            ]}]
            txt    = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            img_in, _ = process_vision_info(msgs)
            inputs    = processor(text=[txt], images=img_in,
                                  padding=True, return_tensors="pt")

            thw    = inputs["image_grid_thw"][0]
            grid_h = int(thw[1].item()) // sms
            grid_w = int(thw[2].item()) // sms

            noun  = extract_noun(s["prompt"], ds_name, s.get("topic", ""))
            topic = s.get("topic", "")
            print(f"  [{idx+1:02d}] {topic:15s} noun='{noun}' | "
                  f"{s['prompt'][:55].strip().replace(chr(10),' ')}")

            # ---- CLIP saliency (with fallback noun if primary is absent) ----
            clip_result = clip_sal.compute_clip_salience(
                s["image"], noun, grid_h, grid_w, top_k_pct=args.top_k
            )
            clip_noun = noun
            if not clip_result.object_present:
                for fallback in extract_fallback_nouns(s["prompt"], noun):
                    candidate = clip_sal.compute_clip_salience(
                        s["image"], fallback, grid_h, grid_w, top_k_pct=args.top_k
                    )
                    if candidate.max_sim > clip_result.max_sim:
                        clip_result = candidate
                        clip_noun   = fallback
                        if clip_result.object_present:
                            break

            # ---- HSSA saliency ----
            device_inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                             for k, v in inputs.items()}
            img_start, img_end = patch.get_image_token_range(inputs, processor)
            hssa_result = compute_hssa_salience(
                model, device_inputs, img_start, img_end,
                layer_idx=args.hssa_layer, top_k_pct=args.top_k,
            )

            # ---- Plot ----
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            # Col 0: original
            axes[0].imshow(s["image"])
            q_text = s["prompt"][:100].replace("\n", " ")
            title0 = f"{ds_name} | {topic}\nQ: {q_text}\nGT: {s['gt']}"
            axes[0].set_title(title0, fontsize=6, wrap=True)
            axes[0].axis("off")

            # Col 1: CLIP
            axes[1].imshow(saliency_to_overlay(
                clip_result.saliency, grid_h, grid_w, s["image"], cmap_name="jet"))
            present = "" if clip_result.object_present else " (absent)"
            axes[1].set_title(
                f"CLIP ViT-L/14 | '{clip_noun}'{present}\n"
                f"max_sim={clip_result.max_sim:.2f}  grid={grid_h}×{grid_w}  top_k={args.top_k}",
                fontsize=8)
            axes[1].axis("off")

            # Col 2: HSSA
            axes[2].imshow(saliency_to_overlay(
                hssa_result.saliency, grid_h, grid_w, s["image"], cmap_name="jet"))
            axes[2].set_title(
                f"HSSA | layer={args.hssa_layer}  n_qtoks={hssa_result.n_query_toks}\n"
                f"grid={grid_h}×{grid_w}  top_k={args.top_k}",
                fontsize=8)
            axes[2].axis("off")

            fig.suptitle(
                f"{ds_name} | {topic} | sample {idx+1}", fontsize=9)
            fig.tight_layout()
            out_path = os.path.join(ds_dir, f"sample_{idx+1:02d}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"      → {out_path}")

    print(f"\nDone. Check {args.output_dir}/")


if __name__ == "__main__":
    main()
