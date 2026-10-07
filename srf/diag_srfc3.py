"""
diag_srfc3.py — Sanity check for srf_c_v3 (embedding-space visual token zeroing)
               on 10 MME existence samples.

Runs 4 passes per sample:
  (A) Baseline  — no intervention
  (B) SRF-E     — zero whole image pixels (original; known to fail on LLaVA)
  (C) SRF-C v2  — zero salient pixels (still corrupts CLIP)
  (D) SRF-C v3  — zero visual tokens in LLM embedding space (bypass CLIP)

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  source activate mllm
  python srf/diag_srfc3.py --n 10
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_SRF_DIR      = pathlib.Path(__file__).parent
_ANALYSIS_DIR = _SRF_DIR.parent / "my_analysis"
sys.path.insert(0, str(_SRF_DIR / "saliency"))
sys.path.insert(0, str(_SRF_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

import torch
from PIL import Image

MME_DIR = pathlib.Path(
    "/home/sgowda/workspace/ILVAD/data/MME/MME_Benchmark_release_version/MME_Benchmark"
)
MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_mme_existence(n: int):
    task_dir = MME_DIR / "existence"
    samples = []
    for img_path in sorted(task_dir.glob("*.jpg"))[:n * 2]:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                samples.append({"image": img_path, "question": parts[0], "answer": parts[1]})
        if len(samples) >= n:
            break
    return samples[:n]


def decode_first_token(model, inp):
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",     type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.3)
    args = p.parse_args()

    from eval import load_model, _encode_llava
    import srf as srf_mod
    import srf_e as srfe_mod
    import srf_c_v2 as srfc2_mod
    import srf_c_v3 as srfc3_mod
    from llava_attn_patch import patch_model, _STATE as PATCH_STATE
    import llava_attn_patch as patch_mod

    print(f"Loading {MODEL_ID}…")
    model, processor = load_model(MODEL_ID)
    device = next(model.parameters()).device

    for mod in [srf_mod, srfe_mod, srfc2_mod, srfc3_mod]:
        mod.patch = patch_mod

    srf_mod.setup(model, processor, calib_dataset="mme")
    srf_mod.reset_for_dataset(phase="generation", alpha=0.5, layer_end=20,
                              eps=0.2, dataset="mme")
    srfc3_mod.reset_for_dataset(phase="generation", alpha=0.5, layer_end=20,
                                eps=0.2, dataset="mme")

    tok = processor.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id  = tok.encode("No",  add_special_tokens=False)[0]

    samples = load_mme_existence(args.n)
    print(f"\nLoaded {len(samples)} samples. γ={args.gamma}\n")
    print(f"{'#':<3}  {'GT':<4}  {'Base':>5}  {'SRF-E':>6}  {'v2':>5}  {'v3':>5}  {'fallback?'}")
    print("-" * 60)

    counts = {"base": 0, "srfe": 0, "v2": 0, "v3": 0}

    for i, s in enumerate(samples):
        image    = Image.open(s["image"]).convert("RGB")
        question = s["question"]
        gt       = s["answer"].strip()

        inp, img_start, img_end = _encode_llava(processor, model, device, image, question)

        srf_mod.prepare_sample(inp, img_start, img_end, image, question, model, processor)
        srfc3_mod.prepare_sample(inp, img_start, img_end, image, question, model, processor)

        fallback = patch_mod._STATE.get("salience_mask") is None

        # (A) Baseline
        patch_mod._STATE["method"] = "baseline"
        logits_base = decode_first_token(model, inp)
        pred_base = "Yes" if logits_base[0, yes_id] > logits_base[0, no_id] else "No"

        # (B) SRF-E (zero whole image pixels)
        logits_srfe = srfe_mod.get_contrastive_logits(model, inp, gamma=args.gamma)
        pred_srfe = "Yes" if logits_srfe[0, yes_id] > logits_srfe[0, no_id] else "No"

        # (C) SRF-C v2 (zero salient pixels)
        logits_v2 = srfc2_mod.get_contrastive_logits(model, inp, gamma=args.gamma)
        pred_v2 = "Yes" if logits_v2[0, yes_id] > logits_v2[0, no_id] else "No"

        # (D) SRF-C v3 (zero visual tokens in embedding space)
        logits_v3 = srfc3_mod.get_contrastive_logits(model, inp, gamma=args.gamma)
        pred_v3 = "Yes" if logits_v3[0, yes_id] > logits_v3[0, no_id] else "No"

        def mark(p): return "✓" if p == gt else "✗"

        counts["base"] += (pred_base == gt)
        counts["srfe"] += (pred_srfe == gt)
        counts["v2"]   += (pred_v2   == gt)
        counts["v3"]   += (pred_v3   == gt)

        fb_str = "fallback" if fallback else ""
        print(f"{i+1:<3}  {gt:<4}  {pred_base+mark(pred_base):>5}  "
              f"{pred_srfe+mark(pred_srfe):>6}  {pred_v2+mark(pred_v2):>5}  "
              f"{pred_v3+mark(pred_v3):>5}  {fb_str}")

    n = len(samples)
    print("-" * 60)
    print(f"Acc  Base={counts['base']}/{n}  SRF-E={counts['srfe']}/{n}  "
          f"v2={counts['v2']}/{n}  v3={counts['v3']}/{n}")


if __name__ == "__main__":
    main()
