"""
diag_srfc2.py — Sanity check for srf_c_v2 on a handful of MME existence samples.

Runs 3 passes per sample:
  (A) Baseline  — no intervention
  (B) SRF-E     — zero whole image (original contrastive)
  (C) SRF-C v2  — zero salient regions only (new method)

Prints per-sample:
  - saliency mask stats (n_salient tokens, pixel % zeroed)
  - yes/no logit delta for each method vs baseline
  - predicted answer for each method
  - whether SRF-C v2 mask was active (or fell back to zero-image)

Usage:
  cd /home/sgowda/workspace/SRF/lmms-eval
  source activate mllm
  python srf/diag_srfc2.py --n 10
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
import torch.nn.functional as F
from PIL import Image

import clip_salience as clip_sal

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


def decode_first_token_logits(model, inp):
    with torch.inference_mode():
        out = model(**inp)
    return out.logits[:, -1, :].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",     type=int, default=10, help="Number of samples")
    p.add_argument("--gamma", type=float, default=0.3)
    args = p.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from eval import load_model, _encode_llava
    import srf as srf_mod
    import srf_e as srfe_mod
    import srf_c_v2 as srfc2_mod
    from llava_attn_patch import patch_model, _STATE as PATCH_STATE

    print(f"Loading {MODEL_ID}…")
    model, processor = load_model(MODEL_ID)
    device = next(model.parameters()).device

    patch_mod = sys.modules.get("llava_attn_patch")
    if patch_mod is None:
        import llava_attn_patch as patch_mod

    # Inject patch into all methods
    for mod in [srf_mod, srfe_mod, srfc2_mod]:
        mod.patch = patch_mod

    # Setup SRF (calibrate once — used by all methods)
    srf_mod.setup(model, processor, calib_dataset="mme")
    srf_mod.reset_for_dataset(
        phase="generation", alpha=0.5, layer_end=20,
        eps=0.2, dataset="mme",
    )

    tok = processor.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id  = tok.encode("No",  add_special_tokens=False)[0]

    samples = load_mme_existence(args.n)
    print(f"\nLoaded {len(samples)} samples. γ={args.gamma}\n")
    print(f"{'#':<3}  {'GT':<4}  {'Base':>5}  {'SRF-E':>6}  {'SRFCv2':>7}  "
          f"{'mask%':>6}  {'fallback?'}")
    print("-" * 60)

    n_correct_base = n_correct_srfe = n_correct_srfc2 = 0

    for i, s in enumerate(samples):
        image   = Image.open(s["image"]).convert("RGB")
        question = s["question"]
        gt       = s["answer"].strip()

        # Encode
        inp, img_start, img_end = _encode_llava(processor, model, device, image, question)

        # SRF prepare (sets saliency mask in patch._STATE)
        srf_mod.prepare_sample(inp, img_start, img_end, image, question, model, processor)

        salience_mask = patch_mod._STATE.get("salience_mask")
        fallback = salience_mask is None

        # Pixel % zeroed by SRF-C v2
        if salience_mask is not None:
            grid_h, grid_w = clip_sal.get_grid_dims(inp, 2, "llava")
            pv = inp["pixel_values"]
            H, W = pv.shape[-2], pv.shape[-1]
            sal_2d   = salience_mask.reshape(grid_h, grid_w).unsqueeze(0).unsqueeze(0).float()
            sal_full = F.interpolate(sal_2d, size=(H, W), mode="bilinear", align_corners=False)
            mask_pct = (sal_full.squeeze() > 0.5).float().mean().item() * 100
        else:
            mask_pct = 100.0  # fell back to zero-image

        # ── (A) Baseline ───────────────────────────────────────────────────
        patch_mod._STATE["method"] = "baseline"
        logits_base = decode_first_token_logits(model, inp)
        pred_base = "Yes" if logits_base[0, yes_id] > logits_base[0, no_id] else "No"

        # ── (B) SRF-E (zero whole image) ───────────────────────────────────
        logits_srfe = srfe_mod.get_contrastive_logits(model, inp, gamma=args.gamma)
        pred_srfe = "Yes" if logits_srfe[0, yes_id] > logits_srfe[0, no_id] else "No"

        # ── (C) SRF-C v2 (zero salient regions) ────────────────────────────
        logits_srfc2 = srfc2_mod.get_contrastive_logits(model, inp, gamma=args.gamma)
        pred_srfc2 = "Yes" if logits_srfc2[0, yes_id] > logits_srfc2[0, no_id] else "No"

        correct_base  = pred_base  == gt
        correct_srfe  = pred_srfe  == gt
        correct_srfc2 = pred_srfc2 == gt
        n_correct_base  += correct_base
        n_correct_srfe  += correct_srfe
        n_correct_srfc2 += correct_srfc2

        base_mark  = "✓" if correct_base  else "✗"
        srfe_mark  = "✓" if correct_srfe  else "✗"
        srfc2_mark = "✓" if correct_srfc2 else "✗"
        fb_str = "YES (zero-img)" if fallback else ""

        print(f"{i+1:<3}  {gt:<4}  {pred_base+base_mark:>5}  "
              f"{pred_srfe+srfe_mark:>6}  {pred_srfc2+srfc2_mark:>7}  "
              f"{mask_pct:>5.1f}%  {fb_str}")

    n = len(samples)
    print("-" * 60)
    print(f"Acc  Baseline={n_correct_base}/{n}  "
          f"SRF-E={n_correct_srfe}/{n}  "
          f"SRF-C v2={n_correct_srfc2}/{n}")


if __name__ == "__main__":
    main()
