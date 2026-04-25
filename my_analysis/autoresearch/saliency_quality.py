#!/usr/bin/env python3
"""
Saliency quality assessment with balanced baseline sampling.

SAMPLING STRATEGY
-----------------
Rather than random samples (88% correct baseline → few hard cases), this script:
  1. Runs baseline inference on a large pool (N_POOL samples)
  2. Selects N_WRONG_TARGET wrong + N_CORRECT_TARGET correct samples
  3. Runs MULTIPLE saliency configs on the balanced set

This gives a 50/50 hard/easy mix so improvements to salient regions
are visible against real failure modes.

SALIENCY CONFIGS COMPARED
--------------------------
  CLIP-7x7-soft    — default, 7×7 coarse grid, soft saliency, top-30%
  CLIP-5x5-soft    — coarser grid (larger patches, better for small objects)
  CLIP-9x9-soft    — finer grid (more spatial precision)
  CLIP-7x7-binary  — hard top-k mask instead of soft
  HSSA-L8          — early decoder layer (raw image features)
  HSSA-L12         — early-mid layer
  HSSA-L16         — mid layer (current default)
  HSSA-L20         — mid-late layer
  HSSA-L24         — late layer (high-level semantics)
  Ens-7030         — CLIP-7x7-soft (70%) + HSSA-L16 (30%)
  Ens-5050         — CLIP-7x7-soft (50%) + HSSA-L16 (50%)

VISUALISATIONS
--------------
  vis/quality_NN_<noun>_<gt>_<CORRECT|WRONG>.png
Each figure title includes: question | GT | Pred | CLIP max_sim

Usage:
    cd /volumes2/mllm/lmms-eval
    CUDA_VISIBLE_DEVICES=0 conda run -n mllm python \\
        my_analysis/autoresearch/saliency_quality.py 2>&1 | tee sal_quality.log
"""
from __future__ import annotations
import json, os, pathlib, random, sys
os.environ["HF_HOME"] = "/volumes2/hugging_face_cache"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SCRIPT_DIR   = pathlib.Path(__file__).parent
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import torch
import torch.nn.functional as F
from datasets import load_dataset as hf_load
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import clip_salience as clip_sal
import qwen_attn_patch as patch

MODEL_ID         = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_TOKEN      = "<|image_pad|>"
N_POOL           = 400    # baseline inference pool to find wrong/correct split
N_WRONG_TARGET   = 50     # wrong baseline predictions to include
N_CORRECT_TARGET = 50     # correct baseline predictions to include
SEED             = 42
HSSA_LAYERS      = [8, 12, 16, 20, 24]
VIS_DIR          = SCRIPT_DIR / "vis"
QWEN_SPECIAL_TOK = 151644


# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}…")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager").eval()
processor    = AutoProcessor.from_pretrained(MODEL_ID)
img_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
patch.patch_model(model, "baseline", 1.0)
device = next(model.parameters()).device


def get_inputs(image, question):
    msgs = [{"role":"user","content":[
        {"type":"image","image":image},{"type":"text","text":question}]}]
    text    = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inp = processor(text=[text], images=img_in, return_tensors="pt", padding=True).to(device)
    ids = inp["input_ids"][0].tolist()
    s2  = next(i for i,t in enumerate(ids) if t==img_token_id)
    e2  = len(ids)-1-next(i for i,t in enumerate(reversed(ids)) if t==img_token_id)
    return inp, s2, e2


def get_pred(inp):
    patch._STATE["method"] = "baseline"
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=5, do_sample=False)
    s = processor.decode(out[0, inp["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip().lower()
    return "yes" if s.startswith("yes") else "no"


def hssa_multi_layer(model, inputs, img_start, img_end, layers=HSSA_LAYERS):
    """One forward pass → saliency at multiple HSSA layers."""
    model_inputs = {k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)

    results = {}
    tok_ids = inputs["input_ids"][0].cpu()[img_end + 1:]
    keep    = tok_ids < QWEN_SPECIAL_TOK
    for L in layers:
        hs     = outputs.hidden_states[L + 1][0].float()
        h_img  = hs[img_start : img_end + 1]
        h_post = hs[img_end + 1:]
        h_txt  = h_post[keep] if keep.any() else h_post

        h_img_n = F.normalize(h_img, dim=-1)
        h_txt_n = F.normalize(h_txt, dim=-1)
        sims    = (h_img_n @ h_txt_n.T).max(dim=-1).values.cpu()

        p5, p95  = torch.quantile(sims, 0.05), torch.quantile(sims, 0.95)
        saliency = (sims - p5) / (p95 - p5 + 1e-8)
        results[L] = saliency.clamp(0.0, 1.0)
    return results


# ── Phase 1: run baseline on pool to find balanced set ───────────────────────
print(f"\nPhase 1: baseline inference on {N_POOL} pool samples to find wrong/correct split…")
ds   = hf_load("lmms-lab/POPE", split="test")
pool = [r for r in ds if str(r.get("category","")).strip().lower() == "adversarial"]
rng  = random.Random(SEED); rng.shuffle(pool); pool = pool[:N_POOL]

wrong_samples, correct_samples = [], []

for i, r in enumerate(pool):
    gt = "yes" if str(r["answer"]).strip().lower() == "yes" else "no"
    q  = str(r["question"]).strip() + "\nAnswer with Yes or No only."
    img = r["image"].convert("RGB")

    if len(wrong_samples) >= N_WRONG_TARGET and len(correct_samples) >= N_CORRECT_TARGET:
        break

    inp, _, _ = get_inputs(img, q)
    pred      = get_pred(inp)
    ok        = pred == gt

    entry = {"image": img, "question": q, "gt": gt, "pred_baseline": pred}
    if not ok and len(wrong_samples) < N_WRONG_TARGET:
        wrong_samples.append(entry)
    elif ok and len(correct_samples) < N_CORRECT_TARGET:
        correct_samples.append(entry)

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{N_POOL}] wrong={len(wrong_samples)} correct={len(correct_samples)}")

print(f"\n  Balanced set: {len(wrong_samples)} wrong + {len(correct_samples)} correct")
if len(wrong_samples) < N_WRONG_TARGET:
    print(f"  WARNING: only found {len(wrong_samples)} wrong cases (pool too small?)")

samples = (
    [dict(s, group="WRONG")   for s in wrong_samples] +
    [dict(s, group="CORRECT") for s in correct_samples]
)
print(f"  Total evaluation samples: {len(samples)}")


# ── Phase 2: saliency quality on balanced set ─────────────────────────────────
print(f"\nPhase 2: saliency quality assessment on {len(samples)} balanced samples…")
VIS_DIR.mkdir(parents=True, exist_ok=True)
records = []

for idx, s in enumerate(samples):
    print(f"\n── sample {idx+1}/{len(samples)} [{s['group']}] ──")
    inp, img_s, img_e = get_inputs(s["image"], s["question"])
    grid_h, grid_w    = clip_sal.get_grid_dims(inp, 2)
    noun              = clip_sal.extract_query_noun(s["question"])

    print(f"  noun='{noun}'  gt={s['gt']}  pred_baseline={s['pred_baseline']}  [{s['group']}]")

    # ── CLIP: multiple configs (one GPU call each, CLIP is tiny) ─────────────
    clip_cfgs = {
        "CLIP-7x7-soft":   dict(coarse_n=7, top_k_pct=0.30, use_soft=True),
        "CLIP-5x5-soft":   dict(coarse_n=5, top_k_pct=0.30, use_soft=True),
        "CLIP-9x9-soft":   dict(coarse_n=9, top_k_pct=0.30, use_soft=True),
        "CLIP-7x7-binary": dict(coarse_n=7, top_k_pct=0.30, use_soft=False),
    }
    clip_results = {}
    clip_sals    = {}
    for cfg_name, cfg in clip_cfgs.items():
        res = clip_sal.compute_clip_salience(
            s["image"], s["question"], grid_h, grid_w,
            top_k_pct=cfg["top_k_pct"], coarse_n=cfg["coarse_n"])
        sal = res.saliency if cfg["use_soft"] else res.mask
        clip_results[cfg_name] = res
        clip_sals[cfg_name]    = sal

    # Use default CLIP for ensemble
    default_clip_sal = clip_sals["CLIP-7x7-soft"]
    default_max_sim  = clip_results["CLIP-7x7-soft"].max_sim
    print(f"  CLIP-7x7: max_sim={default_max_sim:.3f}  "
          f"object_present={clip_results['CLIP-7x7-soft'].object_present}")

    # ── HSSA: all layers in ONE forward pass ───────────────────────────────
    hssa_sals = hssa_multi_layer(model, inp, img_s, img_e, HSSA_LAYERS)
    for L, sal in hssa_sals.items():
        print(f"  HSSA-L{L:2d}: mean={sal.mean():.3f}  max={sal.max():.3f}")

    # ── Ensembles (CLIP-7x7-soft + HSSA-L16) ─────────────────────────────
    def _ens(wc, wh, hssa_layer=16):
        sh = hssa_sals[hssa_layer]
        c  = wc * default_clip_sal + wh * sh
        mn, mx = c.min(), c.max()
        return (c - mn) / (mx - mn + 1e-8)

    ens_7030 = _ens(0.7, 0.3)
    ens_5050 = _ens(0.5, 0.5)

    # ── Rich visualisation ─────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.cm as cm
        import numpy as np
        from PIL import Image as PILImage

        def overlay(sal_t, image):
            sal_np  = sal_t.float().cpu().numpy().reshape(grid_h, grid_w)
            mn, mx  = sal_np.min(), sal_np.max()
            sal_np  = (sal_np - mn) / (mx - mn + 1e-8)
            sal_img = PILImage.fromarray((sal_np*255).astype("uint8")
                      ).resize(image.size, PILImage.BILINEAR)
            sal_arr = np.array(sal_img) / 255.0
            heat    = cm.get_cmap("jet")(sal_arr)[..., :3]
            img_arr = np.array(image.convert("RGB")) / 255.0
            return (np.clip(0.45*img_arr + 0.55*heat, 0, 1)*255).astype("uint8")

        panels = [("Original", np.array(s["image"].convert("RGB")))]
        for cfg_name, sal in clip_sals.items():
            ms = clip_results[cfg_name].max_sim
            panels.append((f"{cfg_name}\n(sim={ms:.2f})", overlay(sal, s["image"])))
        for L in HSSA_LAYERS:
            panels.append((f"HSSA-L{L}", overlay(hssa_sals[L], s["image"])))
        panels.append((f"Ens70/30", overlay(ens_7030, s["image"])))
        panels.append((f"Ens50/50", overlay(ens_5050, s["image"])))

        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5*n_cols, 4))
        for ax, (title, arr) in zip(axes, panels):
            ax.imshow(arr); ax.set_title(title, fontsize=6); ax.axis("off")

        # Title: question (truncated) | GT | Pred | CLIP threshold
        q_short = s["question"].replace("\nAnswer with Yes or No only.", "").strip()
        q_short = q_short[:70] + ("…" if len(q_short) > 70 else "")
        marker  = "✓" if s['pred_baseline'] == s['gt'] else "✗"
        fig.suptitle(
            f"#{idx+1} [{s['group']}] {marker}  Q: \"{q_short}\"\n"
            f"GT: {s['gt']}  Pred: {s['pred_baseline']}  "
            f"CLIP-max_sim: {default_max_sim:.3f}  noun: '{noun}'",
            fontsize=7, y=1.02)
        fig.tight_layout()

        safe_noun = noun.replace(" ","_").replace("/","_")[:20]
        out = VIS_DIR / (f"quality_{idx+1:03d}_{safe_noun}_"
                         f"gt{s['gt']}_pred{s['pred_baseline']}_{s['group']}.png")
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out.name}")
    except Exception as e:
        print(f"  [vis] WARN: {e}")

    records.append({
        "idx": idx+1, "group": s["group"],
        "noun": noun, "gt": s["gt"], "pred_baseline": s["pred_baseline"],
        "correct_baseline": s["pred_baseline"] == s["gt"],
        "clip_max_sim": float(default_max_sim),
        "clip_object_present": clip_results["CLIP-7x7-soft"].object_present,
        "hssa_sal_means": {L: float(hssa_sals[L].mean()) for L in HSSA_LAYERS},
    })


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
wrong_recs   = [r for r in records if r["group"] == "WRONG"]
correct_recs = [r for r in records if r["group"] == "CORRECT"]

print(f"\n  WRONG cases ({len(wrong_recs)}) — clip max_sim distribution:")
sims_w = [r["clip_max_sim"] for r in wrong_recs]
if sims_w:
    print(f"    min={min(sims_w):.3f}  max={max(sims_w):.3f}  mean={sum(sims_w)/len(sims_w):.3f}")
    print(f"    below 0.25: {sum(1 for s in sims_w if s<0.25)}  "
          f"0.25-0.30: {sum(1 for s in sims_w if 0.25<=s<0.30)}  "
          f"above 0.30: {sum(1 for s in sims_w if s>=0.30)}")
    print(f"    nouns: {[r['noun'] for r in wrong_recs[:10]]}")

print(f"\n  CORRECT cases ({len(correct_recs)}) — clip max_sim distribution:")
sims_c = [r["clip_max_sim"] for r in correct_recs]
if sims_c:
    print(f"    min={min(sims_c):.3f}  max={max(sims_c):.3f}  mean={sum(sims_c)/len(sims_c):.3f}")
    print(f"    below 0.25: {sum(1 for s in sims_c if s<0.25)}  "
          f"0.25-0.30: {sum(1 for s in sims_c if 0.25<=s<0.30)}  "
          f"above 0.30: {sum(1 for s in sims_c if s>=0.30)}")

print(f"\n  Suggested clip_suppress_thresh range: look for the value that separates")
print(f"  the WRONG case max_sim distribution from CORRECT case max_sim distribution.")
print(f"  (Higher sim → likely present object → boost; lower → absent → suppress)")

out_json = SCRIPT_DIR / "last_saliency_quality.json"
with open(out_json, "w") as f:
    json.dump(records, f, indent=2)
print(f"\n  Stats → {out_json}")
print(f"  Vis  → {VIS_DIR}/quality_*.png")
print(f"\n  INSPECT the vis to find which saliency config best localises objects.")
print(f"  Then update SALIENCY['hssa_layer'] and 'clip_suppress_thresh' in srf.py.")
