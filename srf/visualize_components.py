#!/usr/bin/env python3
"""
visualize_components.py — per-sample walkthrough of the SRF encoder stages.

One row of four panels per sample, for the appendix.

  1  input image
  2  semantic relevance map from the frozen alignment model, overlaid
  3  foveated image, what the vision encoder actually receives
  4  vision attention ratio per (layer, head) for THIS sample, selected slots marked

Every stage is computed by the shipped code. The relevance map comes from
clip_salience, the blur from srf_fovea's own helpers, the VTAR score from
head_calibration. Prompts are built exactly as eval.run_pope and
eval.run_vlmbias build them, including the per-dataset noun extraction mode.

Usage
-----
  python srf/visualize_components.py --dataset vlmbias --topic Flags --n 1
  python srf/visualize_components.py --dataset vlmbias --topic Logos --n 1
  python srf/visualize_components.py --dataset pope --question_id 31
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRF = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SRF / "saliency"))
sys.path.insert(0, str(_SRF))
sys.path.insert(0, str(_SRF.parent / "my_analysis"))

import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from datasets import load_dataset

import clip_salience as clip_sal
import eval as eval_mod
import head_calibration as hc
import qwen_attn_patch as patch
import srf as srf_mod
import srf_fovea as fov

BAND = (6, 31)
TOPK = 0.20

# Record the noun the shipped extractor produced. Guessing a _STATE key is wrong.
_SEEN = {}
_orig_noun = srf_mod.extract_clip_noun


def _rec(q, *a, **k):
    n = _orig_noun(q, *a, **k)
    _SEEN["noun"] = n
    return n


srf_mod.extract_clip_noun = _rec


def pick_samples(dataset, match, qid, topic, n):
    if dataset == "vlmbias":
        ds = load_dataset("anvo25/vlms-are-biased", split="main")
        rows = [r for r in ds
                if (not topic or str(r["topic"]) == topic)
                and (not match or match.lower() in str(r["prompt"]).lower())]
        # Each item appears at several resolutions. Keep the largest of each.
        seen, out = set(), []
        for r in sorted(rows, key=lambda r: -r["image"].size[0]):
            key = str(r["prompt"])[:60] + str(r["ground_truth"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= n:
                break
    else:
        ds = load_dataset("lmms-lab/POPE", split="test")
        out = []
        for r in ds:
            if qid is not None and str(r["question_id"]) != str(qid):
                continue
            if match and match.lower() not in str(r["question"]).lower():
                continue
            out.append(r)
            if len(out) >= n:
                break
    if not out:
        raise SystemExit("No matching sample.")
    return out


def sample_fields(dataset, r):
    """(image, prompt as the model sees it, ground truth, filename tag)."""
    if dataset == "vlmbias":
        return (r["image"].convert("RGB"), str(r["prompt"]).strip(),
                str(r["ground_truth"]), str(r["topic"]).replace(" ", ""))
    suffix = "\nAnswer with Yes or No only."
    return (r["image"].convert("RGB"), str(r["question"]).strip() + suffix,
            str(r.get("answer", "")), str(r["question_id"]))


def build_inputs(processor, image, q):
    """Exactly the prompt construction in eval.run_pope / eval.run_vlmbias."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text",  "text":  q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", padding=True)


def vtar_matrix(model, inp, device, s, e):
    """(n_layers, n_heads) vision attention ratio, via head_calibration."""
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.inference_mode():
        attns = model(**inp, output_attentions=True).attentions
    return np.stack([hc._vision_attention_ratio(a[0], s, e).float().cpu().numpy()
                     for a in attns])


def upsample(vec, gh, gw, size):
    t = torch.tensor(np.asarray(vec), dtype=torch.float32).reshape(1, 1, gh, gw)
    W, H = size
    return torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear",
                                           align_corners=False)[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser(description="SRF encoder-stage walkthrough.")
    ap.add_argument("--model", default=CFG.DEFAULT_MODEL)
    ap.add_argument("--dataset", default="vlmbias", choices=["vlmbias", "pope"])
    ap.add_argument("--topic", default=None, help="VLMBias topic, e.g. Flags or Logos.")
    ap.add_argument("--match", default=None)
    ap.add_argument("--question_id", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--no_cbar", action="store_true",
                   help="Omit the per-figure colourbar. Used when several rows "
                        "are combined into one figure that carries a single "
                        "shared colourbar.")
    ap.add_argument("--no_suptitle", action="store_true",
                   help="Omit the question header, for composites that draw it.")
    ap.add_argument("--out", default="results/components/")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Foveal blur scale in pixels. Default None uses the shipped "
                         "srf_fovea.SIGMA. Raising it above the shipped value renders a "
                         "figure the method does not produce, so disclose it in the caption.")
    args = ap.parse_args()

    if args.sigma is not None:
        fov.SIGMA = float(args.sigma)
        print(f"  [fovea] sigma overridden -> {fov.SIGMA:g}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pick_samples(args.dataset, args.match, args.question_id, args.topic, args.n)

    model, processor = eval_mod.load_model(args.model)
    device = next(model.parameters()).device
    img_tok = processor.tokenizer.convert_tokens_to_ids(CFG.IMAGE_TOKEN)
    srf_mod.setup(model, processor, calib_dataset=args.dataset)

    for r in rows:
        image, q, gt, tag = sample_fields(args.dataset, r)
        inp = build_inputs(processor, image, q)
        s, e = eval_mod.get_img_range(inp["input_ids"], img_tok)
        gh, gw = clip_sal.get_grid_dims(inp, 2)

        vtar = vtar_matrix(model, inp, device, s, e)

        # SRF stages, through the shipped code and the dataset's own parameters.
        srf_mod.reset_for_dataset(dataset=args.dataset, layer_start=BAND[0],
                                  layer_end=BAND[1], head_top_k_pct=TOPK)
        inp2 = build_inputs(processor, image, q)
        fov.prepare_sample(inp2, s, e, image, q, model, processor)
        sal = patch._STATE.get("salience_mask")
        gated = sal is not None
        if gated:
            w = fov._saliency_to_weight(sal, *image.size, gh, gw)
            fim = fov._apply_foveal_blur(image, w, fov.SIGMA)
            relmap = upsample(sal.float().cpu().numpy(), gh, gw, image.size)
        else:
            fim, relmap = image, np.zeros(image.size[::-1], dtype=np.float32)
        lcr = dict(srf_mod.last_clip_result)
        srf_mod.cleanup()

        n_l, n_h = vtar.shape
        sel = np.zeros_like(vtar, dtype=bool)
        k = max(1, round(n_h * TOPK))
        for L in range(BAND[0], min(BAND[1], n_l - 1) + 1):
            sel[L, np.argsort(-vtar[L])[:k]] = True

        # Paper style. All four axes are forced to the SAME BOX SHAPE with
        # set_box_aspect, using the photo's aspect. Without it the photos
        # letterbox inside their boxes while imshow with a numeric aspect
        # resizes its own box, so panel (d) came out 16 percent taller than
        # (a) to (c). With equal box aspect the photos fill their boxes
        # exactly and the heatmap, drawn with aspect="auto", fills its box
        # too, so all four render identically.
        _W, _H = image.size
        _BA = _H / _W
        fig, axes = plt.subplots(1, 4, figsize=(21, 5.6),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1],
                                              "wspace": 0.20})
        im = np.array(image)
        for a in axes[:3]:
            a.imshow(im); a.axis("off"); a.set_box_aspect(_BA)
        axes[1].imshow(relmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].clear(); axes[2].imshow(np.array(fim))
        axes[2].axis("off"); axes[2].set_box_aspect(_BA)

        axes[0].set_title("(a) Original image", fontsize=21)
        axes[1].set_title("(b) Semantic relevance map", fontsize=21)
        axes[2].set_title("(c) Semantic foveation", fontsize=21)

        # The question sits under panel (a) as its x label.
        question_only = q.split("Answer with")[0].strip().rstrip()
        axes[0].axis("on")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for sp in axes[0].spines.values():
            sp.set_visible(False)
        axes[0].set_xlabel(f'"{question_only}"', fontsize=22, labelpad=8,
                           loc="left")

        ax = axes[3]
        h = ax.imshow(vtar, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.85)
        ax.set_box_aspect(_BA)
        ys, xs = np.where(sel)
        ax.scatter(xs, ys, s=11, c="red", marker="s", linewidths=0)
        ax.set_title("(d) Vision attention ratio", fontsize=21)
        ax.set_xlabel("head", fontsize=18)
        ax.set_ylabel("layer", fontsize=18)
        ax.tick_params(labelsize=15)

        if not args.no_cbar:
            cb = fig.colorbar(h, ax=ax, fraction=0.045)
            cb.ax.tick_params(labelsize=13)
            cb.set_label("vision attention ratio", fontsize=14)

        sim = lcr.get("full_img_sim")
        p = out / f"components_{args.dataset}_{tag}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {p}\n      noun={_SEEN.get('noun')!r}  "
              f"CLIP sim={sim if sim is None else round(sim, 3)}  "
              f"gate={'fired' if gated else 'rejected'}")


if __name__ == "__main__":
    main()
