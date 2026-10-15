#!/usr/bin/env python3
"""
logit_lens_animals.py — where in the network does the answer "4" get decided?

VLMBias Animals shows a 5-legged horse and asks how many legs it has. The
model answers 4 under every intervention tried, including foveation that
keeps the legs sharp and grey-out that deletes the rest of the animal. This
asks a different question. Not whether we can change the answer, but at which
layer the answer is already fixed.

Method. The prompt ends with the assistant writing "{", so the very next token
is the digit. At every decoder layer the hidden state at that position is
passed through the final norm and the LM head, and P("4") and P("5") are read
off. If P(4) is already dominant in early layers the answer is a language
prior that never consulted the image. If it only becomes dominant late, the
visual evidence was present and then overridden.

Three input conditions per image, so the comparison is like for like.
  original   unmodified image
  foveated   SRF foveation, sigma=20, legs sharp and background blurred
  grey       binary top-30 percent mask with flat grey fill, animal deleted

Usage
-----
  python srf/logit_lens_animals.py --n 4
"""
from __future__ import annotations
import argparse, os, pathlib, sys

_S = pathlib.Path(__file__).parent
sys.path.insert(0, str(_S / "saliency")); sys.path.insert(0, str(_S))
sys.path.insert(0, str(_S.parent / "my_analysis"))
import config as CFG
os.environ.setdefault("HF_HOME", CFG.HF_HOME)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
import clip_salience as cs, eval as ev, srf_fovea as fov
from datasets import load_dataset
from noun_extract import extract_clip_noun


def variants(img, gh, gw, sal):
    fov.FILL_MODE, fov.MASK_MODE = "blur", "linear"
    w = fov._saliency_to_weight(sal, *img.size, gh, gw)
    fovd = fov._apply_foveal_blur(img, w, 20.0)
    fov.FILL_MODE, fov.MASK_MODE, fov.MASK_Q = "grey", "binary", 0.70
    w2 = fov._saliency_to_weight(fov._transform_mask(sal), *img.size, gh, gw)
    grey = fov._apply_foveal_blur(img, w2, 50.0)
    fov.FILL_MODE, fov.MASK_MODE = "blur", "linear"
    return {"original": img, "foveated": fovd, "grey": grey}


def lens(model, proc, image, question, ids):
    """Per-layer P(token) at the digit position. Returns (n_layers+1, n_tok)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "{"
    inp = proc(text=[text], images=[image], return_tensors="pt", padding=True)
    dev = next(model.parameters()).device
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True)
    lm = model.language_model if hasattr(model, "language_model") else model.model
    norm = getattr(lm, "norm", None) or getattr(lm.model, "norm")
    head = model.lm_head
    probs = []
    for h in out.hidden_states:
        logits = head(norm(h[:, -1, :]))
        p = torch.softmax(logits.float(), dim=-1)[0]
        probs.append([float(p[i]) for i in ids])
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/logit_lens/"); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("anvo25/vlms-are-biased", split="main")
    rows, seen = [], set()
    for r in sorted(ds, key=lambda r: -r["image"].size[0]):
        if str(r["topic"]) != "Animals": continue
        k = r["image"].size + (str(r["ground_truth"]),)
        if k in seen: continue
        seen.add(k); rows.append(r)
        if len(rows) >= a.n: break

    model, proc = ev.load_model(CFG.DEFAULT_MODEL)
    tok = proc.tokenizer
    ids = [tok.encode("4")[0], tok.encode("5")[0]]
    print(f"  token ids: '4'={ids[0]}  '5'={ids[1]}")

    fig, ax = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.4), squeeze=False)
    for i, r in enumerate(rows):
        img = r["image"].convert("RGB")
        q = str(r["prompt"]).strip()
        inp0 = proc(text=["x"], images=[img], return_tensors="pt", padding=True)
        gh, gw = cs.get_grid_dims(inp0, 2)
        sal = cs.compute_clip_salience_full_gate_v3(img, extract_clip_noun(q, mode="vlmbias"),
                                                    gh, gw).saliency.float()
        for name, im in variants(img, gh, gw, sal).items():
            p = lens(model, proc, im, q, ids)
            L = np.arange(p.shape[0])
            ax[0][i].plot(L, p[:, 0], label=f"{name}: P('4')")
            ax[0][i].plot(L, p[:, 1], "--", label=f"{name}: P('5')")
            print(f"  img{i} {name:<9} final P(4)={p[-1,0]:.3f}  P(5)={p[-1,1]:.3f}  "
                  f"first layer P(4)>P(5): "
                  f"{int(np.argmax(p[:,0] > p[:,1])) if (p[:,0] > p[:,1]).any() else -1}")
        ax[0][i].set_title(f"Animals #{i}  gt=5", fontsize=11)
        ax[0][i].set_xlabel("decoder layer"); ax[0][i].set_ylabel("probability")
        ax[0][i].legend(fontsize=6); ax[0][i].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "logit_lens_animals.png", dpi=130, bbox_inches="tight")
    print("  saved ->", out / "logit_lens_animals.png")


if __name__ == "__main__":
    main()
