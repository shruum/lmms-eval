# Follow-up project: post-training for visual evidence grounding

> Scoping document, written 2026-10-07, revised 2026-10-08. For a new paper,
> separate from the SRF ICLR submission. Read `SRF_details.md` first for what
> was already measured. This file is the plan, the literature position, and the
> open decisions. **Nothing here has been executed.**

**Target machine.** Snellius, A100 and H100. The relevant baselines train on
1.4k to 21.5k samples with LoRA, so this project is **data-construction bound,
not compute bound**.

---

## 0. Term hygiene, read before writing anything

This is a new paper for a new venue. **None of the SRF vocabulary carries over.**

| Retire | Reason |
|---|---|
| SRF, Semantic Re-Focus | the prior paper's name |
| semantic foveation, foveation | prior component, and the name the FoveateR / GazeVLM line already owns |
| semantic relevance map | prior term |
| attention re-focus, re-focusing | prior term |
| vision-responsive heads, VTAR | prior term |
| `lambda_sem`, `lambda_bg`, `lambda_sys` | prior notation |

Build the vocabulary around **visual evidence**. Evidence region, evidence
grounding, evidence dependence, counterfactual evidence supervision, prior
suppression. Pick the final name late.

**Citation decision to settle early.** If the SRF submission is live or
published, this paper cites it as related work, third person under anonymous
review. The evidence-estimation step genuinely overlaps, so decide how much of
it is presented as new before writing Section 3.

---

## 1. Terminology, because the literature is sloppy about it

Everything splits on **what actually changes**.

| | What changes | Gradient steps | Examples |
|---|---|---|---|
| Prompting / ICL / few-shot | the context window | **none** | k examples in the prompt |
| Decoding intervention | the output distribution | none | VCD, contrastive decoding |
| Activation / attention editing | internal activations at runtime | none | **the prior SRF work**, VAF, ILVAD |
| **Post-training** | **the weights** | yes | SFT, DPO, RLHF, GRPO |

**Post-training** is the umbrella for anything after pretraining.
**Fine-tuning** is the older looser synonym, often used to mean SFT
specifically. Do not agonise over the distinction, they overlap almost
entirely.

Inside post-training, two independent axes get conflated constantly.

**Axis 1, the objective.**

| Objective | Sampling loop | Reward model |
|---|---|---|
| SFT, cross-entropy on target tokens | no | no |
| DPO / IPO / KTO / SimPO, offline preference | no | no |
| RLHF-PPO | yes | yes |
| RLVR / GRPO | yes | no, verifiable |

**Axis 2, which weights.** Full fine-tune, LoRA, projector only, vision encoder
frozen or not. Independent of the objective.

**ICL and few-shot are not training.** Zero gradient steps, nothing persists.
They appear in this literature only as cheap baselines.

### What DPO actually is

Classic RLHF is three stages, SFT then a reward model then PPO. DPO
(Rafailov et al., NeurIPS 2023) observes that the optimal RLHF policy has a
closed form in the reward, inverts it, and optimises the policy directly. No
reward model, no sampling loop.

Given triples `(x, y_w, y_l)`, prompt plus preferred and rejected response,

```
L = -log sigmoid( beta * [ log pi_theta(y_w|x)/pi_ref(y_w|x)
                         - log pi_theta(y_l|x)/pi_ref(y_l|x) ] )
```

`pi_ref` is a frozen copy of the starting model and `beta` controls drift.
Plainly, raise the good answer and lower the bad one relative to where you
started. Offline, differentiable, closer to supervised learning than to RL.

This is why the hallucination field runs on DPO. Once you can produce a
hallucinated version of a correct answer you have a preference pair for free.

---

## 2. Why pivot at all

From `SRF_details.md`, three measured facts.

1. **The evidence map contributes 1 pair at the decoder and 6 through the
   encoder** (7.5). Random map is inert at 40.67 even with full suppression and
   foveation, while the real map goes 41.33 to 45.33. The decoder intervention,
   which is what the whole training-free literature does, is the weaker half.
2. **Full POPE regresses significantly.** Minus 0.68, McNemar p=5.0e-04, over
   9000 questions (12). The AND-gate fix moved it +0.03 at p=0.88, so the
   gate-precision hypothesis is refuted. A measured ceiling on inference-time
   attention surgery.
3. **Animals and Chess score exactly 0.0% under every method tested.**
   Attention reweighting cannot turn a 4 into a 5.

External hook. *Vision Language Models are Biased* measured **+21.09 points from
removing image backgrounds**. The background fires the prior. Attenuating it is
what the encoder stage does, and it has only ever been applied at inference, to
a model never trained to exploit it.

---

## 3. The one architectural idea

**Move the evidence estimator to training time.**

The current estimator is capped at 74.67% gate accuracy against a VLM baseline
of 86 to 88% (10.5). It is weak because it must be cheap, because it runs per
sample at inference.

Once it only constructs a training signal that constraint disappears and
inference becomes a single clean forward pass with no external model. Better
perception **and** a better efficiency story than the current pipeline, which
pays a CLIP pass VAF does not.

What becomes affordable as an annotator.

| Option | Note |
|---|---|
| **COCO instance masks** | free, already annotated, highest quality. The default for arm A below |
| Grounding DINO 1.6 / OWLv2 / Florence-2 / SAM2 | open-vocabulary regions where COCO has no annotation |
| A large VLM as region annotator | the GazeVLM recipe, which used Qwen3-VL-235B-A22B |
| Ensemble plus agreement filtering | GazeVLM kept 11,080 trajectories from 10 candidates per sample |
| SigLIP2 | cheap floor. **Recalibrate every threshold**, `_MODEL_ABSENCE_THRESH` is 0.20 for ViT-B/32 and 0.06 for SigLIP |

---

## 4. What the literature actually does

Numbers from the comparison in *On-Policy Data Hold the Key*
(https://arxiv.org/html/2501.09695v1) plus primary sources.

| Method | Venue | Backbone | Training data | N | Objective | Weights |
|---|---|---|---|---|---|---|
| LLaVA-RLHF | ICLR'24 | LLaVA-1.5 7B/13B | human prefs + fact-augmented RM | 122k | PPO | full |
| RLHF-V | CVPR'24 | LLaVA-1.5-13B | **human** segment-level corrections | **1.4k** | dense DPO | full |
| HA-DPO | arXiv'23 | LLaVA-1.5 7B/13B | GPT-4 rewritten COCO captions | 6k | DPO | LoRA, all linear |
| HSA-DPO | — | LLaVA-1.5-13B | fine-grained GPT-4/4V feedback | 8k | DPO | LoRA |
| mDPO | EMNLP'24 | LLaVA-1.5 | VLFeedback subset | 10k | **image-conditional** DPO | LoRA |
| RLAIF-V | CVPR'25 hl | LLaVA-1.5 7B/13B | self-generated AI feedback, 83k pool | 16k | iterative DPO | LoRA |
| POVID | ICLR'24 **workshop** | LLaVA-1.5 7B/13B | LLaVA-Instruct + GPT-4V errors + **image noise** | 17k | DPO | LoRA |
| HALVA | ICLR'25 | LLaVA-1.5 7B/13B | GPT-4V phrase level | 21.5k | contrastive tuning | LoRA |
| V-DPO | EMNLP'24 F | LLaVA-1.5 | vision-guided pairs | — | DPO | LoRA |
| Silkie | — | Qwen-VL-Chat | VLFeedback | 80k | DPO | LoRA |

Shared evaluation set, essentially universal in this literature:
**POPE adversarial** 3000, **AMBER** 1004 images, **Object HalBench** 300,
**MMHal-Bench** 96, plus MME / MMBench / SEED as no-regression controls.

### Five patterns that matter more than any row

1. **The backbone is LLaVA-1.5-7B/13B, near universally.** Use it and published
   numbers become free baselines. Largest single compute saving available.
2. **Training sets are tiny, 1.4k to 21.5k.** Not FoveateR's 765k.
3. **Training data is never the evaluation benchmark.** Always COCO,
   LLaVA-Instruct or VLFeedback derived.
4. **GRPO is absent from the hallucination literature. The field is DPO.**
   GRPO lives in the reasoning and perception line (FoveateR, GazeVLM, Faithful
   GRPO). An earlier draft of this doc framed GRPO as the thing to avoid. That
   was wrong, we were never in that lane.
5. **LoRA dominates.** Only the two earliest papers do full fine-tuning.

### How these papers evaluate

Standard **zero-shot** generation on the full benchmark, scored the way
`srf/eval.py` already does. No in-context examples, no few-shot prompting.
Qualitative cases appear in figures, never as evidence.

**AMBER and Object HalBench are in the standard set and are not in our harness.**
Adding them is table stakes.

---

## 5. Why FoveateR and GazeVLM are NOT in that table

1. Wrong backbone. Qwen2.5-VL-3B/7B and Qwen3-VL-4B against LLaVA-1.5.
2. **Neither reports a single hallucination or bias benchmark.** No POPE, AMBER,
   MMHal, ObjHal, MMVP or VLMBias. There is nothing to put in the cells.
3. Reproducing FoveateR is 765k SFT plus 70k GRPO steps. That is a project, not
   a baseline.
4. Different problem, token efficiency and high-resolution perception rather
   than language-prior resistance.

**The exception, which may be a contribution.** If either releases checkpoints,
*evaluating* them on POPE / AMBER / MMHal costs inference only. "First
evaluation of the learned-foveation line on hallucination benchmarks, and it
does not transfer" pre-empts the obvious reviewer question, does learned
cropping already solve this. No public code or weights found as of 2026-10-08.
FoveateR page https://juhongm999.github.io/FoveateR/ , worth watching.
**Name collision warning**, FoveaTer (ICLR 2023) is an unrelated
image-classification paper.

Their full setups, for the related-work section.

| | FoveateR | GazeVLM |
|---|---|---|
| Base | Qwen2.5-VL-Instruct 3B / 7B | Qwen3-VL-4B-Instruct, traces from Qwen3-VL-235B-A22B |
| Mechanism | `<fov>` token plus 2-layer MLP predicting a box, re-encode the crop | `<LOOK>` tags, attention-logit suppression outside the gazed region **during training only**, removed at deployment |
| SFT data | Visual CoT 438k, RefCOCO/+/g 321k, ScienceQA 6k = **765k** | GQA, ChartQA, PlotQA, InfoVQA filtered to **11,080** verified trajectories |
| RL | GRPO, accuracy + format + region-size penalty applied only when already correct, lambda_fov 1.0, lambda_reg 0.2 | GRPO on **4,453** calibrated samples, six-term reward |
| Evals | Visual CoT 12 datasets, V\* Bench | MathVista, ChartQA, MMBench, MMStar, CV-Bench, HRBench-4k/8k |
| Headline | ~307 visual tokens vs 1,152, ~9x latency cut | HRBench-4k 79.5 to 83.4, 8k 73.6 to 78.0 |

GazeVLM is also the **precedent that the internalisation mechanic works**, it
applies an attention bias during training, removes it at deployment, and the
model self-steers.

---

## 6. The training data, and whether it transfers

**None of the evaluation benchmarks have training splits.**

| Benchmark | Size | Built from | Train split |
|---|---|---|---|
| POPE | 9,000 | COCO **val2014**, 500 images, templated yes/no | no |
| MMVP | 150 pairs | CLIP-blind pairs | no, far too small |
| VLMBias | 2,784 | **synthetically generated** counterfactual images | no |
| MME | 2,374 | curated | no |
| MMHal-Bench | 96 | curated | no |

That is normal. Nobody trains on POPE. You train a **behaviour** on constructed
data and evaluate zero-shot.

### The concrete construction

Using COCO train2017 instance masks.

```
image      COCO train2017 photo containing a backpack
question   "Is there a backpack in the image?"      <- POPE's own template
answer     "Yes"
evidence   the backpack instance mask               <- free, from annotations

x_bg       same photo, backpack inpainted out, background UNTOUCHED
answer'    "No"                                     <- the label must flip
```

That pair is the whole idea. The background is **identical** across both views,
so every co-occurrence cue is preserved and only the object changed. A model
answering from the prior says yes to both. A model answering from evidence
flips. **We supervise the dependence, not the answer.**

Inpainting cost is real. LaMa or SD-inpaint over ~15k instances. Budget it.

### Why this transfers to POPE

**POPE's adversarial split samples negatives by co-occurrence statistics**,
objects that frequently appear alongside what is present. That is exactly a
language-prior attack, and exactly what the counterfactual pair teaches the
model to resist. POPE is COCO val2014, training is COCO train2017, so images
are disjoint while distribution and templates match. This is the textbook
transfer case and it is empirically why HA-DPO, POVID and RLAIF-V all improve
POPE while training on COCO-derived data.

**Verify image-ID disjointness explicitly, do not assume it.**

### Why it will probably NOT transfer to VLMBias

Different mechanism. VLMBias images are **synthetic counterfactuals**, a dog
with five legs, a four-stripe Adidas logo, and the prior being overridden is a
**memorised attribute**, not a co-occurrence. COCO contains no five-legged dogs.
Nothing in COCO-derived data teaches a model to recount legs against a
memorised four.

Our own measurements agree. Animals and Chess are 0.0% under every method, and
the VLMBias noun fix moved nothing (10.5). The model is retrieving, not
mis-attending.

**To move VLMBias, generate counterfactual training images with the benchmark's
own recipe on held-out object categories.** Train on modified animals, test on
modified chess boards.

### So the project splits in two

| Arm | Data | Likely to work | Field |
|---|---|---|---|
| **A. Hallucination** | COCO masks + inpainting, ~15k pairs | yes, known transfer path | crowded, POVID / mDPO / RLAIF-V |
| **B. Counterfactual bias** | generated counterfactual images, held-out categories | risky | **empty** |

**Run A first** as proof the objective works at all, since it has comparable
baselines and a known transfer path. **B is the differentiator** and the reason
the paper is interesting. A alone is incremental. B alone is unvalidated.
Together they are a paper.

---

## 7. Proposed method

### 7.1 Evidence estimation, training time only

Section 3. Per training sample, a partition into evidence region and
background. Quality matters in a way it never did before, because errors
propagate into the training signal rather than into one forward pass.

### 7.2 Evidence-counterfactual supervision, the novel component

**Hard constraint. Do not make this another decoder attention bias.** That was
worth 1 pair of 7 and every widening hurt.

Three views per sample, `x` original, `x_fg` evidence preserved and background
neutralised, `x_bg` evidence removed and background preserved.

**(a) Consistency and divergence.**
```
pull   p(y | x)     toward  p(y | x_fg)
push   p(y | x_bg)  away from the prior-supplied answer
```

**(b) Prior-weighted token loss.** Weight the LM loss on each answer token by
its evidence dependence, the gap between `p(y|x)` and `p(y|x_bg)`. Gradient
concentrates on tokens currently answered from memory.

**Free preference pairs.** `argmax p(y | x_bg)` is literally what the prior
says, so it is a rejected sample with no human annotation and no GPT-4 judge.
That gives an optional DPO second stage off the RL path entirely.

### 7.3 Internalisation

Inference is one forward pass, no external model, no extra decoding pass.
Precedent is GazeVLM.

**The falsifiable headline.** Does a model trained with the intervention beat
the same model with the intervention applied at inference, at zero inference
cost? If yes, that is the paper. If no, that is the ceiling result, and better
known in week two than month four.

### 7.4 The novel angle, one sentence

*We supervise evidence dependence rather than answer correctness, using
query-conditioned counterfactual views that are affordable at training time and
absent at inference.*

Three claims that hold up.

1. **Moving the estimator to training time inverts the training-free tradeoff.**
   It can be arbitrarily expensive while inference stays clean.
2. **The objective expresses something no inference-time method can.** "This
   answer should not survive without evidence" is a statement about a
   counterfactual forward pass. Attention steering cannot represent it, which is
   the principled reason weights are required and why the POPE regression
   happened.
3. **The evaluation lane is vacated.** The learned-foveation literature reports
   zero hallucination or bias numbers.

### 7.5 THE THREAT, read before committing

**POVID and mDPO are closer neighbours than this doc originally admitted.**

- **POVID** corrupts the image with noise to generate dispreferred responses.
- **mDPO** adds an image-conditional preference term precisely because standard
  DPO ignores the image.

Both already exploit "what the model says when vision is degraded". The
defensible difference rests on three things.

1. Their corruption is **global noise**. Ours is a **query-conditioned evidence
   partition**, a semantically meaningful foreground/background split with the
   background held fixed, not degradation.
2. They use the corrupted view to **build a response pair**. We use it to impose
   a **dependence constraint** plus token-level gradient weighting. A different
   objective, not a different dataset.
3. They evaluate object hallucination. We target **counting and counterfactual
   bias**, where every method including ours scores 0.0%, and where the +21pp
   background-removal finding supplies a mechanism nobody has exploited.

**Read POVID and mDPO in full before any data construction.** If those three
distinctions do not survive close reading, the project needs rethinking. This
is the highest-value task on the list.

Also cite and distinguish CF-VLM (minimally edited causal pairs) and VPPO
(sparse gradient masks on visually grounded tokens, but in RL and with no
evidence partition).

---

## 8. What to compare against

| Tier | Methods | Action |
|---|---|---|
| A. Inference-time | VAF, VCD, VHR, ILVAD, baseline | **run**, already in `srf/eval.py --method` |
| B. Internal controls | **vanilla SFT on identical data**, our method applied at inference, random-partition control, each loss term ablated | **run**, these decide the paper |
| C. Trained baselines | RLHF-V, POVID, HA-DPO, mDPO, RLAIF-V | **cite published** if backbone matches, run 1-2 otherwise |
| D. Learned foveation | FoveateR, GazeVLM | **cite only**, state non-comparability, evaluate checkpoints if released |

**Tier B is where reviewers attack.** Vanilla SFT on identical data is
non-negotiable, without it the result reads as "more data helps".

---

## 9. Evaluation plan

Standard set for this literature, **POPE adversarial, AMBER, Object HalBench,
MMHal-Bench**, plus **MMVP and VLMBias** as the differentiator, plus
**MMBench or MMStar** as a no-regression control. Post-training papers die on
catastrophic-forgetting objections.

Two decisions to lock before any training run.

1. **Held-out categories, not held-out samples**, for arm B. Train on modified
   animals, test on modified chess boards. Otherwise the model learns the
   counting task and the paper learns nothing about routing, which is exactly
   the objection *VLMs are Biased* raises when it argues the failure is
   retrieval from pretraining.
2. **Power.** 150 MMVP pairs cannot resolve a 5-point effect, measured at p=0.20
   with CI [-1.33, +12.67]. MMHal at 96 behaves the same way. POPE at 9000,
   AMBER at 1004 and MME at 2374 can carry a claim. Decide which benchmarks the
   story rests on before running. Reuse `srf/significance.py`, paired bootstrap
   plus McNemar over `--save_records`, no GPU needed.

---

## 10. Staged plan with kill criteria

| Stage | What | Kill criterion |
|---|---|---|
| 0 | **Read POVID and mDPO in full.** Write the one-paragraph distinction | cannot articulate a distinction a reviewer would accept |
| 1 | Annotator bake-off on the POPE gate task, current ceiling 74.67% against a VLM at 86-88% | nothing clears the VLM baseline, so the map was never the bottleneck |
| 2 | Arm A data construction, COCO masks plus inpainting, ~15k counterfactual pairs | fewer than a few thousand survive quality filtering |
| 3 | **Vanilla SFT on that data, no new objective.** The Tier B control | SFT alone closes most of the gap, so the objective is not the contribution |
| 4 | Full objective, arm A. Headline comparison against the inference-time version | trained model does not beat inference-time steering, report the ceiling and stop |
| 5 | Ablations, random partition, each loss term | random partition matches the real one, the most damaging outcome, check early |
| 6 | Arm B, counterfactual image generation, held-out categories | generation pipeline cannot produce convincing counterfactuals |
| 7 | Second architecture, significance testing, no-regression control | — |

Stages 0, 1 and 3 can falsify the project in about two weeks and cost almost no
GPU time. Do them before anything else.

---

## 11. Open decisions

1. **Backbone.** LLaVA-1.5-7B gives free published baselines and matches the
   whole comparison literature, and the `srf-llava` branch already exists.
   Qwen2.5-VL-3B keeps continuity with every number in `SRF_details.md`.
   **Leaning LLaVA-1.5-7B**, the free baselines outweigh the continuity.
2. **LoRA or full fine-tune.** LoRA is what the field does and reviewers accept
   it. Decide after stage 3.
3. **What "background neutralised" and "evidence removed" mean concretely.**
   Inpaint, grey fill, blur, or shuffle. Inpainting is the principled choice for
   `x_bg` because it preserves scene plausibility, but it is the most expensive
   and can hallucinate content. **Resolve before stage 2**, it determines the
   entire dataset.
4. Whether all three views are needed at train time, since three views triples
   the forward passes. Check whether `x_bg` alone suffices.
5. Whether arm B generation is feasible at all. Scope it in stage 6, not before.
6. Final method name.

---

## 12. Reading list

**Read first, these decide the project.**
- POVID, https://arxiv.org/pdf/2402.11411 (ICLR'24 workshop)
- mDPO, https://aclanthology.org/2024.emnlp-main.460/ (EMNLP'24)

Closest methodological neighbours.
- Foveated Reasoning / FoveateR, https://arxiv.org/html/2604.21079
- GazeVLM, https://arxiv.org/abs/2605.07817

Motivating evidence.
- Vision Language Models are Biased, https://arxiv.org/html/2505.23941v4
- Training on Foveated Images Improves Robustness (R-Blur), https://arxiv.org/pdf/2308.00854

Baselines and background.
- DPO, Rafailov et al., NeurIPS 2023
- RLHF-V, CVPR'24, https://github.com/RLHF-V/RLHF-V
- RLAIF-V, CVPR'25 highlight, https://github.com/RLHF-V/RLAIF-V
- HA-DPO, https://github.com/opendatalab/HA-DPO
- HALVA, ICLR'25, https://arxiv.org/abs/2405.18654
- On-Policy Data Hold the Key, https://arxiv.org/html/2501.09695v1

Saturated, cite as related work, do not build on.
- Training-free attention steering 2026. AGE / Imitating the Truth (ICLR'26),
  CAST, FADE, ACG, AdaVBoost, AdaIAT, CAI
- GRPO / RLVR perception rewards. Faithful GRPO, PEARL, Vision-SR1, SIVA-RL

---

## 13. Repo notes for the Snellius session

- Branch `autoresearch/mmvp-srf`. Upstream `main` is untouched lmms-eval.
- **`origin/srf-llava` exists** with the LLaVA dispatch in `eval.py::load_model`,
  `my_analysis/llava_attn_patch.py`, `run_mmhalbench` and
  `srf/score_mmhalbench.py`. It does **not** have the VTAR head selection. See
  `BRANCH_DIFF.md`. Merging the branches is a prerequisite for any LLaVA run,
  and decision 11.1 makes that likely.
- **Two copies of `SRF_details.md` exist.** Canonical is
  `lmms-eval/SRF_details.md`, in git. A stale copy at
  `/volumes2/mllm/SRF_details.md` is not version controlled and will not be on
  the cluster. The `load-mllm` skill points at the stale one. They differ by ten
  lines in section 11 about figure scripts.
- **AMBER and Object HalBench have no loader in `srf/`.** Adding them is a
  prerequisite for Tier C comparison.
- `attn_implementation="eager"` is mandatory wherever the attention patch is
  used. SDPA and flash never call `torch.nn.functional.softmax`, so the
  intervention silently becomes a no-op.
- `set -o pipefail` on every piped command, `tee` hides crashes.
- Every run command ends with `2>&1 | tee /tmp/<name>.log`.
