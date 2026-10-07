# Follow-up project: post-training for visual evidence grounding

> Scoping document written 2026-10-07 for a new paper, separate from the SRF
> ICLR submission. Read `SRF_details.md` first for what was already measured.
> This file is the plan, the literature position, and the open decisions.
> Nothing here has been executed.

**Target machine.** Snellius, A100 and H100. Full fine-tuning and GRPO are both
affordable, which was not true on the local box (one usable GPU, 11 GB, the
second card is a GTX 1080 Ti at sm_61 and unusable).

---

## 0. Term hygiene, read before writing anything

This is a new paper for a new venue. **None of the SRF vocabulary carries over.**

| Retire | Reason |
|---|---|
| SRF, Semantic Re-Focus | the prior paper's name |
| semantic foveation, foveation | the prior paper's component, and also the name the FoveateR and GazeVLM line already owns |
| semantic relevance map | prior paper's term |
| attention re-focus, re-focusing | prior paper's term |
| vision-responsive heads, VTAR | prior paper's term |
| `lambda_sem`, `lambda_bg`, `lambda_sys` | prior paper's notation |

Build the new vocabulary around **visual evidence**. Evidence region, evidence
grounding, evidence-counterfactual supervision, prior suppression,
evidence dependence. Pick the final name late, after the method settles.

**Citation decision to make early.** If the SRF submission is live or published,
this paper must cite it as related work, in the third person under anonymous
review. The relevance-estimation step genuinely overlaps. Decide how much of the
estimator is presented as new before writing, because it changes the framing of
Section 3.

---

## 1. Why pivot at all

From `SRF_details.md`, three measured facts that together say the training-free
route is exhausted for this problem.

1. **The relevance map contributes 1 pair at the decoder and 6 through the
   encoder** (section 7.5). Random map is inert at 40.67 even with full
   suppression and foveation, while the semantic map goes 41.33 to 45.33. The
   decoder intervention, which is what the entire 2026 training-free literature
   is doing, is the weaker half.
2. **Full POPE regresses significantly.** Minus 0.68, McNemar p=5.0e-04, over
   9000 questions (section 12). The AND-gate fix moved it by +0.03 at p=0.88, so
   the precision hypothesis is refuted. This is a ceiling on inference-time
   attention surgery, measured with enough samples to be believed.
3. **Animals and Chess score exactly 0.0% under every method tested**, including
   every published baseline. Attention reweighting cannot turn a 4 into a 5.

External hook that makes (3) actionable. *Vision Language Models are Biased*
measured **+21.09 points from removing image backgrounds**. The background is
what fires the prior. Attenuating the background is exactly what the encoder
stage does, and it has only ever been applied at inference, to a model that was
never trained to exploit it.

---

## 2. The one architectural idea

**Move the relevance estimator to training time.**

The current estimator is measurably weak. POPE gate accuracy is 64.0% shipped and
74.67% at its tuned ceiling, against a VLM baseline of 86 to 88% (section 10.5).
It is weak because it has to be cheap, because it runs per sample at inference.

Once the estimator only constructs a training signal, that constraint is gone and
inference becomes a single clean forward pass with no external model. That is
simultaneously a better perception story and a better efficiency story than the
current one, which pays a CLIP pass that VAF does not.

What becomes affordable as an annotator:

| Option | Note |
|---|---|
| Grounding DINO 1.6 / OWLv2 / Florence-2 + SAM2 | real object masks instead of an 83-crop similarity heatmap. Qualitatively different, not an incremental swap |
| A large VLM as region annotator | the GazeVLM recipe, which used Qwen3-VL-235B-A22B to synthesise traces |
| Ensemble plus agreement filtering | generate k candidates, keep only those where annotators agree. GazeVLM kept 11,080 of a much larger pool |
| SigLIP2 | cheap floor. **Recalibrate every threshold.** `_MODEL_ABSENCE_THRESH` is 0.20 for ViT-B/32 and 0.06 for SigLIP, a 3x difference for the same task |

---

## 3. Proposed method, three components

### 3.1 High-perception evidence estimation (training time only)

As section 2. Produces, per training sample, a partition of the image into
evidence region and background. Quality matters here in a way it never did
before, because errors propagate into the training signal rather than into one
forward pass.

### 3.2 Evidence-counterfactual supervision, the novel component

**Hard constraint. Do not make this another decoder attention bias.** That was
worth 1 pair of 7 and every attempt to widen it hurt. Repeating it under a new
name is the version a reviewer catches.

Build three views per sample from the partition.

```
x      original image
x_fg   evidence preserved, background neutralised
x_bg   evidence removed, background preserved
```

Train on two things at once.

**(a) Consistency and divergence.** The answer must be recoverable from the
evidence alone and must not survive without it.

```
pull   p(y | x)     toward  p(y | x_fg)
push   p(y | x_bg)  away from the prior-supplied answer
```

**(b) Prior-weighted token loss.** Weight the LM loss on each answer token by its
current evidence dependence, measured as the gap between `p(y|x)` and `p(y|x_bg)`.
Gradient then concentrates on exactly the tokens the model is answering from
memory rather than from the image.

Why this component and not the alternatives.

- It is a **loss**, not an input transform and not an inference-time edit.
  Structurally different from foveation, from crop-and-zoom, and from the whole
  saturated training-free pile.
- It has a hard empirical hook, the +21.09 points from background removal, and it
  targets the categories where every existing method scores 0.0%.
- It explains why weights are required at all. Inference-time steering cannot
  express "this answer should not survive without evidence", because that is a
  statement about a counterfactual forward pass.

**Must cite and distinguish from.** Image-DPO (image corruption for preference
pairs), CF-VLM (minimally edited pairs for causal decision points), VPPO (sparse
gradient masks on visually grounded tokens, but in RL and with no relevance
partition). The distinction is that the counterfactual is query-conditioned and
evidence-partitioned, and that the target is the language prior rather than
general robustness. **State this in the abstract** or the paper reads as CF-VLM
with a saliency mask.

Alternatives considered and rejected. Token-budget reallocation (FAVE and FocusUI
have it), self-predicted relevance maps (Self-Grounded Attention and Self-Saliency
are close), evidence-calibrated abstention (clean, but it becomes a
trustworthiness paper rather than a grounding paper).

### 3.3 Internalisation

Inference is one forward pass, no external model, no extra decoding pass. The
precedent that this works is GazeVLM, which applies attention suppression during
training, removes it at deployment, and reports the model self-steers and retains
most of the effect.

**The falsifiable headline.** Does a model trained with the intervention beat the
same model with the intervention applied at inference, at zero inference cost? If
yes, that is the paper. If no, that is the ceiling result and worth knowing in
week two rather than month four.

---

## 4. Why everyone uses GRPO, and why this project should not

**Why they use it.** The action in FoveateR and GazeVLM is *discrete and
non-differentiable*. You cannot backpropagate through "which box to crop". So they
need either a policy gradient or pseudo-labelled supervised box regression, and
both papers do both, coldstart SFT on pseudo-labels followed by GRPO to refine.
GRPO specifically because it is critic-free, needing no value network, and because
these tasks have verifiable answers so RLVR applies with no learned reward model.

**Why this project does not need it.** The proposed intervention is continuous and
differentiable end to end. A relevance-weighted soft blur is differentiable. A
counterfactual consistency loss is a standard supervised objective. There is no
discrete action to select, so there is nothing for a policy gradient to do that a
gradient cannot.

**Make this a claim, not a concession.** "We match or exceed RL-trained
active-vision methods using only supervised objectives, at a fraction of the
training cost and with deterministic reproducibility." That is a real selling
point in a field where every neighbouring paper needs a GRPO run.

**The honest caveat, have the answer ready.** GRPO optimises the end-task metric
directly. Supervised training optimises a proxy, token cross-entropy plus the
consistency terms. If proxy and metric diverge you lose, and a reviewer will ask
"why not RL". The answer is differentiability, cost, and reproducibility, in that
order. Mitigate by keeping the counterfactual objective as close to the evaluation
metric as the format allows.

---

## 5. What to compare against

Four tiers. Tier B is the one that decides whether the paper is accepted.

### Tier A, no weight update, already implemented in `srf/`

`baseline`, `vaf` (ClearSight), `vcd`, `vhr`, `ilvad`, plus the renamed
training-free method from the prior work. All run through `srf/eval.py --method`.
These are the floor. Running them again costs nothing because the harness exists.

Optionally add a 2026 training-free method for currency. CAI and AdaVBoost are the
closest in spirit.

### Tier B, the decisive internal baselines

| Baseline | What it isolates |
|---|---|
| **Vanilla SFT on identical data** | the objective, separated from the data. Reviewers will demand exactly this. Without it the result is "more data helps" |
| **Training-free intervention at inference** | whether training beats steering. This is the headline comparison |
| Full method minus the counterfactual term | whether the novel component carries its weight |
| Full method with a random evidence partition | the random-map control, which was decisive in the prior ablation (inert at 40.67) |

### Tier C, post-training methods that are not GRPO

The expected comparison class, all offline preference optimisation rather than
online RL. HA-DPO, POVID, V-DPO, mDPO, Image-DPO, RLHF-V, CF-VLM.
Pick two or three, not all. Prefer the ones with public checkpoints for the chosen
backbone.

### Tier D, cite but do not run

FoveateR and GazeVLM. Different backbones, different benchmarks, and **neither
evaluates a single hallucination or bias benchmark**. State the non-comparability
explicitly rather than leaving the reader to wonder.

---

## 6. The two closest competitors, full setups

| | **FoveateR** (Foveated Reasoning) | **GazeVLM** |
|---|---|---|
| Base model | Qwen2.5-VL-Instruct 3B and 7B | Qwen3-VL-4B-Instruct, traces from Qwen3-VL-235B-A22B, also Qwen3.5-4B-Think |
| Mechanism | `<fov>` trigger token plus a 2-layer MLP predicting a continuous box, then re-encode that crop | `<LOOK>` tags with coordinates, attention-logit suppression outside the gazed region **during training only**, removed at deployment |
| SFT data | Visual CoT 438K, RefCOCO/+/g 321K, ScienceQA 6K. **765K total** | GQA, ChartQA, PlotQA, InfoVQA, filtered to **11,080 verified trajectories** from 10 candidates per sample |
| RL | GRPO on the same 765K. Accuracy plus format plus a region-size penalty applied only when already correct. `lambda_fov`=1.0, `lambda_reg`=0.2 | GRPO on **4,453 difficulty-calibrated samples**. Six-term reward, correctness +1.5 with gaze and +0.7 without, format +0.15, bbox validity +0.10, overlap penalty -0.15 times IoU, more than 10 looks -0.15, length -0.05 |
| Evals | Visual CoT 12 datasets (DocVQA, TextCaps, TextVQA, DUDE, SROIE, InfoVQA, Flickr30k, Visual7W, GQA, OpenImages, VSR, CUB) plus V\* Bench | MathVista, ChartQA, MMBench, MMStar, CV-Bench, HRBench-4k, HRBench-8k |
| Headline | 7B beats prior 7B on Visual CoT at ~307 visual tokens against 1,152. 3B beats 7-12B on V\*. ~9x latency reduction | HRBench-4k 79.5 to 83.4, HRBench-8k 73.6 to 78.0, MathVista +1.6 |
| **Hallucination or bias benchmarks** | **none** | **none** |

**That last row is the territory claim.** The entire learned-foveation line is an
efficiency and high-resolution-perception literature. Owning language-prior bias
and hallucination does not require fighting them on token budgets or V\* Bench.

**Second observation.** GazeVLM trains on 11K SFT plus 4.4K RL samples. That is
small. On A100 and H100 this project is not compute-bound at that scale, it is
data-construction bound. Budget accordingly.

---

## 7. Evaluation plan

**Claim the lane the competitors vacated.** POPE, MMVP, VLMBias, MMHal-Bench, MME.

**Add a no-regression control.** MMBench or MMStar. Post-training papers die on
catastrophic-forgetting objections and the column needs to exist before a reviewer
asks for it.

Two decisions to lock before any training run.

1. **Held-out categories, not held-out samples.** Train on Animals, test on Chess.
   Otherwise the model learns the counting task and the paper learns nothing about
   routing, which is precisely the objection *VLMs are Biased* sets up when it
   argues the failure is retrieval from pretraining.
2. **Power.** 150 MMVP pairs cannot resolve a 5-point effect, already measured at
   p=0.20 with CI [-1.33, +12.67]. MMHal is 96 samples and will behave the same
   way. POPE at 9000 and MME at 2374 can carry a claim. Decide which benchmarks
   the story rests on before running, and reuse `srf/significance.py`, which does
   paired bootstrap and McNemar over `--save_records` output and needs no GPU.

---

## 8. Staged plan with kill criteria

| Stage | What | Kill criterion |
|---|---|---|
| 0 | Annotator bake-off. Compare Grounding DINO / OWLv2 / SAM2 / SigLIP2 / large-VLM evidence partitions against the current CLIP pipeline on the POPE gate task, where the current ceiling is 74.67% against a VLM at 86 to 88% | no annotator clears the VLM baseline, in which case the whole premise that the map was the bottleneck is wrong |
| 1 | Data construction. Build the three-view counterfactual set. Agreement-filter | fewer than a few thousand samples survive filtering |
| 2 | Vanilla SFT on the data, no new objective. The Tier B control | SFT alone closes most of the gap, in which case the objective is not the contribution and the paper is a dataset paper |
| 3 | Full objective. The headline comparison against the training-free intervention at inference | trained model does not beat inference-time steering. Report as the ceiling result and stop |
| 4 | Ablations, random partition, no-counterfactual, no-prior-weighting | random partition matches the real one, which is the single most damaging outcome and must be checked early |
| 5 | Second architecture, significance testing, no-regression control | — |

Run stage 0 and stage 2 before committing to anything else. Between them they can
falsify the project in roughly a week.

---

## 9. Open decisions

1. **Backbone.** Qwen2.5-VL-3B keeps continuity with every number in
   `SRF_details.md` and the whole `srf/` harness works on it. Qwen3-VL-4B matches
   GazeVLM. Picking the former means the prior results are free baselines.
2. **Full fine-tune or LoRA.** A100 80GB makes full FT of a 3B feasible with
   FSDP or ZeRO. LoRA is cheaper and reviewers accept it, but full FT removes an
   objection about expressivity. Decide after stage 2.
3. **What "background neutralised" means concretely.** Blur, grey fill, inpaint,
   or shuffle. Each has a different confound. Blur keeps the prior partly alive,
   grey fill creates an out-of-distribution image, inpaint is expensive and can
   hallucinate, shuffle destroys spatial structure. The prior work already
   measured that patch shuffling costs more than blurring on POPE, which is an
   argument for blur, but blur is also the component being replaced. **Resolve
   this before stage 1**, it determines the entire dataset.
4. **Whether `x_fg` or `x_bg` or both are used at train time.** Three views
   triples the forward passes. Check whether `x_bg` alone suffices.
5. Final method name.

---

## 10. Reading list

Closest competitors, read in full before writing a proposal.
- Foveated Reasoning / FoveateR, https://arxiv.org/html/2604.21079
- GazeVLM, Active Vision via Internal Attention Control, https://arxiv.org/abs/2605.07817

Motivating evidence.
- Vision Language Models are Biased, https://arxiv.org/html/2505.23941v4
- Training on Foveated Images Improves Robustness to Adversarial Attacks (R-Blur),
  https://arxiv.org/pdf/2308.00854

Must distinguish from.
- CF-VLM, counterfactual vision-language fine-tuning
- Image-DPO, self-generated VQA pairs plus image corruption
- VPPO, sparse gradient masks on visually grounded tokens
- JoLA, per-head learned gates, https://pith.science/paper/2502.01179

Saturated areas, cite as related work, do not build on.
- Training-free attention steering 2026. AGE / Imitating the Truth (ICLR'26)
  https://proceedings.iclr.cc/paper_files/paper/2026/file/48d467d310502791a97d05d1631c5b0f-Paper-Conference.pdf ,
  CAST https://arxiv.org/pdf/2605.04641 , FADE https://arxiv.org/pdf/2606.29431 ,
  ACG (CVPR'26), AdaVBoost https://arxiv.org/pdf/2602.13600 ,
  AdaIAT https://arxiv.org/html/2603.04908 ,
  CAI https://arxiv.org/html/2606.29847
- DPO family for hallucination. RLHF-V, V-DPO, HA-DPO, POVID, RLAIF-V, mDPO, P2-DPO
- GRPO / RLVR with perception rewards. Faithful GRPO
  https://arxiv.org/pdf/2604.08476 , PEARL (CVPR'26), perception-centric PRMs
  (CVPR'26), Vision-SR1 (ICLR'26), SIVA-RL

---

## 11. Repo notes for the Snellius session

- Branch for this work: `autoresearch/mmvp-srf`. Upstream `main` is untouched
  lmms-eval.
- **`origin/srf-llava` exists** and carries the LLaVA dispatch in
  `eval.py::load_model`, `my_analysis/llava_attn_patch.py`, `run_mmhalbench` and
  `srf/score_mmhalbench.py`. It does **not** have the VTAR head selection. See
  `BRANCH_DIFF.md`. Merging the two branches is a prerequisite for any LLaVA run.
- **Two copies of `SRF_details.md` exist.** The canonical one is
  `lmms-eval/SRF_details.md`, in git. A stale copy sits at `/volumes2/mllm/SRF_details.md`
  and is not version controlled, so it will not be on the cluster. The
  `load-mllm` skill points at the stale one. They differ by ten lines in section
  11 about figure scripts.
- `attn_implementation="eager"` is mandatory on any model the attention patch
  touches. SDPA and flash attention never call `torch.nn.functional.softmax`, so
  the intervention silently becomes a no-op.
- `set -o pipefail` on every piped command. `tee` hides crashes.
- Every run command should end with `2>&1 | tee /tmp/<name>.log`.
