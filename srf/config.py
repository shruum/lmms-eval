"""
Central config — single source of truth for paths, model IDs, and SRF hyperparameters.
All eval scripts import from here; nothing is hardcoded in eval scripts.

═══════════════════════════════════════════════════════════════════════════════
QUICK-START: BEST CONFIG (Qwen2.5-VL-3B-Instruct, as of 2026-06-16)
═══════════════════════════════════════════════════════════════════════════════

  Method: SRF-E (contrastive), gamma=3.0
  saliency_mode   = clip_full_gate_v3   ← already the default in SRF_ARCH_PARAMS
  layer_start     = 8                   ← tuned by coordinate-descent sweep
  layer_end       = dataset-specific (MMVP=16, POPE=12, VLMBias=14)
  head_top_k_pct  = 0.20
  alpha           = 2.0 (MMVP/POPE), 8.0 (VLMBias)
  eps             = 0.2 (MMVP/POPE), 0.5 (VLMBias)
  phase           = generation (POPE/VLMBias), both (MMVP)
  sys_beta        = 0.30
  bias_mode       = additive_logit     ← default; see "Boosting methods" below
  gamma           = 3.0                ← SRF-E contrastive amplification

Results (Qwen2.5-VL-3B):
  MMVP pair_acc = 49.33%  (baseline 40.0%,  +9.33pp)
  POPE adv_acc  = 87.33%  (baseline 86.37%, +0.96pp)
  VLMBias       = 19.0%   (baseline 19.04%, ≈0pp — SRF-E broken for multi-token)

NOTE: SRF-E (gamma>0) only works for single/short-token answers.
      For VLMBias (multi-token {Yes}/{No} format), use SRF base (gamma=0).

Full POPE:   conda run -n mllm python srf/eval.py --method srfe --datasets pope --output results/ --gamma 3.0
Quick val:   conda run -n mllm python srf/eval_pope_val.py --srf --out results/pope_val_srf.json

═══════════════════════════════════════════════════════════════════════════════
SALIENCY MODES  (--saliency_mode / SRF_ARCH_PARAMS["saliency_mode"])
═══════════════════════════════════════════════════════════════════════════════

  clip_full_gate_v3  [DEFAULT, TUNED]
      Full-image CLIP similarity gate (threshold 0.21).  Single GPU block.
      Confidence-capped: alpha * min(sim/thresh, 1.0).
      Best on POPE val: acc=0.900, FPR=0.000.

  clip               [BASIC]
      Patch-level max_sim gate (threshold = clip_fallback_thresh, default 0.20).
      Older method; weaker than v3 on POPE.

  hssa               [EXPERIMENTAL]
      Hidden-state semantic alignment. No CLIP needed.

  lta                [EXPERIMENTAL]
      Last-token attention saliency.

  clip_lta           [EXPERIMENTAL]
      Weighted combination of CLIP + LTA (clip_weight + lta_weight must sum to 1).

  clip_full_gate_v3_iou / _adaptive / _ramp / _dynhead / _dynlayer
      [EXPERIMENTAL variants of v3 — not validated on full POPE]

  srf2               [EXPERIMENTAL]
      0.7 * GradCAM + 0.3 * HSSA ensemble.

═══════════════════════════════════════════════════════════════════════════════
BOOSTING METHODS  (--bias_mode / SRF_DEFAULTS["bias_mode"])
═══════════════════════════════════════════════════════════════════════════════

  additive_logit   [DEFAULT, BEST]
      Adds alpha to attention logits of salient tokens before softmax.
      Subtracts eps from background image tokens.

  budget_shift     [B3, EXPERIMENTAL — no gain on POPE]
      Post-softmax: redistributes within the image token budget only.
      Text attention is NOT changed. Fixes HallusionBench regression in theory,
      but zero gain on POPE val set.

  prob_interp      [EXPERIMENTAL]
      Interpolates between uniform and boosted attention distributions.
      interp_lambda controls mixing weight.

  prob_scale       [EXPERIMENTAL]
      Scales salient token probabilities post-softmax (unnormalized).

  Visual Reliance Compensation [B1] — use with any bias_mode:
      --vr_target 0.15 --vr_k 3.0
      Measures actual image attention fraction per generation step.
      If fraction < vr_target, scales alpha up: alpha *= (1 + k * deficit).
      Recovers ~3 FNs on POPE val, same ceiling as B4.

  Two-pass retry [B4] — eval_pope_val.py only:
      --b4 --b4_threshold 0.25 --b4_multiplier 2.0
      Generates first token; if "No" and CLIP sim >= threshold, re-runs with
      alpha * multiplier. Same ceiling as B1; no additive benefit when combined.

  Absent suppression — off by default:
      --neg_absent_alpha 2.0
      Applies -neg_absent_alpha logit when CLIP says object absent.
      Not yet validated; see Open Tasks in RESEARCH_STATUS.md.

═══════════════════════════════════════════════════════════════════════════════
ARCH PARAMS  (SRF_ARCH_PARAMS — vary with model size, must be tuned per model)
═══════════════════════════════════════════════════════════════════════════════

  layer_start / layer_end   vision-language fusion zone (tuned by sweep_heads_layers.py)
  dataset_layer_end         per-dataset override for layer_end
  head_top_k_pct            fraction of heads calibrated as vision-aware (default 0.20)
  clip_coarse_grid          CLIP patch grid: 7 for Qwen/448px, 6 for LLaVA/336px
  clip_top_k_pct            fraction of image tokens boosted (default 0.30)
  clip_fallback_thresh      absence gate for basic "clip" mode only (v3 uses 0.21 hardcoded)
  saliency_mode             see SALIENCY MODES above

Models: Qwen2.5-VL-3B TUNED | Qwen2.5-VL-7B NOT TUNED | Qwen-VL-Chat IN PROGRESS
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HF_HOME      = os.environ.get("HF_HOME", "/volumes2/hugging_face_cache")
MMVP_CSV     = os.path.join(HF_HOME, "mmvp_questions/Questions.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Model ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_PIXELS = 512 * 28 * 28
IMAGE_TOKEN        = "<|image_pad|>"   # Qwen default; LLaVA uses model.config.image_token_index

# ── Dataset: full sizes ────────────────────────────────────────────────────────
# POPE — run all three splits by default (adversarial + popular + random = 9000 total)
POPE_SPLITS        = ["adversarial", "popular", "random"]
POPE_N_FULL        = -1    # -1 = all available samples across selected splits
POPE_SEED          = 42

VLM_BIAS_CATEGORIES = [
    "Animals", "Chess Pieces", "Flags", "Game Boards",
    "Logos", "Optical Illusion", "Patterned Grid",
]
VLM_BIAS_N_PER_CAT  = None   # None = use all available per category
VLM_BIAS_SEED       = 42

MMVP_GT_CORRECTIONS = {99: "A", 279: "A"}   # known GT errors in original CSV

# ── VLind-Bench ────────────────────────────────────────────────────────────────
VLIND_BENCH_REPO_ID     = "klee972/VLind-Bench"
VLIND_BENCH_VOTE_THRESH = 2          # min human votes for an image to be used
VLIND_BENCH_CONCEPTS    = [
    "climate", "color", "diet", "folklore", "habitat",
    "history", "landmark", "location", "size", "time", "weight",
]

# ── SRF shared defaults (not arch-specific) ────────────────────────────────────
# These do NOT vary with model size. Arch-specific tunables live in SRF_ARCH_PARAMS.
SRF_DEFAULTS = {
    "sys_beta":         0.30,   # system-prompt attention suppression
    "text_beta":        0.0,    # text-token suppression (disabled)
    "text_layer_start": 20,     # text suppression zone (Qwen 3B proportions)
    "text_layer_end":   27,
    "bias_mode":        "additive_logit",
    "interp_lambda":    1.0,
    "prob_floor":       0.005,
    "img_scale":        1.5,
    "calib_n":          20,     # calibration samples for head identification
    "calib_seed":       0,
}

# ── SRF per-dataset params ─────────────────────────────────────────────────────
# Tunable per dataset, arch-agnostic.
# layer_start / layer_end live in SRF_ARCH_PARAMS (they scale with model depth).
SRF_DATASET_PARAMS = {
    # alpha tuned by coordinate-descent sweep (autoresearch_mmvp_v2, 2026-06-16)
    # MMVP: α=2.0 + γ=3.0 → 49.33% pair_acc (+9.33pp)
    # POPE: α=2.0 or α=4.0 both give 87.33%; use 2.0 for unified config
    "mmvp":        {"phase": "both",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
    "pope":        {"phase": "both",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 2.0},
    "vlmbias":     {"phase": "generation", "alpha": 8.0, "eps": 0.5, "neg_absent_alpha": 0.0},
    "mme":         {"phase": "generation", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
    "mmbench":     {"phase": "generation", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
    "hallusionbench": {"phase": "generation", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
    # VLind-Bench: counterfactual visual reasoning — same structure as MMVP (True/False pair)
    # phase="both": boost helps in both prefill (question understanding) and generation
    "vlind":       {"phase": "both",       "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
    # MMHal-Bench: open-ended image description (96 samples, 8 question types)
    # generation phase only — free-form answers generated token-by-token
    "mmhalbench":  {"phase": "generation", "alpha": 2.0, "eps": 0.2, "neg_absent_alpha": 0.0},
}

# ── SRF-E (evidence amplification) defaults ────────────────────────────────────
SRFE_DEFAULT_GAMMA = 3.0    # tuned: best on both MMVP (49.33%) and POPE (87.33%)
SRFE_GAMMA_SWEEP   = [1.0, 2.0, 3.0, 4.0]

# ── Architecture-specific hyperparameters ──────────────────────────────────────
# These all VARY with model size/architecture and must be tuned per model.
#
# layer_start / layer_end: vision-language fusion zone.
#   Scales with depth: use ~(8/28)*n_layers and ~(15/28)*n_layers as starting points.
#   dataset_layer_end: per-dataset fine-tuning (overrides the shared layer_end).
#
# head_top_k_pct: fraction of heads selected as vision-aware.
#   Typically 0.20 works across models; re-tune if accuracy drops.
#
# clip_coarse_grid: CLIP patch grid size. Larger = finer spatial resolution.
#   Adjust if image resolution changes significantly (e.g. LLaVA uses 336px vs Qwen's 448px).
#
# clip_top_k_pct: fraction of image tokens boosted by CLIP saliency.
#   Lower = more focused; higher = more context.
#
# clip_fallback_thresh: below this CLIP max_sim, object is likely absent → uniform boost.
#   0.20 is a reasonable default; lower for datasets with unusual visual content.
#
# image_token: token string used to find image token range in input_ids.
#   None = use model.config.image_token_index (LLaVA-style).
#
# All values marked "NOT tuned" are proportional starting points — sweep before paper.

SRF_ARCH_PARAMS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "n_layers":             28,
        "spatial_merge_size":   2,
        "image_token":          "<|image_pad|>",
        # ── tuned by coordinate-descent sweep (autoresearch_mmvp_v2, 2026-06-16) ──
        # ls=8 le=16 α=2.0 γ=3.0 → MMVP pair_acc=49.33% (+9.33pp vs 40% baseline)
        # ls=8 le=12 α=2.0 γ=3.0 → POPE adv=87.33% (+0.96pp vs 86.37% baseline)
        "layer_start":          8,
        "layer_end":            12,    # POPE-tuned default; dataset_layer_end overrides per dataset
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     7,
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        "clip_patch_thresh":    0.27,
        # saliency mode: "clip_full_gate_v3" (best, tuned) | "clip" (basic) | "hssa"
        "saliency_mode":        "clip_full_gate_v3",
        "clip_model":           "openai/clip-vit-base-patch32",   # swap to SigLIP to improve
        "hssa_layer_idx":       12,     # middle of 28-layer model for HSSA
        # CLIP saliency method: "clip_patch" | "clip_gradcam" | "srf2"
        "clip_saliency_method": "clip_patch",   # default = no behavior change
        "srf2_clip_weight":     0.7,
        "srf2_hssa_weight":     0.3,
        # LTA (last-token attention) params
        "lta_layer_idx":        -1,     # -1 = last decoder layer
        "lta_weight":           0.6,    # weight in clip_lta combined mode
        "clip_weight":          0.4,    # weight in clip_lta combined mode
        # per-dataset layer_end fine-tuning (overrides layer_end above)
        "dataset_layer_end":    {"mmvp": 16, "pope": 12, "vlmbias": 14, "mme": 16, "vlind": 16, "mmhalbench": 16},
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "n_layers":             32,
        "spatial_merge_size":   2,
        "image_token":          "<|image_pad|>",
        # ── NOT tuned — proportionally scaled from 3B ──
        "layer_start":          9,    # round(8/28 * 32)
        "layer_end":            17,   # round(15/28 * 32)
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     7,
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        "clip_patch_thresh":    0.27,
        "saliency_mode":        "clip",
        "clip_model":           "openai/clip-vit-base-patch32",
        "hssa_layer_idx":       16,     # ~middle of 32-layer model
        "clip_saliency_method": "clip_patch",
        "srf2_clip_weight":     0.7,
        "srf2_hssa_weight":     0.3,
        "lta_layer_idx":        -1,
        "lta_weight":           0.6,
        "clip_weight":          0.4,
        "dataset_layer_end":    {"mmvp": 17, "pope": 17, "vlmbias": 16, "mme": 17, "vlind": 17, "mmhalbench": 17},
    },
    "llava-hf/llava-1.5-7b-hf": {
        "n_layers":             32,
        "spatial_merge_size":   1,
        "image_token":          None,   # use model.config.image_token_index
        # ── NOT tuned — ClearSight paper starting point ──
        "layer_start":          8,
        "layer_end":            20,
        "head_top_k_pct":       0.20,
        "clip_coarse_grid":     6,      # LLaVA uses 336px images → slightly smaller grid
        "clip_top_k_pct":       0.30,
        "clip_fallback_thresh": 0.20,
        "clip_patch_thresh":    0.27,   # v3 patch-presence backup gate threshold
        "saliency_mode":        "clip_full_gate_v3",
        "clip_model":           "openai/clip-vit-base-patch32",
        "hssa_layer_idx":       16,
        "clip_saliency_method": "clip_patch",
        "srf2_clip_weight":     0.7,
        "srf2_hssa_weight":     0.3,
        "lta_layer_idx":        -1,
        "lta_weight":           0.6,
        "clip_weight":          0.4,
        "dataset_layer_end":    {"mmvp": 20, "pope": 20, "vlmbias": 19, "mme": 20, "vlind": 20, "mmhalbench": 20},
    },
}

# Fallback arch params for unknown model IDs (conservative starting points)
SRF_ARCH_FALLBACK = {
    "n_layers":             32,
    "spatial_merge_size":   2,
    "image_token":          "<|image_pad|>",
    "layer_start":          9,
    "layer_end":            17,
    "head_top_k_pct":       0.20,
    "clip_coarse_grid":     7,
    "clip_top_k_pct":       0.30,
    "clip_fallback_thresh": 0.20,
    "clip_patch_thresh":    0.27,
    "saliency_mode":        "clip",
    "clip_model":           "openai/clip-vit-base-patch32",
    "hssa_layer_idx":       16,
    "clip_saliency_method": "clip_patch",
    "srf2_clip_weight":     0.7,
    "srf2_hssa_weight":     0.3,
    "lta_layer_idx":        -1,
    "lta_weight":           0.6,
    "clip_weight":          0.4,
    "dataset_layer_end":    {},
}


def get_arch(model_id: str) -> dict:
    """Return arch params for model_id, falling back to SRF_ARCH_FALLBACK."""
    return SRF_ARCH_PARAMS.get(model_id, SRF_ARCH_FALLBACK)
