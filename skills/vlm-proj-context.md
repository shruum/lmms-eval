  <!-- cat > /home/anna2/shruthi/lmms-eval/skills/vlm-proj-context/SKILL.md << 'EOF' -->
  ---
  name: vlm-proj-context
  description: This skill should be used when the user asks to "load vlm project context", "load hallucination mitigation context",
  "/vlm-proj-context", or discusses VLM hallucination mitigation, SRF (Spatial Reasoning Focus), CLIP-guided saliency steering, or the lmms-eval SRF   
  project. Provides comprehensive project context for VLM hallucination research.
  version: 1.0.0                                                                                                                                       
  ---             

  # VLM Hallucination Mitigation Research — Project Context

  ## Research Goal

  **Mitigate hallucination in Vision-Language Models (VLMs) by steering attention toward semantically relevant image regions using inference-time
  strategies.**

  **Core Problem**: VLMs frequently hallucinate objects that don't exist in images (e.g., answering "yes" to "Is there a cat?" when no cat is present).
   
  **Approach**: CLIP-guided spatial reasoning steering (SRF) to boost attention to relevant regions and suppress background.                           
                  
  ---

  ## Step 1 — Locate the Repository

  ```bash
  cd /home/anna2/shruthi/lmms-eval

  Call this {REPO}. All paths below are relative to {REPO}.

  ---
  Step 2 — Read Core Context Files (Read in Parallel)

  ┌───────────────────────────────┬─────────────────────────────────────────────────────────────────┐
  │             File              │                             Purpose                             │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ {REPO}/srf/CONTEXT.md         │ Algorithm, file map, hyperparameters, CLI reference             │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ {REPO}/auto_research/Lit.md   │ Literature review: foveation, token pruning, negative prompting │                                                  
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ {REPO}/srf/config.py          │ Central configuration (models, datasets, hyperparameters)       │                                                  
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ {REPO}/srf/RESEARCH_STATUS.md │ Current results, open tasks, experiment logs                    │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────┤
  │ {REPO}/srf/autoresearch_loop/  │ AutoResearch setup (program.md, prepare.py, experiments)     │
  └───────────────────────────────┴─────────────────────────────────────────────────────────────────┘

  ---
  Step 3 — Understand the Research Context

  Problem Statement

  VLMs suffer from object hallucination: generating text that describes objects not present in the image. This is particularly problematic for:        
  - Absent objects (most common): "Is there a cat?" → "yes" (when no cat exists)
  - Spatial reasoning: Failing to focus on relevant regions for answering                                                                              
  - Language prior: Answering based on question wording rather than visual evidence

  Solution: SRF (Spatial Reasoning Focus)

  Core Mechanism:
  1. Extract query nouns from questions (e.g., "cat" from "Is there a cat?")
  2. Compute CLIP-based saliency maps to identify relevant image patches    
  3. During inference, in cross-modal fusion layers (8-15):             
    - Boost attention to salient image tokens by α (default 4.0)
    - Suppress background tokens by ε (default 0.2)                                                                                                    
  4. This steers the VLM to look at the right regions when answering
                                                                                                                                                       
  Variants:       
  - SRF (base): Single-pass attention steering
  - SRF-E (evidence-amplified): Two-pass contrastive (with image vs. zeroed image), amplifies visual evidence

  Key Innovations

  1. CLIP cross-modal guidance: Uses external CLIP model to identify relevant regions based on query                                                   
  2. Query-conditioned selection: Different questions focus on different regions
  3. Training-free: Pure inference-time intervention, no model weight updates                                                                          
  4. Layer-specific: Targets middle fusion layers where cross-modal reasoning happens
  5. Noun-based: Extracts relevant concepts from questions to guide saliency

  Related Work (from Literature Review)

  ┌───────────────────────┬─────────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │        Method         │                Technique                │                            Key Idea                            │
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ LLMind (CVPR 2026)    │ Möbius warp + SPSA optimization         │ Foveation via adaptive sampling, but requires ground-truth     │
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Foveated Reasoner     │ Autoregressive foveation                │ Triggers high-res crops during decoding, but requires training │                 
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ ADSC                  │ Attention-driven self-compression       │ LLM prunes its own vision tokens based on attention            │                 
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Multimodal Unlearning │ Negative prompts                        │ Steer away from biased associations at inference time          │
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ AIR (LookCarefully)   │ OT-guided patch selection               │ Prunes non-salient patches, reinforces salient ones            │
  ├───────────────────────┼─────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ VAF (ClearSight)      │ Boost visual attention in middle layers │ Similar to SRF but without CLIP guidance                       │
  └───────────────────────┴─────────────────────────────────────────┴────────────────────────────────────────────────────────────────┘                 
   
  SRF Differentiators:                                                                                                                                 
  - CLIP-based external guidance (vs. internal attention only)
  - Noun-based query conditioning (vs. fixed saliency)
  - Two-pass contrastive amplification (SRF-E)
  - Focus on absence detection (hallucination mitigation)

  ---
  Step 4 — Current Implementation Status

  Models Tested

  ┌───────────────┬──────────────┬──────────────────────────────────────────┐
  │     Model     │    Status    │                  Notes                   │
  ├───────────────┼──────────────┼──────────────────────────────────────────┤
  │ Qwen2.5-VL-3B │ ✅ Tuned     │ layer_start=8, layer_end=15 — main model │
  ├───────────────┼──────────────┼──────────────────────────────────────────┤
  │ Qwen2.5-VL-7B │ ⚠️  Not tuned │ Proportional scaling: start=9, end=17    │
  ├───────────────┼──────────────┼──────────────────────────────────────────┤                                                                          
  │ LLaVA-1.5-7B  │ ⚠️  Not tuned │ start=8, end=20                          │
  └───────────────┴──────────────┴──────────────────────────────────────────┘                                                                          
                  
  Datasets

  ┌──────────┬─────────────────────┬────────────┬───────────────────┬──────────────────┐
  │ Dataset  │          N          │    Task    │      Metric       │      Status      │
  ├──────────┼─────────────────────┼────────────┼───────────────────┼──────────────────┤
  │ MMVP     │ 150 pairs (300 img) │ A/B choice │ Pair accuracy     │ ✅ Strong gains  │
  ├──────────┼─────────────────────┼────────────┼───────────────────┼──────────────────┤
  │ POPE     │ 9000 (3×3000)       │ Yes/No     │ Question accuracy │ ✅ Solid gains   │
  ├──────────┼─────────────────────┼────────────┼───────────────────┼──────────────────┤                                                               
  │ MME      │ 2374 (14 cats)      │ Yes/No     │ Score, pair acc   │ ⚠️  Mixed results │
  ├──────────┼─────────────────────┼────────────┼───────────────────┼──────────────────┤                                                               
  │ VLM Bias │ ~300 (7 cats)       │ Generation │ Exact match       │ ✅ Tested        │
  └──────────┴─────────────────────┴────────────┴───────────────────┴──────────────────┘

  Hyperparameter System

  Three-tier config in srf/config.py:

  Priority: CLI args → SRF_ARCH_PARAMS → SRF_DATASET_PARAMS → SRF_DEFAULTS

  SRF_DEFAULTS          # Shared: sys_beta=0.10, calib_n=20, bias_mode, prob_floor
  SRF_DATASET_PARAMS    # Per-dataset (arch-agnostic)                                                                                                  
  SRF_ARCH_PARAMS       # Per-model (scales with depth)
                                                                                                                                                       
  Key Parameters: 
  - layer_start, layer_end: Fusion zone boundaries (model-specific)
  - head_top_k_pct: Fraction of vision-aware heads (default 0.20)  
  - alpha: Boost factor for salient tokens (default 4.0)         
  - eps: Suppression factor for background (default 0.2)
  - clip_coarse_grid: CLIP patch grid (7 for Qwen/448px, 6 for LLaVA/336px)
  - clip_top_k_pct: Fraction of image tokens boosted (default 0.30)
                                                                                                                                                       
  ---
  Step 5 — Quick Start Commands

  **NEW: AutoResearch Loop**
  ```bash
  # Start autonomous experiments (Claude Code)
  # Tell Claude: "Read srf/autoresearch_loop/program.md and start the SRF autoresearch loop"
  ```
                                                                                        
  Environment Setup

  # Conda environment
  conda activate mllm
  export HF_HOME=/path/to/hf_cache
  export CUDA_VISIBLE_DEVICES=0

  # Repository location
  cd /home/anna2/shruthi/lmms-eval

  Evaluation Commands

  # Quick test (POPE, adversarial split)
  python srf/eval.py \
    --method srf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets pope \                                                                                                                                  
    --pope_splits adversarial \
    --n_pope 10 \                                                                                                                                      
    --output results/test_run/

  # Full evaluation (all datasets)
  python srf/eval.py \
    --method srf \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets mmvp pope mme vlmbias \
    --output results/srf_full/

  # SRF-E (evidence-amplified, with beta sweep)
  python srf/eval.py \
    --method srfe \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --datasets mmvp \                                                                                                                                  
    --beta 0.5 1.0 2.0 \
    --output results/srfe_sweep/                                                                                                                       
                  
  Code Architecture

  srf/
  ├── config.py              # SINGLE SOURCE OF TRUTH for all hyperparams
  ├── eval.py                # Unified evaluation CLI
  ├── srf.py                 # SRF base implementation
  ├── srf_e.py               # SRF-E (two-pass contrastive)
  ├── eval_datasets.py       # Dataset loaders
  ├── noun_extract.py        # CLIP query noun extraction
  ├── autoresearch_loop/      # AutoResearch setup (program.md, prepare.py, experiments)
  │   ├── program.md         # Instructions for Claude Code agent
  │   ├── prepare.py         # Fixed utilities (DO NOT MODIFY)
  │   └── autoresearch_llava.py  # Experiment runner
  └── saliency/                                                                                                                                        
      ├── clip_salience.py   # CLIP patch saliency (cross-modal encoder)
      └── hssa_salience.py   # Hidden-state saliency (experimental)                                                                                    
                  
  my_analysis/
  ├── qwen_attn_patch.py     # Attention patching engine (shared by srf/)
  │                           # Contains: patch_model, identify_visual_heads,
  │                           #          update_sample, _STATE{}
  └── autoresearch*/         # Completed experiment loops (reference only)

  Current Branch

  git branch --show-current

  Expected: autoresearch/mmvp-srf or similar SRF-focused branch.
   
  **AutoResearch:** New autoresearch loop in `srf/autoresearch_loop/` inspired by https://github.com/karpathy/autoresearch
  - AI agents autonomously test SRF improvements
  - Target: LLaVA-7B + POPE adversarial (n=100)
  - Goal: +1-3% improvement over baseline
  - Usage: "Read {REPO}/srf/autoresearch_loop/program.md and start the SRF autoresearch loop"                                                                                       
   
  ---
  Step 6 — AutoResearch Loop (NEW)

  **Inspired by:** https://github.com/karpathy/autoresearch

  **Goal:** AI agents autonomously run SRF experiments on LLaVA-7B + POPE (adversarial, n=100) while you sleep.

  **Location:** `{REPO}/srf/autoresearch_loop/`

  **Structure (Karpathy-style):**
  - `program.md` — Instructions for Claude Code agent (YOU edit this)
  - `prepare.py` — Fixed utilities (DO NOT MODIFY)
  - `srf.py` — Agent modifies this file (algorithm changes)
  - `autoresearch_llava.py` — Script to run experiments

  **How to use:**
  1. Give Claude: "Read {REPO}/srf/autoresearch_loop/program.md and start the SRF autoresearch loop"
  2. Agent reads instructions, suggests changes to `srf.py`
  3. Runs experiments (n=100, ~10 min each)
  4. Keeps winners (Δ > +0.5%), discards losers
  5. Reports best configuration

  **Current experiments testing:**
  - Option 1: Better salience (5×5, 9×9 grids, multi-scale CLIP)
  - Option 2: Graduated boost (multi-stage, stronger/weaker variants)

  **Target:** +1-3% improvement over baseline (~84% → >85%)

  ---
  Step 7 — Key Design Principles

  Based on literature review, the approach incorporates:

  ┌─────────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────┐
  │          Principle          │        Source         │                     Application                     │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤
  │ Absence-aware processing    │ Multimodal Unlearning │ Different strategies for present vs. absent objects │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤
  │ Query-conditioned selection │ Foveated Reasoner     │ Different tokens boosted per question               │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤
  │ Negative prompting          │ PhysVid               │ Potential: steer away from biased patterns          │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤
  │ Training-free optimization  │ LLMind (CSF)          │ Inference-time only, no gradient updates            │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤                                        
  │ Adaptive thresholds         │ AdaptVis (ICML 2025)  │ Layer-gated interventions                           │
  ├─────────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────┤                                        
  │ Token selectivity           │ AIR (ICLR 2026)       │ Boost salient, suppress non-salient                 │
  └─────────────────────────────┴───────────────────────┴─────────────────────────────────────────────────────┘

  ---
  Step 8 — Future Directions (from Lit.md + autoresearch)

  **Current autoresearch experiments:**
  1. Multi-scale CLIP (5×5, 9×9 grids, ensemble)
  2. Graduated multi-stage attention (α_high=4.0, α_mid=2.0, α_low=1.0)
  3. Cross-modal refinement (VLM attention refines CLIP saliency)
  
  **Planned experiments:**
  1. CLIP-Guided Negative Prompting: Different strategies for present vs absent
  2. Contrastive Saliency: Subtract background distractor saliency from query saliency
  3. Möbius Warp + Attention: Zoom into salient regions before VLM processing
  4. Layer-Adaptive Thresholds: Per-layer absence detection calibration                                                                                
  5. Ensemble Saliency: Combine CLIP (external) + attention rollout (internal)
  
  **All experiments tracked in:** `{REPO}/srf/autoresearch_loop/`
                                                                                                                                                       
  ---             
  Important Notes

  - All results should be logged in srf/RESEARCH_STATUS.md
  - Code changes update srf/CONTEXT.md if structure changes
  - Hyperparameter tuning should update srf/config.py (not hardcoded)
  - New datasets require updates to both srf/config.py and srf/eval_datasets.py

  ---
  This context loads automatically when you invoke /vlm-proj-context. Always verify the current branch and recent commits before making changes.
  