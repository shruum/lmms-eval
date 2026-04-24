---
name: autoresearch
version: v1.0
description: Autonomous research loop for lmms-eval — modifies task prompts, model adapters, or scoring logic; evaluates on a fixed benchmark subset; keeps improvements, discards regressions, and repeats overnight. Inspired by Karpathy's autoresearch. Invoke with /autoresearch.
---

# autoresearch — Autonomous Research Loop for lmms-eval

Runs an autonomous, git-ratcheted improvement loop on a single target (task prompts, model adapter, or scorer). Each cycle: propose one change → evaluate on a fixed held-out subset → keep if better, revert if worse → log → repeat indefinitely until interrupted.

**Key principle**: if you can define a scalar metric and automate evaluation, you can discover improvements while you sleep.

---

## Setup Phase

Work with the user to configure exactly once before entering the loop.

### 1. Agree on a run tag

Propose a tag based on today's date and target (e.g. `apr23-chartqa-prompts`). The branch `autoresearch/<tag>` must not already exist.

```bash
git checkout -b autoresearch/<tag>
```

### 2. Define the research target

Pick **exactly one** modifiable scope from the table below. Do not mix scopes across iterations — one scope, one metric, one loop.

| Target | Modifiable files | What you change |
|--------|-----------------|-----------------|
| **Task prompts** | `lmms_eval/tasks/<task>/*.yaml` (`lmms_eval_specific_kwargs`) | `pre_prompt`, `post_prompt`, answer format instructions |
| **Task scorer** | `lmms_eval/tasks/<task>/utils.py` | `process_results`, normalization, answer extraction |
| **Model adapter** | `lmms_eval/models/chat/<model>.py` or `lmms_eval/models/simple/<model>.py` | Inference, preprocessing, batching |

### 3. Read context files (read-only — never modify)

- `README.md` — project overview
- `lmms_eval/api/task.py` — task base class and evaluation contract
- `lmms_eval/evaluator.py` — core eval loop (immutable ground truth)
- `lmms_eval/tasks/<target-task>/*.yaml` — task config
- `lmms_eval/tasks/<target-task>/utils.py` — current scoring logic

### 4. Define the metric

Choose **one** scalar from lmms-eval output. Lower-index examples must not appear in your training assumptions:

| Task type | Metric key | Direction |
|-----------|-----------|-----------|
| VQA / MC  | `accuracy` | higher |
| Captioning | `CIDEr` or `BLEU_4` | higher |
| Regression | `RMSE` | lower |
| Custom     | as defined in `process_results` | specify |

### 5. Choose the eval subset size

Balance speed vs. reliability. A fixed `--limit N` is used for every run so results are comparable:

| Goal | Suggested `--limit` | Approx time per run |
|------|---------------------|---------------------|
| Fast iteration (overnight) | 50–100 | 1–3 min |
| Moderate confidence | 200–500 | 5–15 min |
| High confidence | full split | 30+ min |

Record the chosen limit. Never change it mid-loop.

### 6. Establish baseline

Run the unmodified code first:

```bash
python -m lmms_eval \
  --model <model_id> \
  --model_args pretrained=<checkpoint> \
  --tasks <task_name> \
  --batch_size 1 \
  --limit <N> \
  --log_samples \
  --output_path run_output/ \
  > run.log 2>&1
```

Extract the metric:

```bash
python -c "
import json, glob, sys
files = glob.glob('run_output/**/*.json', recursive=True)
for f in files:
    d = json.load(open(f))
    results = d.get('results', {})
    for task, metrics in results.items():
        print(task, metrics)
"
```

### 7. Initialize results.tsv

```
commit	metric	status	description
<hash>	<baseline_value>	keep	baseline
```

Do NOT commit `results.tsv` — leave it untracked.

### 8. Confirm and begin

Confirm setup with the user, then enter the loop. **Do not ask for permission to continue once the loop starts.**

---

## File Ownership Rules

```
IMMUTABLE (never touch):
  lmms_eval/evaluator.py          — evaluation harness
  lmms_eval/api/task.py           — task base class
  lmms_eval/api/model.py          — model base class
  lmms_eval/api/registry.py       — registration system

AGENT MODIFIABLE (one scope only):
  lmms_eval/tasks/<task>/*.yaml   — if target = task prompts
  lmms_eval/tasks/<task>/utils.py — if target = task scorer
  lmms_eval/models/chat/<m>.py    — if target = model adapter
  lmms_eval/models/simple/<m>.py  — if target = model adapter (legacy)

HUMAN ONLY:
  skills/autoresearch/SKILL.md    — this file
  CLAUDE.md                       — project instructions
```

---

## The Experiment Loop

LOOP FOREVER until manually interrupted:

### Step 1 — Review state

```bash
git log --oneline -10           # what has been tried
cat results.tsv                 # metric history
git diff HEAD                   # current working state
```

Synthesize: what patterns emerge from results? What angle hasn't been tried?

### Step 2 — Propose ONE change

Pick the single most promising untested hypothesis. Ground your choice in prior results, not intuition alone. Examples:

- **Prompt target**: Add chain-of-thought reasoning instruction; change answer format from free-text to letter (A/B/C/D); add explicit "look at the image carefully" prefix
- **Scorer target**: Normalize answer case; strip punctuation before exact-match; add synonym mapping for common answer variants
- **Adapter target**: Adjust temperature/top-p; change image resize strategy; modify system prompt

One change per cycle. This is a hard rule — isolating variables is what makes results interpretable.

### Step 3 — Apply and commit

Edit the file(s), then commit before running:

```bash
git add lmms_eval/tasks/<task>/utils.py  # or whichever file changed
git commit -m "experiment: <brief description of change>"
```

The commit message must start with `experiment:` so the git log is parseable.

### Step 4 — Run evaluation

```bash
python -m lmms_eval \
  --model <model_id> \
  --model_args pretrained=<checkpoint> \
  --tasks <task_name> \
  --batch_size 1 \
  --limit <N> \
  --output_path run_output/ \
  > run.log 2>&1
```

Redirect all output — do NOT use `tee` or let output flood your context.

**Timeout**: If a run exceeds 3× the baseline runtime, kill it (`kill <pid>`) and treat as a crash.

### Step 5 — Extract the metric

```bash
python -c "
import json, glob
files = sorted(glob.glob('run_output/**/*.json', recursive=True))
if not files:
    print('NO_OUTPUT')
else:
    d = json.load(open(files[-1]))
    results = d.get('results', {})
    for task, metrics in results.items():
        print(task, json.dumps(metrics, indent=2))
"
```

If output is `NO_OUTPUT` or the run crashed, see the crash protocol below.

### Step 6 — Keep or discard

Compare to the current best:

- **Improved** (metric is better by any amount): advance the branch. This commit is the new baseline.
- **Equal or worse**: revert.
  ```bash
  git reset --hard HEAD~1
  ```
- **Simplification win** (equal metric, less code): keep — always prefer simpler.

### Step 7 — Log to results.tsv

```bash
HASH=$(git rev-parse --short HEAD)
echo -e "${HASH}\t<metric_value>\t<keep|discard|crash>\t<description>" >> results.tsv
```

Never commit `results.tsv`.

### Step 8 — Repeat

Go to Step 1. Do NOT pause. Do NOT ask the user if you should continue. **NEVER STOP** until manually interrupted.

---

## Crash Protocol

If a run produces no output or throws an exception:

1. Check `tail -50 run.log` for the Python traceback.
2. If it's a trivial fix (syntax error, wrong key name, missing import): fix it, amend the commit, re-run.
3. If the idea is fundamentally broken (incompatible architecture, metric undefined): revert and log as `crash`.
4. If you see an OOM error: revert, log as `crash`, note memory constraint in description, avoid similar ideas.
5. After 2–3 failed fix attempts on the same idea: give up, revert, move on.

```bash
git reset --hard HEAD~1
echo -e "$(git rev-parse --short HEAD)\t0.000000\tcrash\t<description>" >> results.tsv
```

---

## Idea Generation When Stuck

If the last 5 experiments all produced no improvement, try a qualitatively different angle:

**For prompt targets:**
- Switch from instructive to example-based prompts (few-shot style in the post_prompt)
- Try removing all instructions (minimal prompt)
- Add explicit answer constraints ("respond with only the letter")
- Add task-specific context ("This is a chart reading task")

**For scorer targets:**
- Look at failing samples in `run_output/` to find systematic error patterns
- Check if the model's answer format differs from what the scorer expects
- Try fuzzy matching (edit distance) instead of exact match
- Normalize numbers, dates, or units

**For model adapter targets:**
- Try a different image resolution or aspect ratio
- Adjust the generation length limit
- Change the system prompt framing
- Try different sampling strategy

**Radical option**: Combine the top 2 previous near-misses that independently gave small gains — sometimes the combination is super-additive.

---

## Simplicity Criterion

All else equal, simpler is better:

- Small improvement + complex code → probably not worth it
- No improvement + much simpler code → keep (simplification win)
- Any improvement + fewer lines → definitely keep
- Large improvement + complex code → keep, but note complexity cost in description

---

## Results TSV Format

Tab-separated, never comma-separated (commas appear in descriptions).

```
commit	metric	status	description
a1b2c3d	0.723	keep	baseline
b2c3d4e	0.741	keep	add "look at image carefully" prefix
c3d4e5f	0.739	discard	add chain-of-thought instruction (worse)
d4e5f6g	0.000	crash	answer normalization broke JSON parsing
e5f6g7h	0.748	keep	strip trailing punctuation in scorer
```

Columns:
1. Git short hash (7 chars)
2. Metric value (6 decimal places, 0.000000 for crashes)
3. Status: `keep`, `discard`, or `crash`
4. Short description of the change tried

---

## Quick Reference

```bash
# Start a run
python -m lmms_eval --model <m> --model_args pretrained=<ckpt> \
  --tasks <task> --batch_size 1 --limit <N> \
  --output_path run_output/ > run.log 2>&1

# Extract metric from output
python -c "import json,glob; f=sorted(glob.glob('run_output/**/*.json',recursive=True))[-1]; d=json.load(open(f)); print(d.get('results',{}))"

# Keep a change
git add <file> && git commit -m "experiment: <desc>"

# Discard a change
git reset --hard HEAD~1

# Log a result
echo -e "$(git rev-parse --short HEAD)\t<val>\t<status>\t<desc>" >> results.tsv

# See history
cat results.tsv
git log --oneline
```

---

## Example Overnight Session

A user might say: "Run autoresearch on chartqa prompts overnight using qwen2_5_vl with --limit 100."

You would:
1. Create branch `autoresearch/apr23-chartqa-prompts`
2. Read task YAML and utils.py
3. Run baseline, log it
4. Enter the loop — ~20 experiments over 8 hours at ~25 min each (eval) or ~60+ at ~8 min each (fast iteration)
5. User wakes up to `results.tsv` showing which prompt changes improved accuracy, and a git log of kept commits

The user then inspects `results.tsv`, cherry-picks the best improvements to `main`, and optionally starts a new loop with a different target.
