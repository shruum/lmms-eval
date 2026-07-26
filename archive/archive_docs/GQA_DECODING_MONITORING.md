# GQA Decoding Test - Active Monitoring

**Status:** 🟢 **MONITORING ACTIVE** (started 2026-05-14 13:23:42)
**Purpose:** Test if sampling decoding fixes GQA Adversarial baseline discrepancy

---

## What's Happening Now

A background script is monitoring all GPUs (0-7) and will **automatically run the GQA decoding test** when any GPU has ≥15GB free memory.

**Monitoring Script PID:** 645070

**What it does:**
1. Checks all 8 GPUs every 60 seconds
2. When a GPU has ≥15GB free (VLLM worker finished), it launches:
   - **Test 1:** Greedy decoding (100 samples) - our current baseline
   - **Test 2:** Sampling decoding (100 samples) - VCD paper method (temp=0.7)
3. Saves results to `/home/anna2/shruthi/lmms-eval/results/gqa_decoding_test/`

---

## Check Status Anytime

```bash
# Quick status check
bash /home/anna2/shruthi/lmms-eval/check_gqa_test_status.sh

# Real-time monitoring
tail -f /home/anna2/shruthi/lmms-eval/results/gqa_decoding_test/monitor.log

# Check if monitoring script is still running
ps -p 645070
```

---

## Expected Outcomes

### If Sampling Fixes Overprediction (VCD hypothesis confirmed):
- **Greedy (current):** ~70% accuracy, ~75% yes ratio (overpredicts "yes")
- **Sampling (VCD):** ~75% accuracy, ~50% yes ratio (balanced)
- **Conclusion:** Decoding method was the issue

### If Sampling Doesn't Help:
- **Both methods:** ~70% accuracy, ~75% yes ratio
- **Conclusion:** Issue is elsewhere (dataset version, prompt format, answer parsing, etc.)

---

## Test Results Location

When tests complete, results will be saved here:

```
/home/anna2/shruthi/lmms-eval/results/gqa_decoding_test/
├── monitor.log              # Monitoring log
├── monitor_stdout.log       # Script stdout
├── decoding_comparison.json # Test results (metrics + sample responses)
└── gqa_adversarial_test_*.log  # Individual test logs
```

---

## What Happens Next

### If Test Shows Improvement:
1. **Re-run full GQA Adversarial** (3000 samples) with sampling decoding
2. **Compare with VCD paper baseline** (75.08% accuracy)
3. **If it matches:** Re-run all POPE baselines with sampling decoding
4. **Update all investigation documents** with findings

### If Test Shows No Improvement:
1. **Investigate other causes:**
   - Dataset version differences
   - Prompt format variations
   - Answer parsing method
   - Model checkpoint differences
2. **Check VCD GitHub repo** for exact evaluation code
3. **Compare data distributions** (image selections, question formats)

---

## Current GPU Status

All GPUs currently occupied by VLLM workers (~44GB used each):
- **Free memory per GPU:** ~4.8GB
- **Required for LLaVA-1.5-7B:** ~14GB
- **Waiting for:** Any VLLM worker to finish

---

## Manual Control

### Stop Monitoring:
```bash
kill 645070
```

### Restart Monitoring:
```bash
cd /home/anna2/shruthi/lmms-eval
nohup bash monitor_and_run_gqa_test.sh > results/gqa_decoding_test/monitor_stdout.log 2>&1 &
```

### Manually Run Test (if GPU becomes available):
```bash
# On GPU X (replace X with available GPU ID)
conda run -n mllm python srf/eval_pope_vcd_fixed.py \
  --method baseline \
  --model llava-hf/llava-1.5-7b-hf \
  --pope_vcd_file /home/anna2/shruthi/dataset/POPE_images/gqa_adversarial.json \
  --pope_vcd_name gqa_adversarial_sampling \
  --image_dir /home/anna2/shruthi/dataset/POPE_images/images/gqa \
  --output results/gqa_sampling \
  --device cuda:X \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.9
```

---

## Background Information

### Why This Test Matters

Our GQA Adversarial baseline is **-5.45% lower** than VCD paper:
- **Our baseline (greedy):** 69.63% accuracy, 75.83% yes ratio
- **VCD paper (sampling):** 75.08% accuracy, ~50% yes ratio

**Hypothesis:** VCD paper uses sampling decoding (`do_sample=True`), we use greedy (`do_sample=False`). This difference causes our model to overpredict "yes" (75.83% vs expected 50%).

### Files Created

1. **`test_gqa_decoding.py`** - Comprehensive test script comparing greedy vs sampling
2. **`monitor_and_run_gqa_test.sh`** - Background monitoring script (auto-runs when GPU available)
3. **`check_gqa_test_status.sh`** - Quick status check script
4. **`GQA_DECODING_TEST_PLAN.md`** - Complete test plan and documentation

### Modified Files

1. **`srf/eval_pope_vcd_fixed.py`** - Added `--do_sample`, `--temperature`, `--top_p` arguments
2. **`pope.md`** - Added "Running Scripts" section documenting all POPE evaluation scripts

---

## Next Actions

1. ⏳ **Wait for monitoring script to detect available GPU**
2. 🧪 **Tests run automatically** (greedy then sampling, 100 samples each)
3. 📊 **Review results** in `decoding_comparison.json`
4. 📝 **Update investigation documents** with findings
5. 🚀 **If successful:** Re-run all POPE baselines with sampling decoding

---

*Last updated: 2026-05-14 13:23 - Monitoring script active, waiting for GPU availability*
