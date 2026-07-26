#!/bin/bash
# Monitor MME sequential runs - check every 20 minutes

LOG_FILE="/home/anna2/shruthi/lmms-eval/srf_exp_runs/monitoring_20min.log"

echo "========================================" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Check running processes
echo "Running eval processes:" | tee -a "$LOG_FILE"
ps aux | grep "srf/eval.py" | grep -v grep | wc -l | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# GPU status
echo "GPU 1 (Qwen):" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "GPU 2 (LLaVA):" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 2 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Latest log tails
echo "Qwen latest:" | tee -a "$LOG_FILE"
tail -2 /home/anna2/shruthi/lmms-eval/srf_exp_runs/qwen_seq.log | grep -E "\[.*\]|complete|Loading" | tail -1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "LLaVA latest:" | tee -a "$LOG_FILE"
tail -2 /home/anna2/shruthi/lmms-eval/srf_exp_runs/llava_seq.log | grep -E "\[.*\]|complete|Loading" | tail -1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Completed experiments
echo "Completed Qwen: $(ls -1 results/qwen_mme_seq/*.log 2>/dev/null | wc -l)/8" | tee -a "$LOG_FILE"
echo "Completed LLaVA: $(ls -1 results/llava_mme_seq/*.log 2>/dev/null | wc -l)/8" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Next check in 20 minutes..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
