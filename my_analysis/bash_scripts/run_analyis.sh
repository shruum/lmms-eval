#!/bin/bash

set -e  # stop script if any command fails

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
ATTN=""

echo "Running bias analysis..."
PYTHONPATH=. python vlms_are_biased_empirical.py \
    --model_name "$MODEL" \
    --output_dir results/analysis/bias_new \
    --attn_implementation "$ATTN"

#echo "Running POPE analysis..."
#PYTHONPATH=. python pope_empirical_analysis.py \
#    --model_name "$MODEL" \
#    --output_dir results/analysis/pope \
#    --attn_implementation "$ATTN"

#echo "Running MMBench analysis..."
#PYTHONPATH=. python mmbench_empirical_analysis.py \
#    --model_name "$MODEL" \
#    --output_dir results/analysis/mmbench \
#    --attn_implementation "$ATTN"

echo "All analyses completed successfully!"