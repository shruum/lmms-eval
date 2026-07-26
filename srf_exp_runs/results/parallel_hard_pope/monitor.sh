#!/bin/bash
echo "============================================"
echo "Parallel Hard POPE Sweep - Status"
echo "============================================"
echo ""
echo "GPU Utilization:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "GPU %s: %s%% utilized, %sMiB/%sMiB\n", $1, $2, $3, $4}'
echo ""
echo "Experiment Progress:"
for gpu in 0 1 2 3 4 5 6 7; do
    if [ -f "gpu${gpu}_baseline.log" ]; then
        log="gpu${gpu}_baseline.log"
    elif [ -f "gpu${gpu}_post_a015.log" ]; then
        log="gpu${gpu}_post_a015.log"
    elif [ -f "gpu${gpu}_post_a05.log" ]; then
        log="gpu${gpu}_post_a05.log"
    elif [ -f "gpu${gpu}_post_a10.log" ]; then
        log="gpu${gpu}_post_a10.log"
    elif [ -f "gpu${gpu}_layer_8_15.log" ]; then
        log="gpu${gpu}_layer_8_15.log"
    elif [ -f "gpu${gpu}_layer_10_20.log" ]; then
        log="gpu${gpu}_layer_10_20.log"
    elif [ -f "gpu${gpu}_vaf_like.log" ]; then
        log="gpu${gpu}_vaf_like.log"
    elif [ -f "gpu${gpu}_strong.log" ]; then
        log="gpu${gpu}_strong.log"
    else
        continue
    fi

    echo "GPU ${gpu}:"
    tail -3 "$log" | grep -E "\[.*\]|Accuracy|Saved" | tail -1
done
echo ""
echo "Completed experiments:"
ls -1 gpu_*_baseline/pope.json gpu_*/pope.json 2>/dev/null | wc -l
echo "of 8 expected"
