#!/bin/bash
# Autonomous overnight monitoring and error recovery

LOGFILE="autonomous_night.log"
check_interval=300  # 5 minutes

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

restart_experiment() {
    local gpu=$1
    local config=$2
    
    log_msg "RESTARTING GPU $gpu: $config"
    
    case $config in
        "baseline")
            CUDA_VISIBLE_DEVICES=$gpu conda run -n mllm nohup python -u srf/eval.py \
                --method baseline \
                --model "llava-hf/llava-1.5-7b-hf" \
                --datasets pope \
                --pope_splits adversarial \
                --n_pope 100 \
                --output "gpu${gpu}_baseline_rerun/" \
                > "gpu${gpu}_baseline_rerun.log" 2>&1 &
            ;;
        "post_a015")
            CUDA_VISIBLE_DEVICES=$gpu conda run -n mllm nohup python -u srf/eval.py \
                --method srf \
                --model "llava-hf/llava-1.5-7b-hf" \
                --datasets pope \
                --pope_splits adversarial \
                --n_pope 100 \
                --alpha 0.15 \
                --eps 0.0 \
                --layer_start 10 \
                --layer_end 15 \
                --head_top_k_pct 0.5 \
                --output "gpu${gpu}_post_a015_rerun/" \
                > "gpu${gpu}_post_a015_rerun.log" 2>&1 &
            ;;
        *)
            log_msg "Unknown config: $config"
            return 1
            ;;
    esac
    
    log_msg "Restarted GPU $gpu with PID: $!"
}

check_and_restart() {
    log_msg "=== Checking experiment status ==="
    
    # Check running processes
    running=$(ps aux | grep "srf/eval.py" | grep -v grep | wc -l)
    log_msg "Running processes: $running"
    
    # Check GPU status
    log_msg "GPU utilization:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
        awk -F, '{printf "  GPU %s: %s%% utilized, %sMiB\n", $1, $2, $3}' | tee -a "$LOGFILE"
    
    # Check completed experiments
    completed=$(ls -1 gpu*/pope.json 2>/dev/null | wc -l)
    log_msg "Completed experiments: $completed/8"
    
    # Check for errors in logs
    for logfile in gpu*.log; do
        if [ -f "$logfile" ]; then
            errors=$(grep -i "error\|exception\|traceback" "$logfile" | tail -5)
            if [ -n "$errors" ]; then
                log_msg "ERRORS FOUND in $logfile"
                echo "$errors" | tee -a "$LOGFILE"
                
                # Try to identify which GPU/config and restart
                gpu=$(echo "$logfile" | sed 's/gpu\([0-9]\).*/\1/')
                log_msg "Attempting to restart GPU $gpu..."
                # Add specific restart logic based on log analysis
            fi
        fi
    done
    
    # Check for hung processes (low GPU utilization but process running)
    for i in {0..7}; do
        gpu_util=$(nvidia-smi --query-gpu=i,utilization.gpu --format=csv,noheader,nounits | grep "^$i," | awk -F, '{print $2}' | tr -d ' ')
        if [ "$gpu_util" -lt 10 ]; then
            # Check if process should be running on this GPU
            if ps aux | grep "srf/eval.py" | grep -v grep | grep -q "CUDA_VISIBLE_DEVICES=$i"; then
                log_msg "WARNING: GPU $i has low utilization ($gpu_util%) but process running"
                # Could be hung - consider killing and restarting
            fi
        fi
    done
}

# Main monitoring loop
log_msg "=== AUTONOMOUS OVERNIGHT MONITORING STARTED ==="

while true; do
    check_and_restart
    
    # Check if all experiments completed successfully
    completed=$(ls -1 gpu*/pope.json 2>/dev/null | wc -l)
    if [ "$completed" -ge 8 ]; then
        log_msg "All 8 experiments completed! Running comprehensive analysis..."
        
        # Generate summary report
        python -c "
import json
import glob

print('='*60)
print('FINAL RESULTS SUMMARY')
print('='*60)

baseline_acc = None
results = []

for result_file in sorted(glob.glob('gpu*/pope.json')):
    dir_name = result_file.split('/')[0]
    with open(result_file) as f:
        data = json.load(f)
    
    if 'baseline' in data and 'method' not in data:
        baseline_acc = data['baseline']['accuracy']
        print(f'{dir_name:20s}: Baseline={baseline_acc:.4f}')
    else:
        acc = list(data['method'].values())[0]['accuracy']
        delta = acc - baseline_acc
        results.append((dir_name, acc, delta))
        print(f'{dir_name:20s}: SRF={acc:.4f} (Δ={delta:+.4f})')

print()
print('ANALYSIS:')
if baseline_acc:
    print(f'Baseline accuracy: {baseline_acc:.4f}')
    
    best_result = max(results, key=lambda x: x[1])
    worst_result = min(results, key=lambda x: x[1])
    
    print(f'Best result: {best_result[0]} ({best_result[1]:.4f}, Δ={best_result[2]:+.4f})')
    print(f'Worst result: {worst_result[0]} ({worst_result[1]:.4f}, Δ={worst_result[2]:+.4f})')
    
    any_improvement = any(delta > 0.01 for _, _, delta in results)
    if any_improvement:
        print('SUCCESS: Found configurations with >1% improvement!')
    else:
        print('NO IMPROVEMENT: All configurations ≤1% delta from baseline')
        print('Consider: (1) Post-softmax implementation, (2) Layer-specific zones, (3) Gradient-based')
" >> "$LOGFILE"
        
        log_msg "Summary report generated. Sleeping 30 minutes before next check..."
        sleep 1800  # Sleep 30 minutes
    fi
    
    log_msg "Sleeping for $check_interval seconds..."
    sleep $check_interval
done
