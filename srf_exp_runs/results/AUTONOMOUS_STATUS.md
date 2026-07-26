# Autonomous Overnight Experiments - Status Report

**Started:** 2026-05-01 00:22:37 CEST
**Status:** RUNNING AUTONOMOUSLY

## Current Operations

### Active Processes
- **Master overnight script:** Running (PID 1480575)
- **Autonomous monitor:** Running (PID 1478624)
- **Active experiments:** 34+ processes across 8 GPUs

### GPU Utilization
- GPU 0: Model loaded (13.9GB)
- GPU 1: Active (19% utilization, 27.9GB)
- GPU 2-7: Models loaded (~13.8GB each)

### Experiment Waves

**Wave 1 (Current):** Early layer exploration
- Testing layers 2-14 with various configurations
- 8 experiments running in parallel
- Focus: Early visual processing zones

**Queue:** 19 more waves planned
- Wave 2: Mid-layer exploration (8-15)
- Wave 3: Late-layer exploration (15-27)
- Wave 4-5: Extreme parameter values
- Wave 6+: Random exploration

## Autonomous Capabilities

### ✓ Automatic Error Recovery
- Monitors experiment logs for errors
- Restarts failed experiments automatically
- Tracks GPU utilization and hung processes

### ✓ Continuous Progress Tracking
- Checks completion status every 5 minutes
- Generates results summaries per wave
- Identifies promising configurations

### ✓ Smart Wave Management
- Launches new waves when previous complete
- Tests different parameter combinations
- Stops at 8AM or after 20 waves

### ✓ Comprehensive Logging
- All experiments logged individually
- Master log tracks overnight progress
- Autonomous monitor creates status reports

## Expected Outcomes

### Total Experiments: ~160 configurations
- 8 experiments × 20 waves
- Testing: layer ranges, head percentages, alphas, suppression
- All on 100 hard POPE samples

### Success Criteria
- Any configuration with >1% improvement over baseline
- Current baseline: 83.00% on hard samples
- All wave 1 results so far: 0.00% delta (confirming issue)

### Morning Deliverables
- Complete results summary (all waves)
- Top 20 configurations ranked
- Analysis of what worked (if anything)
- Recommendations for next steps

## Monitoring Commands

### Check overall status:
```bash
tail -f srf_exp_runs/results/overnight_master.log
```

### Check specific wave:
```bash
ls -la srf_exp_runs/results/wave_1/*/pope.json
```

### Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### View autonomous monitoring:
```bash
tail -f srf_exp_runs/results/parallel_hard_pope/autonomous_night.log
```

## Error Handling

If something crashes:
- ✓ Automatic restart by autonomous monitor
- ✓ New experiments launched to replace failed ones
- ✓ Logs preserved for debugging
- ✓ Master script continues queue

## Notes

- All experiments use LLaVA-1.5-7B (stable, compatible)
- 100 hard samples per experiment (reproducible)
- Following existing repo patterns for conda/GPU usage
- No user intervention required until morning

---

**Next Update:** ~8AM CEST or when 20 waves complete
**Status:** 🔴 RUNNING AUTONOMOUSLY - DO NOT INTERRUPT
