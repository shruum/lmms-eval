import json, os, re

def extract_srf_params(log_file):
    """Extract SRF parameters from log file"""
    if not os.path.exists(log_file):
        return "---", "---", "---", "---"

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract from the SRF reset line which has all parameters
    srf_line = re.search(r'\[SRF\] reset.*?alpha=(\d+\.?\d*).*?eps=(\d+\.?\d*).*?layers=\[(\d+),(\d+)\].*?head_topk=(\d+\.?\d*)', content)
    if srf_line:
        alpha = srf_line.group(1)
        eps = srf_line.group(2)
        layer_start = srf_line.group(3)
        layer_end = srf_line.group(4)
        heads_pct = int(float(srf_line.group(5)) * 100)
        return alpha, f"{layer_start}-{layer_end}", f"{heads_pct}%", eps

    return "---", "---", "---", "---"

baseline_coco = 79.30
baseline_gqa = 75.50

print("="*110)
print("FINAL COMPREHENSIVE SRF RESULTS TABLE")
print("="*110)

print("\n🔴 COCO ADVERSARIAL - ALL EXPERIMENTS (Baseline: {:.2f}%, Target: {:.2f}%)".format(baseline_coco, baseline_coco + 1))
print("-" * 110)
print(f"{'Config':<8} {'Alpha':<8} {'Layers':<10} {'Heads':<8} {'Eps':<6} {'Accuracy':<10} {'Delta':<10} {'Status':<8}")
print("-" * 110)

coco_results = []
# Round 1 (0-16) + Diagnostic (200-211)
for i in list(range(0, 17)) + list(range(200, 212)):
    result_file = f'results/srf_focused_sweep/coco_adversarial_config{i}/pope_coco_adversarial.json'
    log_file = f'results/srf_focused_sweep/coco_adversarial_config{i}/run.log'

    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
            acc = data['method']['0.0']['accuracy'] * 100
            alpha, layers, heads, eps = extract_srf_params(log_file)
            delta = acc - baseline_coco

            if acc >= baseline_coco + 1:
                status = "✅ TARGET"
            elif acc >= baseline_coco:
                status = "⚠️ CLOSE"
            else:
                status = "❌ FAIL"

            coco_results.append((i, acc, alpha, layers, heads, eps, delta))
            print(f"{i:<8} {alpha:<8} {layers:<10} {heads:<8} {eps:<6} {acc:<10.2f}% {delta:+<10.2f}% {status:<8}")

print("\n🔵 GQA ADVERSARIAL - ALL EXPERIMENTS (Baseline: {:.2f}%, Target: {:.2f}%)".format(baseline_gqa, baseline_gqa + 1))
print("-" * 110)
print(f"{'Config':<8} {'Alpha':<8} {'Layers':<10} {'Heads':<8} {'Eps':<6} {'Accuracy':<10} {'Delta':<10} {'Status':<8}")
print("-" * 110)

gqa_results = []
for i in range(400, 432):
    result_file = f'results/srf_focused_sweep/gqa_adversarial_config{i}/pope_gqa_adversarial.json'
    log_file = f'results/srf_focused_sweep/gqa_adversarial_config{i}/run.log'

    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
            acc = data['method']['0.0']['accuracy'] * 100
            if acc > 50:  # Filter corrupted results
                alpha, layers, heads, eps = extract_srf_params(log_file)
                delta = acc - baseline_gqa

                if acc >= baseline_gqa + 1:
                    status = "✅ TARGET"
                elif acc >= baseline_gqa:
                    status = "⚠️ CLOSE"
                else:
                    status = "❌ FAIL"

                gqa_results.append((i, acc, alpha, layers, heads, eps, delta))
                print(f"{i:<8} {alpha:<8} {layers:<10} {heads:<8} {eps:<6} {acc:<10.2f}% {delta:+<10.2f}% {status:<8}")

print("\n" + "="*110)
print("SUMMARY & KEY INSIGHTS")
print("="*110)

if coco_results:
    coco_results.sort(key=lambda x: x[1], reverse=True)
    print("\n🔴 COCO RESULTS ({} experiments):".format(len(coco_results)))
    print("   Baseline: {:.2f}%, Target: {:.2f}%".format(baseline_coco, baseline_coco + 1))
    cfg_id, acc, alpha, layers, heads, eps, delta = coco_results[0]
    print("   Best: Config {} (α={}): {:.2f}% ({:+.2f}%)".format(cfg_id, alpha, acc, delta))
    gap_to_target = (baseline_coco + 1) - acc
    print("   Gap to target: {:.2f}%".format(gap_to_target))

    print("\n   Top 5 COCO configurations:")
    for j, (cfg, acc, alpha, layers, heads, eps, delta) in enumerate(coco_results[:5], 1):
        print("   {}. Config {}: {:.2f}% (α={}, layers={}, heads={}, eps={})".format(j, cfg, acc, alpha, layers, heads, eps))

if gqa_results:
    gqa_results.sort(key=lambda x: x[1], reverse=True)
    print("\n🔵 GQA RESULTS ({} experiments):".format(len(gqa_results)))
    print("   Baseline: {:.2f}%, Target: {:.2f}%".format(baseline_gqa, baseline_gqa + 1))
    cfg_id, acc, alpha, layers, heads, eps, delta = gqa_results[0]
    print("   Best: Config {} (α={}): {:.2f}% ({:+.2f}%)".format(cfg_id, alpha, acc, delta))
    gap_to_target = (baseline_gqa + 1) - acc
    print("   Gap to target: {:.2f}%".format(gap_to_target))

    print("\n   Top 3 GQA configurations:")
    for j, (cfg, acc, alpha, layers, heads, eps, delta) in enumerate(gqa_results[:3], 1):
        print("   {}. Config {}: {:.2f}% (α={}, layers={}, heads={}, eps={})".format(j, cfg, acc, alpha, layers, heads, eps))

print("\n" + "="*110)
print("CONCLUSION")
print("="*110)
print("❌ SRF FAILED to achieve +1% improvement over baseline on any dataset")
print("❌ SRF FAILED to even match baseline on GQA (-6% gap)")
print("⚠️ SRF came close on COCO (78.77% vs 79.30%, only -0.53% gap)")
print("🔍 Hyperparameter tuning exhausted - 31+ configurations tested")
print("💡 Recommendation: Investigate SRF implementation bugs or try alternative approaches")
print("="*110)