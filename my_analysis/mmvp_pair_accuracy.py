#!/usr/bin/env python3
"""Compute MMVP pair accuracy from results.json.
Pair accuracy = fraction of pairs where BOTH images answered correctly.
Usage: python mmvp_pair_accuracy.py results/new_datasets/mmvp/base/results.json
"""
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/new_datasets/mmvp/base/results.json"
with open(path) as f:
    recs = json.load(f)

mmvp = [r for r in recs if r.get("dataset","mmvp") in ("mmvp","")]

by_pair_method_val = defaultdict(list)
for r in mmvp:
    key = (r["group"], r.get("method","baseline"), r.get("value", 1.0))
    by_pair_method_val[key].append(r["correct"])

# Group by (method, value) → list of pair results
by_mv = defaultdict(list)
for (pair, method, val), corrects in by_pair_method_val.items():
    pair_ok = all(corrects)
    by_mv[(method, val)].append(pair_ok)

print(f"\n{'Method':20s} {'Value':6s}  {'PairAcc':>8s}  {'ImgAcc':>8s}  Pairs")
print("-" * 60)
for (method, val), pair_oks in sorted(by_mv.items()):
    pair_acc = sum(pair_oks) / len(pair_oks)
    # image accuracy: flatten all individual corrects
    img_corrects = []
    for (pair, m, v), corrects in by_pair_method_val.items():
        if m == method and v == val:
            img_corrects.extend(corrects)
    img_acc = sum(img_corrects) / len(img_corrects) if img_corrects else 0
    print(f"  {method:20s} {val:6.1f}  {pair_acc:8.1%}  {img_acc:8.1%}  {len(pair_oks)}")
