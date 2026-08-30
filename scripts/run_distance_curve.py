"""Recall against distance at fine resolution, instead of three coarse bands.

Terciles were a convenience. This plants many needles per stream and reports
recall in narrow distance bins, which locates the boundary rather than
assuming one, and shows whether the near-field falloff is a cliff or a slope.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from collections import defaultdict
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.policies import make_decider
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--policies", nargs="+", default=["recent", "h2o"])
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--needles", type=int, default=16)   # 2x the tercile sweep
p.add_argument("--seeds", type=int, default=4)
p.add_argument("--bins", type=int, default=8)
p.add_argument("--out", default="results/distance-curve.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
rows = []
for policy in a.policies:
    for seed in range(a.seeds):
        ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
        decide, need_attn = make_decider(policy, budget=a.budget, sinks=4,
                                         seed=seed)
        cache = AdmissionCache(budget=a.budget, sinks=4)
        n = prefill(model, ids, cache, device, chunk=128, decide=decide,
                    need_attn=need_attn)
        for nd in needles:
            hit, _ = query_needle(model, tok, cache, n, nd, device)
            rows.append({"policy": policy, "budget": a.budget, "seed": seed,
                         "distance_back": n - nd["pos"], "stream_len": n,
                         "hit": bool(hit)})
        print(f"{policy:8s} seed={seed} "
              f"{sum(r['hit'] for r in rows[-a.needles:])}/{a.needles}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\n=== recall vs distance ===")
width = a.length / a.bins
for policy in a.policies:
    print(f"\n{policy}:")
    binned = defaultdict(list)
    for r in rows:
        if r["policy"] != policy:
            continue
        binned[min(int(r["distance_back"] // width), a.bins - 1)].append(r["hit"])
    for b in range(a.bins):
        v = binned.get(b, [])
        lo, hi = int(b * width), int((b + 1) * width)
        bar = "#" * round(20 * (sum(v) / len(v))) if v else ""
        rate = f"{sum(v)/len(v):.2f}" if v else "  - "
        print(f"  d {lo:>5}-{hi:<5} n={len(v):<3} {rate} {bar}")
