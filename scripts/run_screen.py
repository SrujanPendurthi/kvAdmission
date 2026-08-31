"""Screen every candidate write-time signal against the coverage test.

Prints, per signal and per task, the fraction of answer-span tokens inside its
top B and the fraction of spans retained whole. §7.2's curve says anything
short of ~1.00 fully retained predicts end-to-end failure; phase 10 confirmed
that for surprisal, so this table is a set of predictions, not just statistics.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

from kvadmission.benchmark import build_qa_stream, qa_spans
from kvadmission.oracle import needle_spans
from kvadmission.runner import load_model
from kvadmission.screen import candidate_scores, coverage
from kvadmission.stream import build_stream

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--tasks", nargs="+", default=["needle", "qa"])
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--items", type=int, default=8)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--attn-layers", type=int, default=4)
p.add_argument("--out", default="results/screen.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
rows = []

for task in a.tasks:
    per_sig = defaultdict(list)
    for seed in range(a.seeds):
        if task == "needle":
            ids, items = build_stream(tok, a.items, a.length, seed=seed)
            spans = needle_spans(tok, items)
        else:
            ids, items = build_qa_stream(tok, a.items, a.length, seed=seed)
            spans = qa_spans(items)
        sc = candidate_scores(model, tok, ids, device,
                              attn_layers=a.attn_layers, seed=seed)
        for name, s in sc.items():
            per_sig[name] += coverage(s, spans, a.budget, ids.shape[1])
        print(f"  {task} seed={seed} done", flush=True)

    print(f"\n=== {task}, B={a.budget} (chance = {a.budget/a.length:.3f}) ===")
    print(f"  {'signal':16s} {'mean coverage':>14s} {'spans whole':>12s}")
    for name in sorted(per_sig, key=lambda k: -statistics.mean(per_sig[k])):
        c = per_sig[name]
        whole = sum(x == 1.0 for x in c)
        print(f"  {name:16s} {statistics.mean(c):>14.3f} {whole:>7d}/{len(c)}")
        rows.append({"task": task, "signal": name, "budget": a.budget,
                     "mean_coverage": round(statistics.mean(c), 4),
                     "spans_whole": whole, "n_spans": len(c)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

print("\n§7.2: 0.75 coverage already collapsed the middle band to 1/7.")
print("Anything short of ~1.00 spans-whole is predicted to fail end-to-end.")
