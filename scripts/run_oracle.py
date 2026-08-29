"""Oracle retention experiment: observability vs budget geometry."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path

import torch

from kvadmission.cache import AdmissionCache
from kvadmission.oracle import make_oracle_decider, needle_spans
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle
from kvadmission.sweep import summarize

p = argparse.ArgumentParser()
p.add_argument("--budgets", type=int, nargs="+", default=[512, 2048])
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--out", default="results/oracle.json")
a = p.parse_args()

model, tok, device = load_model()
rows = []
for budget in a.budgets:
    for seed in range(a.seeds):
        ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
        decide, _ = make_oracle_decider(
            needle_spans(tok, needles), budget,
            n_layers=model.config.num_hidden_layers, sinks=4, device=device)
        cache = AdmissionCache(budget=budget, sinks=4)
        n = prefill(model, ids, cache, device, chunk=128, decide=decide)
        for nd in needles:
            hit, text = query_needle(model, tok, cache, n, nd, device)
            rows.append({"policy": "oracle", "budget": budget, "seed": seed,
                         "topic": nd["topic"], "distance_back": n - nd["pos"],
                         "stream_len": n, "hit": bool(hit), "gen": text[:60]})
        print(f"seed={seed} oracle B={budget} "
              f"{sum(r['hit'] for r in rows[-a.needles:])}/{a.needles}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\nrecall by band (mean, 95% upper)")
for key, cell in sorted(summarize(rows).items()):
    pol, b, band = key
    print(f"  {pol:8s} B={b:<6d} {band:7s} {cell['mean']:.2f} "
          f"(<={cell['upper95']:.2f}, n={cell['n']})")
