"""Contiguity control: does the oracle's shortfall come from a spliced cache?

The oracle reached 40/40 on Qwen3-1.7B and only 12/24 on SmolLM2-1.7B. The
hypothesis in the writeup is *contiguity*: the oracle's retained set is one
short run per answer span, spliced between recency fill with gaps in the
position indices, and the two models may differ in whether they can read that.

The control holds the evidence fixed and widens the window kept around it.
Each answer span is dilated by a radius, so the same needles sit inside
progressively longer contiguous blocks at the same budget.

WHAT THIS DOES AND DOES NOT VARY. With 8 needles spread over 8192 tokens the
runs are ~1024 apart, so dilation does not merge them until the radius is
large enough that the flagged set is already several times the budget. Run
count therefore stays at 8 across the usable range, and the variable is
fragment *length* (a 16-token island versus a ~200-token window), not fragment
*count*. That is the more plausible reading of the hypothesis in any case: a
bare answer span carries no surrounding context, and a model may need the
sentence around a fact to use it. Testing run count directly would need a
different manipulation - fewer needles with wider spans at matched total
tokens - which changes the number of questions and is not run here.

Two things must be read together, and the script prints both:

  runs     the independent variable - contiguous blocks in the flagged set
  flagged  total flagged tokens; once this exceeds B the crowding law
           (recall = min(1, B/|flagged|)) takes over and recall falls for a
           reason that has nothing to do with fragmentation

A rise in recall with radius, while flagged stays under budget, is evidence
for the hypothesis. A flat curve is evidence against it. The largest radius is
deliberately over budget, so the curve should turn over there for crowding
reasons - which doubles as another check on the crowding law.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from collections import defaultdict
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.oracle import (dilate_spans, make_oracle_decider,
                                needle_spans, span_stats)
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle
from kvadmission.sweep import _band

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
# 192 is deliberately over budget at B=2048, so the curve should turn over
p.add_argument("--radii", type=int, nargs="+", default=[0, 16, 48, 96, 192])
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--out", default="results/contiguity.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
n_layers = model.config.num_hidden_layers
rows = []

for seed in range(a.seeds):
    ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
    base = needle_spans(tok, needles)
    for r in a.radii:
        spans = dilate_spans(base, r, a.length)
        runs, flagged = span_stats(spans)
        decide, need_attn = make_oracle_decider(
            spans, budget=a.budget, n_layers=n_layers, sinks=4, device=device)
        cache = AdmissionCache(budget=a.budget, sinks=4)
        n = prefill(model, ids, cache, device, chunk=a.chunk, decide=decide,
                    need_attn=need_attn)
        hits = 0
        for nd in needles:
            hit, text = query_needle(model, tok, cache, n, nd, device)
            hits += bool(hit)
            rows.append({"radius": r, "runs": runs, "flagged": flagged,
                         "budget": a.budget, "seed": seed, "topic": nd["topic"],
                         "distance_back": n - nd["pos"], "stream_len": n,
                         "hit": bool(hit), "gen": text[:50]})
        over = "  OVER BUDGET" if flagged > a.budget else ""
        print(f"seed={seed} r={r:<4d} runs={runs:<2d} flagged={flagged:<5d} "
              f"{hits}/{a.needles}{over}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\nrecall vs fragmentation (budget fixed)")
agg = defaultdict(list)
meta = {}
for r in rows:
    agg[r["radius"]].append(r["hit"])
    meta[r["radius"]] = (r["runs"], r["flagged"])
for rad in sorted(agg):
    v = agg[rad]
    runs, flagged = meta[rad]
    over = " (over budget)" if flagged > a.budget else ""
    print(f"  r={rad:<4d} runs={runs:<2d} flagged={flagged:<5d} "
          f"recall {sum(v)}/{len(v)} = {sum(v)/len(v):.2f}{over}")

print("\nmiddle band only")
band = defaultdict(list)
for r in rows:
    if _band(r["distance_back"], r["stream_len"]) == "middle":
        band[r["radius"]].append(r["hit"])
for rad in sorted(band):
    v = band[rad]
    print(f"  r={rad:<4d} {sum(v)}/{len(v)} = {sum(v)/len(v):.2f}")
