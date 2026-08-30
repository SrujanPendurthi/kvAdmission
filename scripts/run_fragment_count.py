"""Fragment count at matched flagged tokens: is it the gaps?

The remaining half of the contiguity hypothesis. `run_contiguity.py` showed
fragment LENGTH does not matter - widening a bare answer span into a coherent
200-token block moved SmolLM2 by 0.8 standard errors. But it could not vary
fragment COUNT, because with 8 needles ~1024 tokens apart dilation never merges
runs before the flagged set overruns the budget.

This isolates count directly. The stream and its 8 needles are identical in
every condition, so distances are held fixed. What varies is how many of them
the oracle flags, with the window around each widened so the TOTAL flagged
token count stays constant:

    k=8  ->  8 runs, narrow windows      8 gaps in the position index
    k=4  ->  4 runs, wider windows       4 gaps
    k=2  ->  2 runs, wider still         2 gaps
    k=1  ->  1 run,  widest              1 gap

Only the flagged needles are queried; the unflagged ones are not part of the
measurement. Seeds per condition are scaled so every k gets a comparable number
of trials, since k=1 yields one query per stream and k=8 yields eight.

If recall rises as k falls at constant flagged tokens, the number of position-
index gaps is what hurts, and the contiguity hypothesis survives in its second
form. If it is flat, contiguity is dead in both forms and the SmolLM2 shortfall
needs a different explanation.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, random
from collections import defaultdict
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.oracle import (expand_spans_to_quota,
                                make_oracle_decider, needle_spans,
                                span_stats)
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle
from kvadmission.sweep import _band

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--ks", type=int, nargs="+", default=[8, 4, 2, 1])
p.add_argument("--flagged-target", type=int, default=1600)
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--trials-per-k", type=int, default=24)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--out", default="results/fragment-count.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
n_layers = model.config.num_hidden_layers
rows = []

for k in a.ks:
    seeds = max(1, round(a.trials_per_k / k))   # equalise trial counts across k
    for seed in range(seeds):
        ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
        base = needle_spans(tok, needles)

        # pick which k needles to flag; vary the choice with the seed so the
        # low-k conditions are not pinned to one set of distances
        rng = random.Random(1000 * k + seed)
        idx = sorted(rng.sample(range(a.needles), k))
        chosen = [base[i] for i in idx]

        # widen each window so the flagged total matches the target exactly.
        # Symmetric dilation would clamp at the stream edges and leave the
        # low-k conditions short, which would confound gap count with how much
        # evidence and recency fill each condition gets.
        spans = expand_spans_to_quota(chosen, a.flagged_target // k, a.length)
        runs, flagged = span_stats(spans)
        assert runs == k, f"k={k} produced {runs} runs"

        decide, need_attn = make_oracle_decider(
            spans, budget=a.budget, n_layers=n_layers, sinks=4, device=device)
        cache = AdmissionCache(budget=a.budget, sinks=4)
        n = prefill(model, ids, cache, device, chunk=a.chunk, decide=decide,
                    need_attn=need_attn)

        hits = 0
        for i in idx:
            nd = needles[i]
            hit, text = query_needle(model, tok, cache, n, nd, device)
            hits += bool(hit)
            rows.append({"k": k, "runs": runs, "flagged": flagged,
                         "budget": a.budget, "seed": seed,
                         "topic": nd["topic"], "distance_back": n - nd["pos"],
                         "stream_len": n, "hit": bool(hit)})
        print(f"k={k} seed={seed:<2d} runs={runs} flagged={flagged:<5d} "
              f"{hits}/{k}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\nrecall vs number of position-index gaps (flagged tokens held constant)")
agg = defaultdict(list); meta = {}
for r in rows:
    agg[r["k"]].append(r["hit"]); meta[r["k"]] = (r["runs"], r["flagged"])
for k in sorted(agg, reverse=True):
    v = agg[k]; runs, flagged = meta[k]
    print(f"  k={k}  runs={runs}  flagged={flagged:<5d}  "
          f"{sum(v)}/{len(v)} = {sum(v)/len(v):.2f}")

print("\nmiddle band only")
band = defaultdict(list)
for r in rows:
    if _band(r["distance_back"], r["stream_len"]) == "middle":
        band[r["k"]].append(r["hit"])
for k in sorted(band, reverse=True):
    v = band[k]
    print(f"  k={k}  {sum(v)}/{len(v)} = {sum(v)/len(v):.2f}")
