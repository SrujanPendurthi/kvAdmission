"""How good would a learned proxy have to be? Degrade the oracle and measure."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.oracle import make_oracle_decider, needle_spans
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--out", default="results/partial-oracle.json")
p.add_argument("--fp", type=float, nargs="*", default=None,
               help="if given, sweep these fp_rates at coverage=1.0 only")
a = p.parse_args()

# (coverage, fp_rate). Needle tokens are ~1.2% of a stream. The flagged set is
# roughly fp_rate * length, so with length 8192 and budget 2048 the crowding
# threshold is fp_rate ~= 0.25 - below that every flagged token fits and
# precision costs nothing, above it false positives evict true ones.
CONFIGS = ([(1.00, f) for f in a.fp] if a.fp else [
    (1.00, 0.00), (0.75, 0.00), (0.50, 0.00), (0.25, 0.00),
    (1.00, 0.01), (1.00, 0.05), (1.00, 0.20),
])

model, tok, device = load_model()
rows = []
for coverage, fp in CONFIGS:
    for seed in range(a.seeds):
        ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
        decide, _ = make_oracle_decider(
            needle_spans(tok, needles), a.budget,
            n_layers=model.config.num_hidden_layers, sinks=4,
            coverage=coverage, fp_rate=fp, seed=seed)
        cache = AdmissionCache(budget=a.budget, sinks=4)
        n = prefill(model, ids, cache, device, chunk=128, decide=decide)
        for nd in needles:
            hit, text = query_needle(model, tok, cache, n, nd, device)
            rows.append({"policy": f"cov{coverage}_fp{fp}", "coverage": coverage,
                         "fp_rate": fp, "budget": a.budget, "seed": seed,
                         "topic": nd["topic"], "distance_back": n - nd["pos"],
                         "stream_len": n, "hit": bool(hit),
                         "flagged_est": int(fp * a.length + 96)})
        got = sum(r["hit"] for r in rows[-a.needles:])
        print(f"coverage={coverage:.2f} fp={fp:.2f} seed={seed}  {got}/{a.needles}",
              flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\n=== recall by proxy quality ===")
for coverage, fp in CONFIGS:
    sel = [r for r in rows if r["coverage"] == coverage and r["fp_rate"] == fp]
    mid = [r for r in sel if r["stream_len"]/3 <= r["distance_back"] < 2*r["stream_len"]/3]
    print(f"coverage={coverage:.2f} fp={fp:.2f}  overall {sum(r['hit'] for r in sel)}/{len(sel)}"
          f"   middle-band {sum(r['hit'] for r in mid)}/{len(mid)}")
