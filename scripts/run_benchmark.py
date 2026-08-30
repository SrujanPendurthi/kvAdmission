"""RULER-style QA sweep: recall by distance band on real passages.

Same axes as `kvadmission.sweep`, but the stream is distractor SQuAD
paragraphs, the query is a human-written question, and scoring is token F1
against the gold span rather than exact match on a digit string. `--oracle`
adds the retention control, flagging exactly the gold paragraphs.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

from kvadmission.benchmark import build_qa_stream, qa_spans, query_qa
from kvadmission.cache import AdmissionCache
from kvadmission.oracle import make_oracle_decider
from kvadmission.policies import make_decider
from kvadmission.runner import load_model, prefill
from kvadmission.sweep import _band

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--policies", nargs="+",
               default=["full", "recent", "h2o", "snapkv", "random"])
p.add_argument("--budgets", type=int, nargs="+", default=[512, 2048])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--items", type=int, default=8)
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--oracle", action="store_true")
p.add_argument("--out", default="results/benchmark.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
n_layers = model.config.num_hidden_layers
rows = []

cells = [(pol, b) for pol in a.policies
         for b in ([a.length] if pol == "full" else a.budgets)]
if a.oracle:
    cells += [("oracle", b) for b in a.budgets]

for seed in range(a.seeds):
    ids, items = build_qa_stream(tok, a.items, a.length, seed=seed)
    for policy, budget in cells:
        if policy == "oracle":
            # make_oracle_decider returns (decide, need_attn) like make_decider
            decide, need_attn = make_oracle_decider(
                qa_spans(items), budget=budget, n_layers=n_layers,
                sinks=4, device=device)
        else:
            decide, need_attn = make_decider(policy, budget=budget, sinks=4,
                                             seed=seed, chunk=a.chunk)
        cache = AdmissionCache(budget=None if policy == "full" else budget,
                               sinks=4)
        n = prefill(model, ids, cache, device, chunk=a.chunk, decide=decide,
                    need_attn=need_attn)

        f1s = []
        for it in items:
            hit, f1, em, text = query_qa(model, tok, cache, n, it, device)
            f1s.append(f1)
            rows.append({"policy": policy, "budget": budget, "seed": seed,
                         "question": it["question"][:70],
                         "answers": it["answers"],
                         "distance_back": n - it["pos"], "stream_len": n,
                         "hit": bool(hit), "f1": round(f1, 3), "em": em,
                         "gen": text.split("\n")[0][:70]})
        print(f"seed={seed} {policy:8s} B={budget:<5d} "
              f"hit={sum(r['hit'] for r in rows[-a.items:])}/{a.items} "
              f"F1={statistics.mean(f1s):.3f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\nrecall by band (hit rate, mean F1)")
agg = defaultdict(lambda: {"hit": [], "f1": []})
for r in rows:
    k = (r["policy"], r["budget"], _band(r["distance_back"], r["stream_len"]))
    agg[k]["hit"].append(float(r["hit"]))
    agg[k]["f1"].append(r["f1"])
for k in sorted(agg):
    v = agg[k]
    n = len(v["hit"])
    hits = sum(v["hit"])
    up = "" if hits else f"  95% upper {min(1.0, 3.0 / n):.2f}"
    print(f"  {k[0]:8s} B={k[1]:<5d} {k[2]:<7s} "
          f"hit {hits:.0f}/{n}  F1 {statistics.mean(v['f1']):.3f}{up}")
