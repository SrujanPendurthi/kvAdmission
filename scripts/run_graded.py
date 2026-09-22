"""Graded eviction vs uniform quantisation, at equal bits, on the needle task."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, time
from pathlib import Path

from kvadmission.graded import GradedCache, graded_prefill
from kvadmission.runner import load_model
from kvadmission.stream import build_stream, query_needle

# name -> (policy, write level). uniform0 is today's fp16 FIFO baseline.
CONFIGS = {"uniform0": ("uniform", 0), "uniform1": ("uniform", 1),
           "uniform2": ("uniform", 2), "uniform3": ("uniform", 3),
           "age": ("age", 0), "kl": ("kl", 0)}

p = argparse.ArgumentParser()
p.add_argument("--configs", nargs="+", default=list(CONFIGS))
p.add_argument("--budgets", type=int, nargs="+", default=[512, 1024])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--eps", type=float, default=0.0)
p.add_argument("--refresh", type=int, default=8)
p.add_argument("--per-token-keys", action="store_true")
p.add_argument("--out", default="results/graded.json")
a = p.parse_args()

model, tok, device = load_model()
out = Path(a.out)
rows = json.loads(out.read_text()) if out.exists() else []
done = {(r["config"], r["budget"], r["seed"]) for r in rows}

for seed in range(a.seeds):
    ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
    for b in a.budgets:
        for name in a.configs:
            if (name, b, seed) in done:
                continue
            policy, level = CONFIGS[name]
            t = time.time()
            cache = GradedCache(budget=b, sinks=4,
                                key_per_channel=not a.per_token_keys)
            n = graded_prefill(model, ids, cache, device, policy,
                               write_level=level, eps=a.eps, refresh=a.refresh)
            levels = cache.level.bincount(minlength=4).tolist()
            for nd in needles:
                hit, text = query_needle(model, tok, cache, n, nd, device)
                rows.append({"config": name, "budget": b, "seed": seed,
                             "topic": nd["topic"], "distance_back": n - nd["pos"],
                             "stream_len": n, "hit": bool(hit), "gen": text[:40],
                             "resident": cache.level.shape[0], "levels": levels,
                             "used_frac": cache.used() / cache.capacity})
            got = sum(r["hit"] for r in rows[-a.needles:])
            print(f"seed={seed} B={b} {name:9s} {got}/{a.needles} "
                  f"resident={cache.level.shape[0]} levels={levels} "
                  f"{time.time() - t:.0f}s", flush=True)
            out.write_text(json.dumps(rows, indent=1))
