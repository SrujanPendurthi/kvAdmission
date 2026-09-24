"""Does the signal that passed the screen actually work end to end?

The probe of `kvadmission.probe` keeps 64/64 needle spans whole under the
coverage screen. The screen only ever predicted; this runs the prediction:
probe-scored eviction against recency at the same budget, on seeds the probe
never saw. Optionally with int2 storage, which stretches the horizon (phase 10
stage 2), to see whether the two gains compose.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, time
from pathlib import Path

import torch

from kvadmission.cache import AdmissionCache
from kvadmission.graded import GradedCache, graded_prefill
from kvadmission.oracle import needle_spans
from kvadmission.policies import make_decider
from kvadmission.probe import features, fit, labels, make_score_decider
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=14)
p.add_argument("--train-seeds", type=int, default=6)
p.add_argument("--test-seeds", type=int, nargs="+",
               default=[100, 101, 102])
p.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
p.add_argument("--policies", nargs="+",
               default=["probe", "recent", "probe-int2"])
p.add_argument("--length", type=int, default=8192)
p.add_argument("--needles", type=int, default=8)
p.add_argument("--feat-dir", default="results/_probe_feats")
p.add_argument("--out", default="results/probe-e2e.json")
a = p.parse_args()

model, tok, device = load_model()
n_layers = model.config.num_hidden_layers
feat_dir = Path(a.feat_dir)
feat_dir.mkdir(parents=True, exist_ok=True)


def cached_features(seed, ids):
    """Reuse the feature files run_probe_scorer.py already wrote."""
    f = feat_dir / f"needle-{seed}-L{a.layer}.pt"
    if not f.exists():
        torch.save(features(model, ids, device, a.layer), f)
    return torch.load(f).float()


xs, ys = [], []
for s in range(a.train_seeds):
    ids, needles = build_stream(tok, a.needles, a.length, seed=s)
    x = cached_features(s, ids)
    xs.append(x)
    ys.append(labels(needle_spans(tok, needles), x.shape[0]))
score = fit(torch.cat(xs), torch.cat(ys))
print(f"probe trained on seeds 0..{a.train_seeds - 1}", flush=True)

out = Path(a.out)
rows = json.loads(out.read_text()) if out.exists() else []
done = {(r["policy"], r["budget"], r["seed"]) for r in rows}

for seed in a.test_seeds:
    ids, needles = build_stream(tok, a.needles, a.length, seed=seed)
    probe_scores = score(cached_features(seed, ids))
    for b in a.budgets:
        for policy in a.policies:
            if (policy, b, seed) in done:
                continue
            t = time.time()
            if policy == "probe-int2":
                cache = GradedCache(budget=b, sinks=4)
                n = graded_prefill(model, ids, cache, device, "uniform",
                                   write_level=3, scores=probe_scores)
            else:
                cache = AdmissionCache(budget=b, sinks=4)
                decide, need_attn = (
                    make_score_decider(probe_scores, b, n_layers) if policy == "probe"
                    else make_decider(policy, budget=b, sinks=4, seed=seed))
                n = prefill(model, ids, cache, device, decide=decide,
                            need_attn=need_attn)
            for nd in needles:
                hit, text = query_needle(model, tok, cache, n, nd, device)
                rows.append({"policy": policy, "budget": b, "seed": seed,
                             "topic": nd["topic"], "distance_back": n - nd["pos"],
                             "stream_len": n, "hit": bool(hit), "gen": text[:40]})
            got = sum(r["hit"] for r in rows[-a.needles:])
            mid = [r for r in rows[-a.needles:]
                   if r["stream_len"] / 3 <= r["distance_back"] < 2 * r["stream_len"] / 3]
            print(f"seed={seed} B={b} {policy:11s} {got}/{a.needles} "
                  f"mid {sum(r['hit'] for r in mid)}/{len(mid)} "
                  f"{time.time() - t:.0f}s", flush=True)
            out.write_text(json.dumps(rows, indent=1))
