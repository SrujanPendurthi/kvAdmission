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

from kvadmission.benchmark import (answer_spans, build_qa_stream, qa_spans,
                                   query_qa)
from kvadmission.cache import AdmissionCache
from kvadmission.graded import GradedCache, graded_prefill
from kvadmission.oracle import needle_spans
from kvadmission.policies import make_decider
from kvadmission.probe import features, fit, labels, make_score_decider
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["needle", "qa"], default="needle")
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--train-style", default="code",
               help="needle answer style the probe is fitted on")
p.add_argument("--test-style", default=None,
               help="needle answer style it is tested on; defaults to --train-style")
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

model, tok, device = load_model(a.model)
tag = a.model.split("/")[-1]
a.test_style = a.test_style or a.train_style
n_layers = model.config.num_hidden_layers
feat_dir = Path(a.feat_dir)
feat_dir.mkdir(parents=True, exist_ok=True)


def build(seed, style):
    """One stream plus the spans to be retained, for either task."""
    if a.task == "needle":
        ids, items = build_stream(tok, a.needles, a.length, seed=seed,
                                  style=style)
        return ids, items, needle_spans(tok, items)
    ids, items = build_qa_stream(tok, a.needles, a.length, seed=seed)
    keep = [i for i, (lo, hi) in enumerate(qa_spans(items)) if hi - lo > 1]
    spans = answer_spans(tok, items)
    return ids, [items[i] for i in keep], [spans[i] for i in keep]


def cached_features(seed, ids, style):
    """Reuse the feature files run_probe_scorer.py already wrote."""
    stem = f"{a.task}-{seed}-L{a.layer}"
    if tag != "Qwen3-1.7B":
        stem += f"-{tag}"
    if style != "code":
        stem += f"-{style}"
    f = feat_dir / f"{stem}.pt"
    if not f.exists():
        torch.save(features(model, ids, device, a.layer), f)
    return torch.load(f).float()


xs, ys = [], []
for s in range(a.train_seeds):
    ids, _, spans = build(s, a.train_style)
    x = cached_features(s, ids, a.train_style)
    xs.append(x)
    ys.append(labels(spans, x.shape[0]))
score = fit(torch.cat(xs), torch.cat(ys))
print(f"probe trained on {tag} {a.task}/{a.train_style} seeds "
      f"0..{a.train_seeds - 1}, tested on {a.test_style}", flush=True)

out = Path(a.out)
rows = json.loads(out.read_text()) if out.exists() else []
done = {(r["policy"], r["budget"], r["seed"]) for r in rows}

for seed in a.test_seeds:
    ids, items, _ = build(seed, a.test_style)
    probe_scores = score(cached_features(seed, ids, a.test_style))
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
            for nd in items:
                if a.task == "needle":
                    hit, text = query_needle(model, tok, cache, n, nd, device)
                    f1 = float(hit)
                else:
                    hit, f1, _, text = query_qa(model, tok, cache, n, nd, device)
                rows.append({"policy": policy, "budget": b, "seed": seed,
                             "task": a.task, "model": tag,
                             "train_style": a.train_style,
                             "test_style": a.test_style, "f1": round(f1, 3),
                             "topic": nd.get("topic", nd.get("title")),
                             "distance_back": n - nd["pos"],
                             "stream_len": n, "hit": bool(hit), "gen": text[:40]})
            got = sum(r["hit"] for r in rows[-len(items):])
            mid = [r for r in rows[-len(items):]
                   if r["stream_len"] / 3 <= r["distance_back"] < 2 * r["stream_len"] / 3]
            print(f"seed={seed} B={b} {policy:11s} {got}/{len(items)} "
                  f"mid {sum(r['hit'] for r in mid)}/{len(mid)} "
                  f"{time.time() - t:.0f}s", flush=True)
            out.write_text(json.dumps(rows, indent=1))
