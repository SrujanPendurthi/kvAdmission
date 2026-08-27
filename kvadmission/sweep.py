"""Sweep driver: policies x budgets x seeds, summarized by distance band."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.policies import make_decider
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

BANDS = ("near", "middle", "far")


def _band(distance, stream_len):
    third = stream_len / 3
    if distance < third:
        return "near"
    return "middle" if distance < 2 * third else "far"


def run_cell(model, tok, device, policy, budget, seed, length, n_needles,
             chunk=128):
    """One policy/budget/seed combination. Returns one row per needle."""
    ids, needles = build_stream(tok, n_needles, length, seed=seed)
    decide, need_attn = make_decider(
        policy, budget=budget, sinks=4, seed=seed, chunk=chunk
    )
    cache = AdmissionCache(
        budget=None if policy == "full" else budget, sinks=4
    )
    stream_len = prefill(
        model, ids, cache, device, chunk=chunk, decide=decide,
        need_attn=need_attn,
    )

    rows = []
    for n in needles:
        hit, text = query_needle(model, tok, cache, stream_len, n, device)
        rows.append({
            "policy": policy, "budget": budget, "seed": seed,
            "topic": n["topic"], "distance_back": stream_len - n["pos"],
            "stream_len": stream_len, "hit": bool(hit), "gen": text[:60],
        })
    return rows


def summarize(rows):
    """Recall by (policy, budget, distance band), averaged across seeds.

    Banding uses each row's own `stream_len`. Inferring it from the observed
    distances would mis-band any subset that happens to contain only near-field
    rows, so the stream length travels with the data instead.
    """
    buckets = defaultdict(list)
    for r in rows:
        key = (r["policy"], r["budget"], _band(r["distance_back"], r["stream_len"]))
        buckets[key].append(1.0 if r["hit"] else 0.0)

    out = {}
    for key, vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        sd = statistics.stdev(vals) if n > 1 else 0.0
        out[key] = {"mean": mean, "n": n, "stderr": sd / (n ** 0.5) if n else 0.0}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--needles", type=int, default=8)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[256, 512, 1024, 2048])
    p.add_argument("--policies", nargs="+",
                   default=["full", "recent", "random", "h2o", "snapkv"])
    p.add_argument("--out", default="results/phase1.json")
    a = p.parse_args()

    model, tok, device = load_model(a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(a.seeds):
        for policy in a.policies:
            budgets = [a.length] if policy == "full" else a.budgets
            for b in budgets:
                rows += run_cell(model, tok, device, policy, b, seed,
                                 a.length, a.needles, a.chunk)
                hits = sum(r["hit"] for r in rows[-a.needles:])
                print(f"seed={seed} {policy:8s} B={b:<6d} "
                      f"{hits}/{a.needles}", flush=True)
                Path(a.out).write_text(json.dumps(rows, indent=1))

    print("\nrecall by band (mean +/- stderr)")
    for key, cell in sorted(summarize(rows).items()):
        policy, budget, band = key
        print(f"  {policy:8s} B={budget:<6d} {band:7s} "
              f"{cell['mean']:.2f} +/- {cell['stderr']:.2f}  (n={cell['n']})")


if __name__ == "__main__":
    main()
