"""Screen the degraded oracles, replaying the flags their end-to-end runs used.

They are the only candidates whose end-to-end recall is already measured
across the whole coverage range, so they test the screen in both directions:
the coverage-1.00 oracle should pass it, the degraded ones should not.
Tokeniser only; no model is loaded.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, statistics
from collections import defaultdict

from transformers import AutoTokenizer

from kvadmission.oracle import needle_spans, oracle_flags
from kvadmission.screen import coverage
from kvadmission.stream import build_stream

ROOT = pathlib.Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
runs = json.load(open(ROOT / "results/partial-oracle.json"))
configs = sorted({(r["coverage"], r["fp_rate"]) for r in runs})
seeds = sorted({r["seed"] for r in runs})
B, L, NEEDLES = runs[0]["budget"], runs[0]["stream_len"], 8

rows = []
for c, fp in configs:
    covs = []
    for seed in seeds:
        ids, needles = build_stream(tok, NEEDLES, L, seed=seed)
        spans = needle_spans(tok, needles)
        n = ids.shape[1]
        flag = oracle_flags(spans, n, coverage=c, fp_rate=fp, seed=seed)
        # the decider's own ranking: flagged first, recency as the tiebreak
        import torch
        score = torch.arange(n, dtype=torch.float32) + 1e6 * flag.float()
        covs += coverage(score, spans, B, n)
    rows.append({"coverage": c, "fp_rate": fp, "budget": B,
                 "mean_coverage": round(statistics.mean(covs), 4),
                 "spans_whole": sum(x == 1.0 for x in covs), "n_spans": len(covs)})
    print(rows[-1])
(ROOT / "results/screen-partial-oracle.json").write_text(json.dumps(rows, indent=1))
