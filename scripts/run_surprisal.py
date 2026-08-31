"""Surprisal admission on both task families, plus a coverage diagnostic.

The partial-oracle ablation showed coverage is the unforgiving half of the
proxy specification: at 0.75 coverage over answer tokens, middle-band recall
collapsed from 7/7 to 1/7. Surprisal satisfies the other half for free, since
top-B means the flagged set is exactly the budget.

So the question that decides this experiment is: does the top-B-by-surprisal
set contain EVERY token of each answer span? `--diagnose` answers that
directly, without generating anything, and is cheap enough to run on a laptop
before renting a GPU.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

import torch

from kvadmission.benchmark import build_qa_stream, qa_spans, query_qa
from kvadmission.cache import AdmissionCache
from kvadmission.oracle import needle_spans
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle
from kvadmission.surprisal import make_surprisal_decider
from kvadmission.sweep import _band

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--task", choices=["needle", "qa"], default="needle")
p.add_argument("--budgets", type=int, nargs="+", default=[2048])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--items", type=int, default=8)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--recency-weight", type=float, default=0.0)
p.add_argument("--diagnose", action="store_true",
               help="report coverage of answer spans by top-B surprisal only")
p.add_argument("--out", default="results/surprisal.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
n_layers = model.config.num_hidden_layers


def build(seed):
    if a.task == "needle":
        ids, items = build_stream(tok, a.items, a.length, seed=seed)
        return ids, items, needle_spans(tok, items)
    ids, items = build_qa_stream(tok, a.items, a.length, seed=seed)
    return ids, items, qa_spans(items)


@torch.no_grad()
def stream_surprisal(ids):
    """Per-position surprisal from an uncompressed pass, for the diagnostic."""
    flat = ids[0]
    cache = AdmissionCache(budget=None, sinks=4)
    out_s, prev = [], None
    for start in range(0, flat.shape[0], a.chunk):
        piece = ids[:, start:start + a.chunk].to(device)
        n_new = piece.shape[1]
        o = model(piece, past_key_values=cache,
                  position_ids=torch.arange(start, start + n_new,
                                            device=device).unsqueeze(0),
                  attention_mask=torch.ones(1, cache.get_seq_length() + n_new,
                                            device=device, dtype=torch.long),
                  use_cache=True)
        lg = o.logits[0].float()
        rows = lg[:-1]
        tgt = flat[start:start + n_new].to(lg.device)
        if prev is not None:
            rows = torch.cat([prev.unsqueeze(0), rows], dim=0)
        else:
            tgt = tgt[1:]
        s = -torch.log_softmax(rows, -1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        if prev is None:
            s = torch.cat([s.new_zeros(1), s])
        prev = lg[-1]
        out_s.append(s.cpu())
    return torch.cat(out_s)


if a.diagnose:
    print("coverage of answer spans by the top-B most surprising tokens\n")
    for B in a.budgets:
        cov_all, full_spans = [], 0
        n_spans = 0
        for seed in range(a.seeds):
            ids, items, spans = build(seed)
            s = stream_surprisal(ids)
            keep = set(torch.topk(s, B).indices.tolist())
            for lo, hi in spans:
                toks = list(range(lo, min(hi, s.shape[0])))
                if not toks:
                    continue
                c = sum(t in keep for t in toks) / len(toks)
                cov_all.append(c)
                full_spans += (c == 1.0)
                n_spans += 1
        print(f"  B={B:<5d} mean coverage {statistics.mean(cov_all):.3f}   "
              f"spans fully retained {full_spans}/{n_spans} "
              f"({full_spans/n_spans:.2f})")
    print("\nThe partial oracle put middle-band recall at 1/7 for 0.75 coverage,")
    print("so anything short of ~1.00 fully-retained predicts failure.")
    sys.exit(0)

rows = []
for seed in range(a.seeds):
    ids, items, _ = build(seed)
    for B in a.budgets:
        decide, need_attn = make_surprisal_decider(
            ids, budget=B, n_layers=n_layers, sinks=4, device=device,
            recency_weight=a.recency_weight)
        cache = AdmissionCache(budget=B, sinks=4)
        n = prefill(model, ids, cache, device, chunk=a.chunk, decide=decide,
                    need_attn=need_attn, need_logits=True)
        hits = 0
        for it in items:
            if a.task == "needle":
                hit, gen = query_needle(model, tok, cache, n, it, device)
                f1 = float(hit)
            else:
                hit, f1, _em, gen = query_qa(model, tok, cache, n, it, device)
            hits += bool(hit)
            rows.append({"policy": "surprisal", "task": a.task, "budget": B,
                         "seed": seed, "distance_back": n - it["pos"],
                         "stream_len": n, "hit": bool(hit),
                         "f1": round(f1, 3), "gen": str(gen)[:60]})
        print(f"seed={seed} surprisal B={B:<5d} {hits}/{a.items}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))

print("\nrecall by band")
agg = defaultdict(list)
for r in rows:
    agg[(r["budget"], _band(r["distance_back"], r["stream_len"]))].append(r["hit"])
for k in sorted(agg):
    v = agg[k]
    up = "" if sum(v) else f"  95% upper {min(1.0, 3.0/len(v)):.2f}"
    print(f"  B={k[0]:<5d} {k[1]:<7s} {sum(v)}/{len(v)} = {sum(v)/len(v):.2f}{up}")
