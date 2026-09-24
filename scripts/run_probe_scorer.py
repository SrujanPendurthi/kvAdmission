"""A supervised write-time scorer: a linear probe on hidden states, screened.

Phases 2-2c learned admission by RL on a KL reward and reproduced recency.
This asks the direct version: given the answer spans as labels, can a linear
read of the frozen model's write-time hidden state find them? Scored by the
same coverage screen as every other candidate (`screen.coverage`), on seeds
never trained on and on the task never trained on.

Prediction from the specification: in-task needles pass (a needle has a
distinctive form), QA fails (the gold paragraph looks like its distractors;
which one matters depends on a question not yet asked), and cross-task fails.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from pathlib import Path

import torch

from kvadmission.benchmark import answer_spans, build_qa_stream, qa_spans
from kvadmission.oracle import needle_spans
from kvadmission.runner import load_model
from kvadmission.probe import features as hidden_states, fit, labels
from kvadmission.screen import coverage
from kvadmission.stats import wilson
from kvadmission.stream import build_stream

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=14)
p.add_argument("--train-seeds", type=int, default=6)
p.add_argument("--test-seeds", type=int, default=8)
p.add_argument("--budget", type=int, default=2048)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--items", type=int, default=8)
p.add_argument("--feat-dir", default="results/_probe_feats")
p.add_argument("--out", default="results/probe-scorer.json")
a = p.parse_args()

model, tok, device = load_model()
feat_dir = Path(a.feat_dir)
feat_dir.mkdir(parents=True, exist_ok=True)


def stream(task, seed):
    if task == "needle":
        ids, items = build_stream(tok, a.items, a.length, seed=seed)
        return ids, needle_spans(tok, items)
    ids, items = build_qa_stream(tok, a.items, a.length, seed=seed)
    keep = [i for i, (lo, hi) in enumerate(qa_spans(items)) if hi - lo > 1]
    spans = answer_spans(tok, items)
    return ids, [spans[i] for i in keep]


@torch.no_grad()
def features(task, seed):
    """Layer-`a.layer` hidden state per token, cached to disk between runs."""
    f = feat_dir / f"{task}-{seed}-L{a.layer}.pt"
    ids, spans = stream(task, seed)
    if not f.exists():
        torch.save(hidden_states(model, ids, device, a.layer), f)
        print(f"  features {task} seed={seed}", flush=True)
    x = torch.load(f).float()
    return x, labels(spans, x.shape[0]), spans


def train(task):
    xs, ys = zip(*[features(task, s)[:2] for s in range(a.train_seeds)])
    return fit(torch.cat(xs), torch.cat(ys))


rows = []
scorers = {t: train(t) for t in ("needle", "qa")}
for train_task, score in scorers.items():
    for test_task in ("needle", "qa"):
        covs, rand = [], []
        g = torch.Generator().manual_seed(0)
        for s in range(100, 100 + a.test_seeds):   # disjoint from train seeds
            x, _, spans = features(test_task, s)
            covs += coverage(score(x), spans, a.budget, x.shape[0])
            rand += coverage(torch.rand(x.shape[0], generator=g), spans,
                             a.budget, x.shape[0])
        whole, rwhole = sum(c == 1.0 for c in covs), sum(c == 1.0 for c in rand)
        lo, hi = wilson(whole, len(covs))
        print(f"train={train_task:6s} test={test_task:6s} spans whole "
              f"{whole}/{len(covs)} [{lo:.2f},{hi:.2f}]  random {rwhole}/{len(rand)}"
              f"  mean cov {sum(covs)/len(covs):.3f}", flush=True)
        rows.append({"train": train_task, "test": test_task, "layer": a.layer,
                     "budget": a.budget, "spans_whole": whole,
                     "n_spans": len(covs), "random_whole": rwhole,
                     "mean_coverage": round(sum(covs) / len(covs), 4),
                     "per_span": [round(c, 4) for c in covs]})
Path(a.out).write_text(json.dumps(rows, indent=1))
