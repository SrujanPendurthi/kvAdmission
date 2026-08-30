"""Closed-book floor: can the model answer without any context at all?

A benchmark scored by containment needs this control. Short common-word golds
("rivers") can be guessed from the question alone, and a guess is
indistinguishable from a retrieval in the scoring. Any policy's recall must be
read against this floor, not against zero.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, statistics
from pathlib import Path

from kvadmission.benchmark import _SHOT, build_qa_stream, score
from kvadmission.cache import AdmissionCache
from kvadmission.runner import generate, load_model, prefill

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--items", type=int, default=8)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--out", default="results/closed-book.json")
a = p.parse_args()

model, tok, device = load_model(a.model)
rows = []
for seed in range(a.seeds):
    _, items = build_qa_stream(tok, a.items, a.length, seed=seed)
    # a cache holding nothing but the few-shot prompt: no haystack, no evidence
    for it in items:
        # prefill the few-shot only, then ask - generate() needs a non-empty
        # prompt, and this keeps the question in the same position it occupies
        # in the real sweep
        cache = AdmissionCache(budget=None, sinks=4)
        ids = tok(_SHOT, add_special_tokens=False, return_tensors="pt")["input_ids"]
        n = prefill(model, ids, cache, device, chunk=128)
        text = generate(model, tok, cache, n,
                        f"Question: {it['question']}\nAnswer:", device)
        pred = text.split("\n")[0].strip()
        f1, em = score(pred, it["answers"])
        rows.append({"seed": seed, "question": it["question"][:70],
                     "answers": it["answers"], "hit": bool(em),
                     "f1": round(f1, 3), "gen": pred[:70]})
        if em:
            print(f"  GUESSED seed={seed} gold={it['answers'][0]!r} gen={pred[:50]!r}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(rows, indent=1))
h = sum(r["hit"] for r in rows)
print(f"\nclosed-book floor: {h}/{len(rows)} = {h/len(rows):.3f}  "
      f"meanF1 {statistics.mean(r['f1'] for r in rows):.3f}")
