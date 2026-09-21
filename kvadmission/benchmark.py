"""RULER-style QA over real passages, with the evidence position controlled.

Everything else in this repo measures recall of a planted 6-digit code in
Pride and Prejudice. That is the easiest possible case for a cache and the
least like real long-context work, and exact substring match scores a
paraphrase as a miss. It is the load-bearing limitation in the writeup.

This module fixes three things at once while keeping the one axis the whole
result depends on:

  natural prose      the haystack is distractor SQuAD paragraphs, so the
                     filler is topically similar to the evidence rather than
                     19th-century fiction surrounding a digit string
  a real question    human-written, answerable only from the gold paragraph
  a real answer      a natural-language span scored by token F1, so a
                     paraphrase is partial credit instead of zero

  *and* known evidence position, by construction. LongBench-v2 is genuinely
  non-synthetic but ships no evidence spans, so it cannot be banded by
  distance - and recall against distance is the entire measurement here.

The gold paragraph is planted at a controlled anchor exactly as `build_stream`
plants a needle, so `sweep`, `oracle` and the distance banding all carry over.
"""

import json
import random
import re
import string
from collections import Counter
from pathlib import Path

import torch

from kvadmission.runner import generate

CACHE = Path(__file__).parent / "_squad.json"
N_CACHE = 4000  # paragraphs pulled once; a long stream needs a few hundred


def _load(limit=N_CACHE):
    """SQuAD paragraphs with their questions, cached to disk after one fetch."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())

    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")

    by_ctx, order = {}, []
    for row in ds:
        c = row["context"]
        if c not in by_ctx:
            if len(by_ctx) >= limit:
                continue
            by_ctx[c] = {"context": c, "title": row["title"], "qas": []}
            order.append(c)
        answers = list(dict.fromkeys(row["answers"]["text"]))
        if answers:
            by_ctx[c]["qas"].append({"question": row["question"],
                                     "answers": answers})

    items = [by_ctx[c] for c in order if by_ctx[c]["qas"]]
    CACHE.write_text(json.dumps(items))
    return items


# SQuAD golds contain typographic dashes and quotes that str.punctuation misses,
# e.g. "100-5,000 hp" uses U+2013. Leaving them in makes an otherwise exact match
# score zero.
_UNICODE_PUNCT = "\u2010\u2011\u2012\u2013\u2014\u2015\u2018\u2019\u201c\u201d\u2026"
_PUNCT = set(string.punctuation) | set(_UNICODE_PUNCT)


def _normalize(s):
    """SQuAD normalisation: lowercase, drop articles, punctuation, extra space."""
    s = s.lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    """Token-level F1, the standard SQuAD metric. Paraphrase gets partial credit."""
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred, answers):
    """Best F1 and best containment-EM over the acceptable answers."""
    f1 = max(f1_score(pred, a) for a in answers)
    em = float(any(_normalize(a) in _normalize(pred) for a in answers))
    return f1, em


def build_qa_stream(tok, n_items, target_len, seed=0):
    """Plant `n_items` gold paragraphs through `target_len` tokens of distractors.

    Returns (ids, items). Each item carries `question`, `answers`, `pos` (the
    token index just past the gold paragraph) and `span` (start, end), so the
    oracle can flag exactly the evidence and nothing else.
    """
    pool = _load()
    if n_items > len(pool) // 4:
        raise ValueError(f"only {len(pool)} paragraphs available")

    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)

    gold_idx = idx[:n_items]

    items = []
    for i in gold_idx:
        qa = rng.choice(pool[i]["qas"])
        items.append({"context": pool[i]["context"], "title": pool[i]["title"],
                      "question": qa["question"], "answers": qa["answers"]})

    # Distractors must not come from any gold paragraph's article. SQuAD groups
    # paragraphs by topic, so a same-article distractor can restate the answer -
    # a hit would then prove nothing about whether the planted evidence survived
    # compression, and the distance banding would be measuring noise.
    banned = {it["title"] for it in items}
    distractor_idx = [i for i in idx[n_items:] if pool[i]["title"] not in banned]
    if len(distractor_idx) < 8:
        raise ValueError("too few out-of-article distractors")

    # even anchors, jittered within their slot - same scheme as build_stream, so
    # seeds differ in placement rather than only in generation noise
    slot = target_len / n_items
    anchors = sorted(
        int(i * slot + rng.uniform(0.2, 0.8) * slot) for i in range(n_items)
    )

    ids, di, placed = [], 0, 0
    while len(ids) < target_len:
        while placed < n_items and len(ids) >= anchors[placed]:
            it = items[placed]
            start = len(ids)
            ids += tok("\n\n" + it["context"] + "\n\n",
                       add_special_tokens=False)["input_ids"]
            it["span"] = (start, len(ids))
            it["pos"] = len(ids)
            placed += 1
        d = pool[distractor_idx[di % len(distractor_idx)]]["context"]
        ids += tok("\n\n" + d, add_special_tokens=False)["input_ids"]
        di += 1

    # a paragraph anchored near the tail can be pushed past the cut; clamp it
    # inside so its distance band is still meaningful
    for it in items:
        if "pos" not in it or it["pos"] > target_len:
            it["span"] = (max(0, target_len - 1), target_len)
            it["pos"] = target_len

    return torch.tensor([ids[:target_len]]), items


def qa_spans(items):
    """Evidence spans for the oracle. Recorded at build time, not re-tokenised."""
    return [it["span"] for it in items]


def answer_spans(tok, items):
    """Token range of the gold answer phrase inside each gold paragraph.

    `qa_spans` is the whole paragraph (~160 tokens), the right unit for the
    oracle but the wrong denominator for coverage: the answer is a short phrase
    inside it. Re-tokenises the paragraph exactly as `build_qa_stream` did and
    maps the first answer that occurs in it through the offset mapping. Uses the
    first occurrence, which is not always the annotated one. An item clamped
    past the cut keeps its clamped span.
    """
    out = []
    for it in items:
        lo, hi = it["span"]
        text = "\n\n" + it["context"] + "\n\n"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        found = [(text.find(a), a) for a in it["answers"] if text.find(a) >= 0]
        if len(enc["input_ids"]) != hi - lo or not found:
            out.append((lo, hi))
            continue
        c0, a = min(found)
        c1 = c0 + len(a)
        toks = [i for i, (s, e) in enumerate(enc["offset_mapping"])
                if s < c1 and e > c0]
        out.append((lo + toks[0], lo + toks[-1] + 1))
    return out


# One neutral in-context example. Without it the model answers in a full
# sentence ("The Mongol armies would divert certain rivers to...") and token F1
# collapses to 0.12 on a one-word gold even though the answer is right. Probed
# against three prompt forms on an uncompressed cache: this one scored 4/6 and
# mean F1 0.61, against 3/6 / 0.21 for a bare instruction. The example is
# invented rather than drawn from SQuAD, so it cannot leak into any stream.
_SHOT = ("\n\nAnswer each question with a short phrase copied from the passage."
         "\n\nQuestion: What colour was the door?\nAnswer: pale green\n\n")


def query_qa(model, tok, cache, stream_len, item, device):
    """Ask one question against the compressed cache. Returns (hit, f1, em, text).

    `hit` is containment - is the gold span present in the generation - which is
    the same criterion the needle task uses, so the two are directly comparable.
    Token F1 is reported alongside it because containment alone is generous to a
    verbose answer and F1 alone is punitive toward one.
    """
    prompt = f"{_SHOT}Question: {item['question']}\nAnswer:"
    text = generate(model, tok, cache, stream_len, prompt, device)
    pred = text.split("\n")[0].strip()
    f1, em = score(pred, item["answers"])
    return bool(em), f1, em, text
