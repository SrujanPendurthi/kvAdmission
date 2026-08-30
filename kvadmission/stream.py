"""Needle-in-haystack stream construction and recall scoring.

Needles are planted at evenly spaced anchors with per-seed jitter, so repeated
seeds vary placement. Without jitter every seed measures the same layout and the
error bars would describe generation noise rather than placement sensitivity.
"""

import random
import re
import urllib.request
from pathlib import Path

import torch

from kvadmission.runner import generate

GUTENBERG = "https://www.gutenberg.org/files/1342/1342-0.txt"
CORPUS = Path(__file__).parent / "_haystack.txt"

# Answer templates spanning length AND redundancy. Note that a 6-digit code
# tokenises to 6 tokens - one per digit - so it is not short; it is
# *irredundant*. No digit is predictable from its neighbours, so every token
# must survive. A phrase or sentence is longer but internally predictable.
#
# The partial-oracle ablation found coverage unforgiving on codes. These styles
# test whether that requirement is about span length or about redundancy.
ANSWER_STYLES = {
    "code": lambda rng: f"{rng.randrange(100000, 999999)}",
    "phrase": lambda rng: rng.choice([
        "seventeen copper lanterns", "a folded brass sextant",
        "the ninth marble stairway", "two salt-stained ledgers",
        "an unsigned harbour permit", "the west quarry gate",
    ]),
    "sentence": lambda rng: rng.choice([
        "the shipment was rerouted through the northern harbour before dawn",
        "the ledger was signed by the quarry foreman and never countersigned",
        "the lantern room stayed lit for three nights after the storm passed",
        "the permit expired the same week the west gate was bricked over",
    ]),
}

TOPICS = [
    "oak", "harbor", "lantern", "quarry", "meridian", "thistle",
    "cobalt", "vellum", "cinder", "marrow", "fathom", "sable",
    "juniper", "ashen", "gallows", "wren", "pumice", "clover",
    "bramble", "onyx", "tallow", "sextant", "harrow", "plinth",
]


def _corpus():
    if not CORPUS.exists():
        raw = urllib.request.urlopen(GUTENBERG).read().decode("utf-8", "ignore")
        CORPUS.write_text(raw)
    return CORPUS.read_text()


def build_stream(tok, n_needles, target_len, seed=0, style="code"):
    """Plant `n_needles` facts through `target_len` tokens of prose.

    `style` selects answer length: "code" (a 6-digit number, few tokens),
    "phrase", or "sentence". Longer answers carry redundancy, so partial
    retention may still support recall - which the code style cannot.
    """
    if n_needles > len(TOPICS):
        raise ValueError(f"only {len(TOPICS)} topics available")

    rng = random.Random(seed)
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", _corpus())
        if 40 < len(s.strip()) < 400
    ]
    rng.shuffle(sentences)

    topics = rng.sample(TOPICS, n_needles)
    make = ANSWER_STYLES[style]
    needles = [{"topic": t, "code": make(rng), "style": style} for t in topics]

    # even anchors, jittered within their own slot so seeds differ in placement
    slot = target_len / n_needles
    anchors = sorted(
        int(i * slot + rng.uniform(0.2, 0.8) * slot) for i in range(n_needles)
    )

    ids, si, placed = [], 0, 0
    while len(ids) < target_len:
        while placed < n_needles and len(ids) >= anchors[placed]:
            n = needles[placed]
            ids += tok(
                f" The secret code for {n['topic']} is {n['code']}. ",
                add_special_tokens=False,
            )["input_ids"]
            n["pos"] = len(ids)
            placed += 1
        ids += tok(" " + sentences[si % len(sentences)],
                   add_special_tokens=False)["input_ids"]
        si += 1

    # a needle anchored near the tail can be pushed past the cut - place it just inside
    for n in needles:
        if "pos" not in n or n["pos"] > target_len:
            n["pos"] = target_len

    return torch.tensor([ids[:target_len]]), needles


def query_needle(model, tok, cache, stream_len, needle, device):
    """Ask for one needle's code against the compressed cache."""
    prompt = (
        f"\n\nQuestion: What is the secret code for {needle['topic']}?\nAnswer:"
    )
    text = generate(model, tok, cache, stream_len, prompt, device)
    return needle["code"] in text, text
