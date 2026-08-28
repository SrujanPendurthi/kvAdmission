"""Does distant context actually move the next-token distribution?

The dense KL reward can only teach mid-context retention if removing distant
tokens measurably changes what the model predicts. Code is the corpus of choice:
an identifier defined thousands of tokens earlier and used again is a genuine,
mechanically detectable long-range dependency.

Deliberately NOT Pride and Prejudice - that is the evaluation haystack, and
training on it would invalidate the result.
"""

import random
import statistics
import sysconfig
from pathlib import Path

import torch

from kvadmission.dual import dual_prefill


def code_corpus(max_files=400):
    """Local Python sources over 20 KB, tests excluded."""
    roots = [Path(sysconfig.get_paths()["stdlib"]),
             Path(sysconfig.get_paths()["purelib"])]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            s = str(p)
            if "test" in s or "site-packages/pip" in s:
                continue
            try:
                if p.stat().st_size > 20_000:
                    out.append(s)
            except OSError:
                continue
            if len(out) >= max_files:
                return sorted(out)
    return sorted(out)


def build_code_stream(tok, target_len, seed=0):
    """Concatenate shuffled source files and cut to exactly target_len tokens."""
    paths = code_corpus()
    rng = random.Random(seed)
    rng.shuffle(paths)

    ids = []
    for p in paths:
        try:
            text = Path(p).read_text(errors="ignore")
        except OSError:
            continue
        ids += tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_len:
            break
    if len(ids) < target_len:
        raise RuntimeError(f"corpus too small: {len(ids)} < {target_len}")
    return torch.tensor([ids[:target_len]])


def long_dependency_positions(tok, ids, min_distance=2048, min_len=4):
    """Positions predicting an identifier last seen at least min_distance back."""
    toks = tok.convert_ids_to_tokens(ids[0].tolist())
    last_seen, out = {}, []
    for i, raw in enumerate(toks):
        key = raw.replace("Ġ", "").strip()  # strip the GPT-2 space marker
        if len(key) >= min_len and key.isidentifier():
            prev = last_seen.get(key)
            if prev is not None and i - prev >= min_distance and i > 0:
                out.append(i - 1)
            last_seen[key] = i
    return out


def signal_report(model, tok, device, budget, length, seed):
    """Compare KL at long-dependency positions against everywhere else."""
    ids = build_code_stream(tok, length, seed=seed)
    long_pos = long_dependency_positions(tok, ids)

    def admit_all(layer_idx, cache, attentions, n_new):
        cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))

    kl, _ = dual_prefill(model, ids, device, budget=budget, decide=admit_all)

    # only positions past the point where eviction actually starts
    warm = budget
    long_set = set(p for p in long_pos if p >= warm)
    other = [p for p in range(warm, length) if p not in long_set]
    if not long_set:
        return {"kl_long": 0.0, "kl_baseline": 0.0, "ratio": 0.0, "n_long": 0}

    kl_long = statistics.mean(kl[p].item() for p in long_set)
    kl_base = statistics.mean(kl[p].item() for p in other)
    return {
        "kl_long": kl_long,
        "kl_baseline": kl_base,
        "ratio": kl_long / kl_base if kl_base > 0 else float("inf"),
        "n_long": len(long_set),
    }
