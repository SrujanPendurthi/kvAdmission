"""Oracle retention: what if the policy already knew what to keep?

This is the experiment that separates the two explanations for the mid-context
dead zone.

  observability  - the information is retainable at this budget, but no signal
                   available at write time identifies which tokens carry it
  budget geometry - even perfect retention cannot answer, because a compressed
                   cache holding the right tokens is still not enough

The oracle scores needle tokens astronomically and everything else by recency,
then reuses the same apply_eviction path the baselines use. Budget, chunking,
sinks and eviction geometry are all unchanged - the only difference is that
this policy cheats.

If oracle recall is high, the dead zone is observability. If it is low, the
limitation is the geometry and no admission policy can fix it.
"""

import torch

BIG = 1e6


def needle_spans(tok, needles):
    """Token ranges each planted fact occupies. Returns [(start, end), ...].

    build_stream records `pos` as the index just past the inserted text, so the
    span is recovered by re-tokenising the same string.
    """
    spans = []
    for n in needles:
        text = f" The secret code for {n['topic']} is {n['code']}. "
        length = len(tok(text, add_special_tokens=False)["input_ids"])
        spans.append((max(0, n["pos"] - length), n["pos"]))
    return spans


def dilate_spans(spans, radius, n_tokens):
    """Widen each span by `radius` on both sides and merge what overlaps.

    This is the knob for the contiguity control. The oracle's retained set is
    normally maximally fragmented - one short run per answer span, spliced
    between recency fill, with gaps in the position indices. Growing the radius
    merges neighbouring runs, so the same evidence is retained inside
    progressively fewer, longer contiguous blocks.

    Radius trades against budget: a wider window flags more tokens, and past
    B the crowding law takes over. Report `flagged_tokens` alongside `runs` so
    a fall in recall from crowding is not read as an effect of fragmentation.
    """
    if not spans:
        return []
    wide = sorted((max(0, a - radius), min(n_tokens, b + radius))
                  for a, b in spans)
    merged = [list(wide[0])]
    for a, b in wide[1:]:
        if a <= merged[-1][1]:          # touching or overlapping - one run
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(m) for m in merged]


def span_stats(spans):
    """(runs, flagged_tokens) for a set of already-merged spans."""
    return len(spans), sum(b - a for a, b in spans)


def make_oracle_decider(spans, budget, n_layers, sinks=4, device="cpu",
                        coverage=1.0, fp_rate=0.0, seed=0):
    """A decider that pins needle tokens and ranks the rest by recency.

    `coverage` and `fp_rate` degrade the oracle into a proxy of measurable
    quality, which is how we bound what a learned signal would need to achieve:

      coverage - probability a true needle token is flagged (the proxy's recall)
      fp_rate  - probability a non-needle token is falsely flagged (1 - precision)

    coverage=1.0, fp_rate=0.0 is the perfect oracle. Note that false positives
    compete for the same budget as true ones, so a high fp_rate crowds the very
    tokens the flag was meant to protect.

    The flag tensor is extended once per chunk and trimmed only after the LAST
    layer is evicted. Trimming after layer 0 would leave it misaligned with
    layers 1..n-1, which still hold the pre-eviction token count.
    """
    state = {"flag": None, "seen": 0, "surviving": None}
    gen = torch.Generator().manual_seed(seed)

    def decide(layer_idx, cache, attentions, n_new):
        if layer_idx == 0:
            start = state["seen"]
            new = torch.zeros(n_new, dtype=torch.bool)
            noise = torch.rand(n_new, generator=gen)
            for i in range(n_new):
                p = start + i
                if any(a <= p < b for a, b in spans):
                    new[i] = noise[i] < coverage      # proxy may miss it
                else:
                    new[i] = noise[i] < fp_rate       # proxy may falsely flag it
            state["flag"] = (new if state["flag"] is None
                             else torch.cat([state["flag"], new]))
            state["seen"] = start + n_new

        flag = state["flag"]
        n = cache.occupancy(layer_idx)
        assert flag.shape[0] == n, f"flag {flag.shape[0]} vs cache {n}"
        # recency baseline plus an unbeatable bonus for needle tokens
        scores = torch.arange(n, dtype=torch.float32) + BIG * flag.float()
        surviving = cache.apply_eviction(layer_idx, scores)
        state["surviving"] = surviving.cpu()

        if layer_idx == n_layers - 1:
            state["flag"] = flag[state["surviving"]]

    return decide, False
