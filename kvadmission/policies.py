"""Baseline compression policies, as deciders for runner.prefill.

Two families. `recent` and `random` are admission policies: they choose which
newly-written tokens enter, and FIFO does the rest. `h2o` and `snapkv` are
eviction policies: they score every resident and keep the top B, which is how
the published methods actually work and how they must be reproduced.
"""

import torch

CHUNK_HINT = 128  # only used to set random's admit rate; see make_decider


def make_decider(policy, budget, sinks=4, window=32, seed=0, chunk=CHUNK_HINT):
    """Build a (decide, need_attn) pair for `runner.prefill`."""
    if policy == "full":
        return None, False

    if policy == "recent":
        def decide(layer_idx, cache, attentions, n_new):
            cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))
        return decide, False

    if policy == "random":
        gen = torch.Generator().manual_seed(seed)
        rate = budget / (budget + chunk)

        def decide(layer_idx, cache, attentions, n_new):
            keep = torch.rand(n_new, generator=gen) < rate
            cache.apply_admission(layer_idx, keep)
        return decide, False

    if policy in ("h2o", "snapkv"):
        acc = {}

        def decide(layer_idx, cache, attentions, n_new):
            a = attentions[layer_idx].float()          # (1, H, n_query, n_kv)
            if policy == "h2o":
                cur = a.sum(dim=(0, 1, 2)).cpu()
                prev = acc.get(layer_idx)
                if prev is not None:
                    cur[: prev.shape[0]] += prev
                scores = cur
            else:
                scores = a[:, :, -window:, :].sum(dim=(0, 1, 2)).cpu()

            surviving = cache.apply_eviction(layer_idx, scores)
            acc[layer_idx] = scores[surviving.cpu()]
        return decide, True

    raise ValueError(f"unknown policy: {policy}")
