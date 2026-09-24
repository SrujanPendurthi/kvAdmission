"""A supervised write-time scorer: logistic regression on hidden states.

Phases 2-2c learned admission by RL on a KL reward and got recency back. This
is the direct supervised version: label every answer-span token 1, every other
token 0, and fit a linear read of the frozen model's layer-`layer` hidden
state. The feature at position t comes from a causal forward pass, so it is
available at write time; the labels are supervision, used only in training.

`results/probe-scorer.json`: on held-out needle seeds it keeps 64/64 spans
whole under the coverage screen, the first realisable signal here to pass it.
It does not transfer - trained on QA it finds 0/64 needle spans.
"""

import torch
from transformers import DynamicCache


@torch.no_grad()
def features(model, ids, device, layer=14, chunk=128):
    """Layer-`layer` hidden state per token, from an uncompressed causal prefill."""
    cache, out = DynamicCache(), []
    for s in range(0, ids.shape[1], chunk):
        o = model(ids[:, s:s + chunk].to(device), past_key_values=cache,
                  use_cache=True, output_hidden_states=True)
        out.append(o.hidden_states[layer][0].float().cpu().half())
    return torch.cat(out)


def labels(spans, n_tokens):
    y = torch.zeros(n_tokens)
    for lo, hi in spans:
        y[lo:hi] = 1
    return y


def fit(x, y, l2=1e-3, iters=200):
    """Class-balanced logistic regression. Returns a scorer over feature rows."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xn = (x - mu) / sd
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    opt = torch.optim.LBFGS([w, b], max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            xn @ w + b, y, pos_weight=pos_weight) + l2 * w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda feats: ((feats.float() - mu) / sd) @ w.detach() + b.detach()


def make_score_decider(scores, budget, n_layers, sinks=4, recency_weight=0.0):
    """Evict by a precomputed per-position score, keeping the top `budget`.

    Same shape as `oracle.make_oracle_decider`, with the oracle's flag replaced
    by any per-position score: the scores arrive chunk by chunk as the stream is
    written, and the resident score vector is trimmed only after the last layer
    is evicted, or it desynchronises from the layers still to come.
    """
    state = {"resident": None, "seen": 0, "surviving": None}

    def decide(layer_idx, cache, attentions, n_new):
        if layer_idx == 0:
            start = state["seen"]
            new = scores[start:start + n_new].float()
            state["resident"] = (new if state["resident"] is None
                                 else torch.cat([state["resident"], new]))
            state["seen"] = start + n_new

        resident = state["resident"]
        n = cache.occupancy(layer_idx)
        assert resident.shape[0] == n, f"scores {resident.shape[0]} vs cache {n}"
        rank = resident
        if recency_weight:
            rank = rank + recency_weight * torch.arange(n, dtype=torch.float32)
        state["surviving"] = cache.apply_eviction(layer_idx, rank).cpu()

        if layer_idx == n_layers - 1:
            state["resident"] = resident[state["surviving"]]

    return decide, False
