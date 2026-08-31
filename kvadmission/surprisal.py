"""Surprisal admission: keep what the context cannot reconstruct.

Every scoring rule that has failed here is a function of position. Recency
keeps the newest, accumulated attention keeps the oldest, observation-window
voting keeps whatever the last few queries looked at. None is a function of
whether a token carries information.

This one is. A token the model already predicts from its prefix is
reconstructible - the surrounding context regenerates it - so storing it buys
nothing. A token the model did not predict is the only kind that carries
information the rest of the cache does not already hold. Score each token by
its own surprisal, keep the top B.

Two properties make it worth testing before anything more elaborate:

  it is genuinely write-time    the logits already exist during prefill, and
                               surprisal at position t depends only on tokens
                               before t. No lookahead, no query, no second pass.
  it satisfies half the spec   top-B means |flagged| = B exactly, so the
                               crowding law of the partial-oracle ablation
                               never binds. Only coverage is at risk.

Surprisal is computed from the COMPRESSED forward pass, not a privileged
uncompressed one, so this is a policy rather than an oracle.

Expect it to do well on synthetic needles precisely because a random 6-digit
code is maximally unpredictable. That is the reason to also run it on real QA,
where answer spans are ordinary language: a signal that cracks the needle task
and fails on SQuAD says more about the benchmark than about the method.
"""

import torch


def make_surprisal_decider(ids, budget, n_layers, sinks=4, device="cpu",
                           recency_weight=0.0):
    """Score residents by their own surprisal and keep the top `budget`.

    `ids` is the stream being compressed - the policy needs the token at each
    position to look up its probability. That is the input itself, not
    privileged information: an online policy sees each token as it is written.

    `recency_weight` optionally blends in position, for probing whether a
    little recency helps; 0.0 is pure surprisal.
    """
    flat = ids[0] if ids.dim() > 1 else ids
    state = {"scores": None, "prev_row": None, "seen": 0, "surviving": None}

    def decide(layer_idx, cache, attentions, n_new, logits=None):
        if layer_idx == 0:
            lg = logits[0].float()                       # (n_new, vocab)
            start = state["seen"]
            chunk_ids = flat[start:start + n_new].to(lg.device)
            # surprisal of token i is -log p(token i | prefix), and the row that
            # predicts it is the one before it: the previous chunk's last row
            # for i=0, then this chunk's own rows.
            rows = lg[:-1]
            if state["prev_row"] is not None:
                rows = torch.cat([state["prev_row"].unsqueeze(0), rows], dim=0)
                targets = chunk_ids
            else:
                # no predecessor for the very first token; it is inside the
                # pinned sink prefix anyway, so give it a neutral score
                targets = chunk_ids[1:]
            lp = torch.log_softmax(rows, dim=-1)
            s = -lp.gather(1, targets.unsqueeze(1)).squeeze(1)
            if state["prev_row"] is None:
                s = torch.cat([s.new_zeros(1), s])
            state["prev_row"] = lg[-1].detach()

            s = s.detach().cpu()
            state["scores"] = (s if state["scores"] is None
                               else torch.cat([state["scores"], s]))
            state["seen"] += n_new

        scores = state["scores"]
        n = cache.occupancy(layer_idx)
        assert scores.shape[0] == n, f"scores {scores.shape[0]} vs cache {n}"
        rank = scores.clone()
        if recency_weight:
            rank = rank + recency_weight * torch.arange(n, dtype=torch.float32)
        surviving = cache.apply_eviction(layer_idx, rank)
        state["surviving"] = surviving.cpu()

        # trim only after the LAST layer, or the score vector desynchronises
        # from layers that have not been evicted yet
        if layer_idx == n_layers - 1:
            state["scores"] = scores[state["surviving"]]

    return decide, False
