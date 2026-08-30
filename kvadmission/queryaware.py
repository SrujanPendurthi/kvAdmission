"""Query-aware admission: score tokens by relevance to the queries that will come.

The oracle (kvadmission/oracle.py) shows the capacity exists but cheats - it is
told which spans hold the answers. This is the honest version of the same idea.
It sees only the query *text*, which any deployed system has, and scores each
cached token by semantic similarity between its hidden state and the query's.

That converts the problem the learned policies could not solve - predict, at
write time, which tokens a future query will need - into a retrieval problem
against a query that is already known.

It is not an oracle: nothing tells it where the answer is, and a token that
merely looks topical will score highly whether or not it carries the answer.
The gap between this and the oracle is the cost of that imprecision.
"""

import torch


def query_embeddings(model, tok, needles, device):
    """One pooled hidden state per query, L2-normalised."""
    embs = []
    for n in needles:
        q = f"What is the secret code for {n['topic']}?"
        ids = tok(q, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0].float().mean(0)
        embs.append(h / (h.norm() + 1e-6))
    return torch.stack(embs)          # (n_queries, hidden)


def make_query_aware_decider(query_embs, budget, n_layers, sinks=4,
                             device="cpu", recency_weight=0.0):
    """Keep the tokens most similar to any pending query.

    `recency_weight` mixes in a positional prior; 0.0 is pure relevance.
    Scores are the max cosine similarity over queries, so a token relevant to
    any one of them is protected.
    """
    state = {"score": None, "hidden": None}

    def decide(layer_idx, cache, attentions, n_new):
        if layer_idx == 0:
            h = state["hidden"][-n_new:]                       # (n_new, hidden)
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (h @ query_embs.T).max(dim=-1).values.cpu()  # best query match
            state["score"] = (sim if state["score"] is None
                              else torch.cat([state["score"], sim]))

        score = state["score"]
        n = cache.occupancy(layer_idx)
        assert score.shape[0] == n, f"score {score.shape[0]} vs cache {n}"
        blended = score + recency_weight * torch.linspace(0, 1, n)
        surviving = cache.apply_eviction(layer_idx, blended)
        if layer_idx == n_layers - 1:
            state["score"] = score[surviving.cpu()]

    decide.state = state
    return decide, False
