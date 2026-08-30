import torch

from kvadmission.cache import AdmissionCache
from kvadmission.queryaware import make_query_aware_decider


def _run(query_embs, hidden, budget=4, steps=3, chunk=4, recency_weight=0.0):
    decide, _ = make_query_aware_decider(query_embs, budget, n_layers=1,
                                         sinks=0, recency_weight=recency_weight)
    cache = AdmissionCache(budget=budget, sinks=0)
    for step in range(steps):
        k = torch.zeros(1, 2, chunk, 4)
        k[0, 0, :, 0] = torch.arange(step * chunk, step * chunk + chunk,
                                     dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        decide.state["hidden"] = hidden[step * chunk:(step + 1) * chunk]
        decide(0, cache, None, chunk)
    return cache.layers[0].keys[0, 0, :, 0].tolist()


def test_relevant_tokens_survive_irrelevant_ones():
    """A token matching the query must outlast newer, unrelated tokens."""
    q = torch.tensor([[1.0, 0.0]])
    hidden = torch.tensor([[1.0, 0.0]] + [[0.0, 1.0]] * 11)  # token 0 matches
    assert 0.0 in _run(q, hidden)


def test_pure_recency_weight_recovers_a_recency_window():
    q = torch.tensor([[1.0, 0.0]])
    hidden = torch.zeros(12, 2)          # no token matches anything
    assert _run(q, hidden, recency_weight=10.0) == [8.0, 9.0, 10.0, 11.0]


def test_matches_any_of_several_queries():
    """Scores are a max over queries, so relevance to one is enough."""
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hidden = torch.tensor([[0.0, 1.0]] + [[0.7, 0.7]] * 11)  # token 0 matches q2
    assert 0.0 in _run(q, hidden)


def test_respects_budget():
    q = torch.tensor([[1.0, 0.0]])
    hidden = torch.randn(12, 2)
    assert len(_run(q, hidden, budget=4)) == 4
