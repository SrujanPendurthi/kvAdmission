import torch

from kvadmission.cache import AdmissionCache
from kvadmission.surprisal import make_surprisal_decider


def _fake_logits(n_new, vocab, favour=None, strength=8.0):
    """Uniform logits, optionally making `favour[i]` highly likely at row i."""
    lg = torch.zeros(1, n_new, vocab)
    if favour is not None:
        for i, tok in enumerate(favour):
            if tok is not None:
                lg[0, i, tok] = strength
    return lg


def test_predictable_tokens_are_evicted_before_surprising_ones():
    # the whole premise: a token the context already predicts is
    # reconstructible, so it should lose to one the context did not predict
    vocab, n = 50, 8
    ids = torch.arange(n).unsqueeze(0)
    # every row strongly predicts the NEXT token except before position 5
    favour = [ids[0, i + 1].item() if i + 1 < n else None for i in range(n)]
    favour[4] = None                       # so token 5 is a surprise
    cache = AdmissionCache(budget=4, sinks=0)
    decide, _ = make_surprisal_decider(ids, budget=4, n_layers=1, sinks=0)
    cache.update(torch.randn(1, 2, n, 4), torch.arange(n, dtype=torch.float32)
                 .view(1, 1, n, 1).expand(1, 2, n, 4).contiguous(), 0)
    decide(0, cache, None, n, _fake_logits(n, vocab, favour))
    kept = cache.layers[0].values[0, 0, :, 0].tolist()
    assert 5 in kept, f"the surprising token was evicted: kept {kept}"


def test_scores_track_cache_occupancy_across_chunks():
    # the score vector must be reindexed on eviction or it desynchronises from
    # the cache, the same failure the oracle's flag had
    vocab, n = 40, 6
    ids = torch.arange(12).unsqueeze(0)
    cache = AdmissionCache(budget=8, sinks=0)
    decide, _ = make_surprisal_decider(ids, budget=8, n_layers=1, sinks=0)
    for c in range(2):
        cache.update(torch.randn(1, 2, n, 4), torch.randn(1, 2, n, 4), 0)
        decide(0, cache, None, n, _fake_logits(n, vocab))
    assert cache.occupancy(0) <= 8


def test_budget_is_respected_so_crowding_never_binds():
    # top-B by construction: |flagged| == B exactly, which is the half of the
    # proxy specification the partial-oracle ablation showed is easy to violate
    vocab, n = 40, 16
    ids = torch.arange(32).unsqueeze(0)
    cache = AdmissionCache(budget=10, sinks=0)
    decide, _ = make_surprisal_decider(ids, budget=10, n_layers=1, sinks=0)
    for _ in range(2):
        cache.update(torch.randn(1, 2, n, 4), torch.randn(1, 2, n, 4), 0)
        decide(0, cache, None, n, _fake_logits(n, vocab))
    assert cache.occupancy(0) == 10


def test_decider_reports_it_does_not_need_attention():
    _, need_attn = make_surprisal_decider(torch.arange(8).unsqueeze(0),
                                          budget=4, n_layers=1)
    assert need_attn is False
