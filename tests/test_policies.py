import torch

from kvadmission.cache import AdmissionCache
from kvadmission.policies import make_decider


def _fake_attn(n_layers, n_kv, n_query, heads=2):
    """Attention with mass concentrated on kv position 0."""
    a = torch.full((1, heads, n_query, n_kv), 0.01)
    a[:, :, :, 0] = 10.0
    return [a for _ in range(n_layers)]


def _run_random(seed):
    decide, _ = make_decider("random", budget=6, sinks=0, seed=seed)
    cache = AdmissionCache(budget=6, sinks=0)
    k = torch.zeros(1, 2, 10, 4)
    k[0, 0, :, 0] = torch.arange(10, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, None, 10)
    return cache.layers[0].keys[0, 0, :, 0].tolist()


def test_full_policy_never_compresses():
    decide, need_attn = make_decider("full", budget=4)
    assert decide is None
    assert need_attn is False


def test_recent_admits_everything_and_fifo_trims():
    decide, need_attn = make_decider("recent", budget=6, sinks=2)
    assert need_attn is False
    cache = AdmissionCache(budget=6, sinks=2)
    k = torch.zeros(1, 2, 8, 4)
    k[0, 0, :, 0] = torch.arange(8, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, None, 8)
    assert cache.occupancy(0) == 6
    # sinks 0,1 pinned, newest four kept
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0]


def test_random_is_deterministic_under_seed():
    assert _run_random(7) == _run_random(7)


def test_random_respects_budget():
    assert len(_run_random(7)) <= 6


def test_h2o_keeps_the_high_attention_token():
    decide, need_attn = make_decider("h2o", budget=2, sinks=0)
    assert need_attn is True
    cache = AdmissionCache(budget=2, sinks=0)
    k = torch.zeros(1, 2, 4, 4)
    k[0, 0, :, 0] = torch.arange(4, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, _fake_attn(1, n_kv=4, n_query=4), 4)
    assert 0.0 in cache.layers[0].keys[0, 0, :, 0].tolist()


def test_h2o_scores_survive_across_chunks():
    """Score bookkeeping must be reindexed through eviction, not reset."""
    decide, _ = make_decider("h2o", budget=3, sinks=0)
    cache = AdmissionCache(budget=3, sinks=0)
    for step in range(3):
        k = torch.zeros(1, 2, 2, 4)
        k[0, 0, :, 0] = torch.arange(step * 2, step * 2 + 2, dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        n_kv = cache.occupancy(0)
        decide(0, cache, _fake_attn(1, n_kv=n_kv, n_query=2), 2)
        assert cache.occupancy(0) <= 3
    # position 0 carried the attention mass every round, so it must still be there
    assert 0.0 in cache.layers[0].keys[0, 0, :, 0].tolist()


def test_snapkv_uses_only_the_observation_window():
    decide, need_attn = make_decider("snapkv", budget=2, sinks=0, window=1)
    assert need_attn is True
    cache = AdmissionCache(budget=2, sinks=0)
    k = torch.zeros(1, 2, 4, 4)
    k[0, 0, :, 0] = torch.arange(4, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    a = torch.full((1, 2, 4, 4), 0.01)
    a[:, :, -1, 3] = 99.0  # only the final query row attends to position 3
    decide(0, cache, [a], 4)
    assert 3.0 in cache.layers[0].keys[0, 0, :, 0].tolist()
