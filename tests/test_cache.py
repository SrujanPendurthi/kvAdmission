import torch

from kvadmission.cache import AdmissionCache


def _write(cache, layer_idx, n_tokens, heads=2, dim=4):
    """Append n_tokens of identifiable KV to a layer. Key value == token ordinal."""
    start = cache.occupancy(layer_idx)
    ords = torch.arange(start, start + n_tokens, dtype=torch.float32)
    k = ords.view(1, 1, n_tokens, 1).expand(1, heads, n_tokens, dim).contiguous()
    cache.update(k, k.clone(), layer_idx)
    return ords


def _resident_ordinals(cache, layer_idx):
    return cache.layers[layer_idx].keys[0, 0, :, 0].tolist()


def test_unbounded_budget_keeps_everything():
    cache = AdmissionCache(budget=None, sinks=4)
    _write(cache, 0, 10)
    cache.apply_admission(0, torch.zeros(10, dtype=torch.bool))
    assert cache.occupancy(0) == 10
    assert _resident_ordinals(cache, 0) == list(range(10))


def test_rejected_tokens_are_removed():
    cache = AdmissionCache(budget=100, sinks=0)
    _write(cache, 0, 6)
    keep = torch.tensor([True, False, True, False, True, False])
    cache.apply_admission(0, keep)
    assert _resident_ordinals(cache, 0) == [0.0, 2.0, 4.0]


def test_occupancy_never_exceeds_budget():
    cache = AdmissionCache(budget=8, sinks=4)
    for _ in range(10):
        n = cache.occupancy(0)
        _write(cache, 0, 5)
        cache.apply_admission(0, torch.ones(5, dtype=torch.bool))
        assert cache.occupancy(0) <= 8, f"overflow after starting from {n}"


def test_sinks_are_never_evicted():
    cache = AdmissionCache(budget=8, sinks=4)
    for _ in range(10):
        _write(cache, 0, 5)
        cache.apply_admission(0, torch.ones(5, dtype=torch.bool))
    assert _resident_ordinals(cache, 0)[:4] == [0.0, 1.0, 2.0, 3.0]


def test_eviction_is_fifo_oldest_non_sink_first():
    cache = AdmissionCache(budget=6, sinks=2)
    _write(cache, 0, 6)
    cache.apply_admission(0, torch.ones(6, dtype=torch.bool))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    _write(cache, 0, 2)
    cache.apply_admission(0, torch.ones(2, dtype=torch.bool))
    # sinks 0,1 pinned; oldest non-sinks 2,3 dropped; 6,7 admitted
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0]


def test_sinks_are_force_admitted_even_if_rejected():
    cache = AdmissionCache(budget=8, sinks=4)
    _write(cache, 0, 6)
    cache.apply_admission(0, torch.zeros(6, dtype=torch.bool))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0]


def test_returned_indices_reindex_parallel_bookkeeping():
    cache = AdmissionCache(budget=100, sinks=0)
    _write(cache, 0, 5)
    scores = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    keep = torch.tensor([True, False, True, False, True])
    surviving = cache.apply_admission(0, keep)
    assert scores[surviving].tolist() == [10.0, 30.0, 50.0]


def test_layers_are_independent():
    cache = AdmissionCache(budget=4, sinks=0)
    _write(cache, 0, 6)
    _write(cache, 1, 6)
    cache.apply_admission(0, torch.ones(6, dtype=torch.bool))
    assert cache.occupancy(0) == 4
    assert cache.occupancy(1) == 6
