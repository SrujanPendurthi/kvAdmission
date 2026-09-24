import torch

from kvadmission.cache import AdmissionCache
from kvadmission.probe import fit, labels, make_score_decider


def test_labels_marks_spans():
    y = labels([(2, 5), (8, 9)], 10)
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]


def test_fit_separates_a_planted_direction():
    torch.manual_seed(0)
    x = torch.randn(400, 8)
    y = (x[:, 0] > 1.0).float()
    score = fit(x, y)
    s = score(x)
    assert s[y == 1].mean() > s[y == 0].mean() + 1.0


def _write(cache, layer_idx, n, heads=2, dim=4):
    start = cache.occupancy(layer_idx)
    ords = torch.arange(start, start + n, dtype=torch.float32)
    k = ords.view(1, 1, n, 1).expand(1, heads, n, dim).contiguous()
    cache.update(k, k.clone(), layer_idx)


def test_score_decider_keeps_the_top_budget_by_score():
    scores = torch.zeros(24)
    scores[4:12] = 5.0                      # the span the signal likes
    cache = AdmissionCache(budget=12, sinks=4)
    decide, need_attn = make_score_decider(scores, 12, n_layers=1, sinks=4)
    assert need_attn is False
    for _ in range(3):
        _write(cache, 0, 8)
        decide(0, cache, None, 8)
    kept = cache.layers[0].keys[0, 0, :, 0].tolist()
    assert cache.occupancy(0) == 12
    assert set(range(4, 12)) <= set(kept)   # high scorers survive to the end
    assert kept[:4] == [0.0, 1.0, 2.0, 3.0]  # sinks pinned
