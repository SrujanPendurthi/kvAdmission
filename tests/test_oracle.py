import torch
from transformers import AutoTokenizer

from kvadmission.cache import AdmissionCache
from kvadmission.oracle import make_oracle_decider, needle_spans
from kvadmission.stream import build_stream

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_spans_cover_the_planted_text():
    ids, needles = build_stream(TOK, 4, 2048, seed=0)
    for (a, b), n in zip(needle_spans(TOK, needles), needles):
        assert 0 <= a < b <= 2048
        assert n["code"] in TOK.decode(ids[0, a:b])


def test_oracle_keeps_needle_tokens_under_pressure():
    """The whole point: needle tokens must survive when recency would drop them."""
    spans = [(0, 4)]
    decide, _ = make_oracle_decider(spans, budget=8, n_layers=1, sinks=0)
    cache = AdmissionCache(budget=8, sinks=0)
    for step in range(6):
        k = torch.zeros(1, 2, 4, 4)
        k[0, 0, :, 0] = torch.arange(step * 4, step * 4 + 4, dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        decide(0, cache, None, 4)
    resident = cache.layers[0].keys[0, 0, :, 0].tolist()
    assert all(t in resident for t in (0.0, 1.0, 2.0, 3.0)), resident


def test_oracle_respects_budget():
    decide, _ = make_oracle_decider([(0, 4)], budget=8, n_layers=1, sinks=0)
    cache = AdmissionCache(budget=8, sinks=0)
    for step in range(6):
        k = torch.zeros(1, 2, 4, 4)
        cache.update(k, k.clone(), 0)
        decide(0, cache, None, 4)
    assert cache.occupancy(0) <= 8


def test_oracle_falls_back_to_recency_outside_spans():
    decide, _ = make_oracle_decider([], budget=4, n_layers=1, sinks=0)
    cache = AdmissionCache(budget=4, sinks=0)
    for step in range(3):
        k = torch.zeros(1, 2, 4, 4)
        k[0, 0, :, 0] = torch.arange(step * 4, step * 4 + 4, dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        decide(0, cache, None, 4)
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [8.0, 9.0, 10.0, 11.0]


def test_flag_stays_aligned_across_multiple_layers():
    """Trimming after layer 0 would desync the flag from layers 1..n-1."""
    decide, _ = make_oracle_decider([(0, 4)], budget=8, n_layers=3, sinks=0)
    cache = AdmissionCache(budget=8, sinks=0)
    for step in range(5):
        for li in range(3):
            k = torch.zeros(1, 2, 4, 4)
            k[0, 0, :, 0] = torch.arange(step * 4, step * 4 + 4, dtype=torch.float32)
            cache.update(k, k.clone(), li)
        for li in range(3):
            decide(li, cache, None, 4)   # must not raise
        assert len({cache.occupancy(li) for li in range(3)}) == 1


def test_zero_coverage_degenerates_to_recency():
    """A proxy that finds nothing must behave exactly like a recency window."""
    decide, _ = make_oracle_decider([(0, 4)], budget=4, n_layers=1, sinks=0,
                                    coverage=0.0, fp_rate=0.0)
    cache = AdmissionCache(budget=4, sinks=0)
    for step in range(3):
        k = torch.zeros(1, 2, 4, 4)
        k[0, 0, :, 0] = torch.arange(step * 4, step * 4 + 4, dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        decide(0, cache, None, 4)
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [8.0, 9.0, 10.0, 11.0]


def test_false_positives_crowd_out_true_ones():
    """The budget is shared, so a noisy proxy evicts what it meant to protect."""
    clean, _ = make_oracle_decider([(0, 2)], budget=6, n_layers=1, sinks=0,
                                   coverage=1.0, fp_rate=0.0, seed=1)
    noisy, _ = make_oracle_decider([(0, 2)], budget=6, n_layers=1, sinks=0,
                                   coverage=1.0, fp_rate=1.0, seed=1)
    out = []
    for decide in (clean, noisy):
        cache = AdmissionCache(budget=6, sinks=0)
        for step in range(8):
            k = torch.zeros(1, 2, 4, 4)
            k[0, 0, :, 0] = torch.arange(step * 4, step * 4 + 4, dtype=torch.float32)
            cache.update(k, k.clone(), 0)
            decide(0, cache, None, 4)
        out.append(cache.layers[0].keys[0, 0, :, 0].tolist())
    assert 0.0 in out[0], "clean oracle lost its needle"
    assert 0.0 not in out[1], "flagging everything should protect nothing"


def test_dilate_with_zero_radius_is_identity():
    from kvadmission.oracle import dilate_spans
    assert dilate_spans([(10, 20), (50, 60)], 0, 100) == [(10, 20), (50, 60)]


def test_dilate_widens_both_sides_and_clamps_to_the_stream():
    from kvadmission.oracle import dilate_spans
    assert dilate_spans([(10, 20)], 5, 100) == [(5, 25)]
    assert dilate_spans([(2, 8)], 5, 100) == [(0, 13)]      # clamped at 0
    assert dilate_spans([(90, 98)], 5, 100) == [(85, 100)]  # clamped at n


def test_dilate_merges_runs_that_touch():
    # the whole point of the control: a wider radius means fewer, longer runs
    from kvadmission.oracle import dilate_spans
    assert dilate_spans([(10, 20), (30, 40)], 4, 100) == [(6, 24), (26, 44)]
    assert dilate_spans([(10, 20), (30, 40)], 5, 100) == [(5, 45)]


def test_span_stats_counts_runs_and_tokens():
    from kvadmission.oracle import span_stats
    assert span_stats([(0, 10), (20, 25)]) == (2, 15)


def test_dilation_is_monotone_in_runs_and_tokens():
    from kvadmission.oracle import dilate_spans, span_stats
    spans = [(100, 110), (400, 410), (800, 810), (1200, 1210)]
    prev_runs, prev_tokens = 5, 0
    for r in (0, 20, 80, 200, 600):
        runs, tokens = span_stats(dilate_spans(spans, r, 2048))
        assert runs <= prev_runs      # merging never increases run count
        assert tokens >= prev_tokens  # widening never retains fewer tokens
        prev_runs, prev_tokens = runs, tokens


def test_matched_token_budget_gives_run_count_equal_to_k():
    # the fragment-count experiment's core invariant: varying k must change the
    # number of runs and leave the flagged token total alone, or it is
    # measuring window width again instead of gap count
    from kvadmission.oracle import dilate_spans, span_stats
    base = [(1000 * i + 100, 1000 * i + 116) for i in range(8)]
    target = 1600
    for k in (8, 4, 2, 1):
        chosen = base[:k]
        base_tokens = sum(b - a for a, b in chosen)
        r = max(0, round((target - base_tokens) / (2 * k)))
        runs, flagged = span_stats(dilate_spans(chosen, r, 8192))
        assert runs == k, f"k={k} gave {runs} runs"
        assert abs(flagged - target) <= 0.1 * target, f"k={k} flagged {flagged}"
