import torch

from kvadmission.screen import coverage


def test_coverage_is_one_when_the_span_is_top_ranked():
    scores = torch.tensor([9.0, 9.0, 9.0, 0.1, 0.1, 0.1])
    assert coverage(scores, [(0, 3)], budget=3, n_tokens=6) == [1.0]


def test_coverage_is_zero_when_the_span_ranks_last():
    scores = torch.tensor([0.1, 0.1, 0.1, 9.0, 9.0, 9.0])
    assert coverage(scores, [(0, 3)], budget=3, n_tokens=6) == [0.0]


def test_partial_coverage_is_reported_as_a_fraction():
    # the number that matters: §7.2 says 0.75 already collapses the middle band
    scores = torch.tensor([9.0, 9.0, 9.0, 0.1, 5.0, 0.1])
    cov = coverage(scores, [(2, 6)], budget=4, n_tokens=6)
    assert cov == [0.5]


def test_spans_past_the_stream_end_are_clipped_not_counted_wrong():
    scores = torch.arange(6, dtype=torch.float32)
    assert coverage(scores, [(4, 99)], budget=2, n_tokens=6) == [1.0]


def test_a_span_entirely_past_the_end_is_skipped():
    scores = torch.arange(6, dtype=torch.float32)
    assert coverage(scores, [(6, 9)], budget=2, n_tokens=6) == []
