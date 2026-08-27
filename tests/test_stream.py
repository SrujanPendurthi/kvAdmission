import torch
from transformers import AutoTokenizer

from kvadmission.stream import build_stream

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_stream_has_requested_length():
    ids, _ = build_stream(TOK, n_needles=4, target_len=2048)
    assert ids.shape == (1, 2048)


def test_all_needles_are_placed_and_distinct():
    ids, needles = build_stream(TOK, n_needles=6, target_len=4096)
    assert len(needles) == 6
    assert len({n["topic"] for n in needles}) == 6
    assert len({n["code"] for n in needles}) == 6


def test_needle_positions_are_increasing_and_in_range():
    ids, needles = build_stream(TOK, n_needles=6, target_len=4096)
    positions = [n["pos"] for n in needles]
    assert positions == sorted(positions)
    assert all(0 < p <= 4096 for p in positions)


def test_needle_text_is_actually_present_in_the_stream():
    ids, needles = build_stream(TOK, n_needles=3, target_len=2048)
    text = TOK.decode(ids[0])
    for n in needles:
        assert n["code"] in text
        assert n["topic"] in text


def test_seed_changes_placement_but_not_count():
    _, a = build_stream(TOK, n_needles=5, target_len=4096, seed=1)
    _, b = build_stream(TOK, n_needles=5, target_len=4096, seed=2)
    assert len(a) == len(b) == 5
    assert [n["pos"] for n in a] != [n["pos"] for n in b]


def test_seed_is_reproducible():
    _, a = build_stream(TOK, n_needles=5, target_len=4096, seed=3)
    _, b = build_stream(TOK, n_needles=5, target_len=4096, seed=3)
    assert [n["pos"] for n in a] == [n["pos"] for n in b]
    assert [n["code"] for n in a] == [n["code"] for n in b]
