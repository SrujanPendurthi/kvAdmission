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


def test_sentence_answers_are_the_longest():
    """A 6-digit code is ~6 tokens (one per digit), so it is not the shortest -
    it is the least redundant. Only the sentence style is reliably longer."""
    lens = {}
    for style in ("code", "phrase", "sentence"):
        _, needles = build_stream(TOK, 3, 2048, seed=0, style=style)
        lens[style] = sum(
            len(TOK(n["code"], add_special_tokens=False)["input_ids"])
            for n in needles) / len(needles)
    assert lens["sentence"] > lens["code"], lens
    assert lens["sentence"] > lens["phrase"], lens


def test_every_style_plants_its_answer_in_the_stream():
    for style in ("code", "phrase", "sentence"):
        ids, needles = build_stream(TOK, 3, 2048, seed=1, style=style)
        text = TOK.decode(ids[0])
        for n in needles:
            assert n["code"] in text, (style, n["code"])


def test_style_is_recorded_on_each_needle():
    _, needles = build_stream(TOK, 3, 2048, seed=0, style="phrase")
    assert all(n["style"] == "phrase" for n in needles)
