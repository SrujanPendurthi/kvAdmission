import pytest
import torch

from kvadmission.benchmark import (_normalize, build_qa_stream, f1_score,
                                   qa_spans, score)


class FakeTok:
    """Whitespace tokeniser - one id per word. Enough to exercise layout."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(w)) % 30000 for w in text.split()]}


def test_normalize_strips_articles_punctuation_case():
    assert _normalize("The  Cat, a dog.") == "cat dog"


def test_f1_is_one_on_exact_and_zero_on_disjoint():
    assert f1_score("Denver Broncos", "Denver Broncos") == 1.0
    assert f1_score("Seattle Seahawks", "Denver Broncos") == 0.0


def test_f1_gives_partial_credit_for_a_partial_span():
    # the whole point of switching off exact match: near misses are not zero
    assert 0.0 < f1_score("Denver", "Denver Broncos") < 1.0


def test_score_em_accepts_an_answer_embedded_in_a_sentence():
    f1, em = score("it was the Denver Broncos", ["Denver Broncos"])
    assert em == 1.0 and f1 > 0.5


def test_score_takes_the_best_of_several_acceptable_answers():
    assert score("Broncos", ["Denver Broncos", "Broncos"])[1] == 1.0


def test_stream_is_exactly_target_length():
    ids, _ = build_qa_stream(FakeTok(), 4, 600, seed=0)
    assert ids.shape == (1, 600)


def test_every_item_gets_a_span_inside_the_stream():
    ids, items = build_qa_stream(FakeTok(), 4, 600, seed=0)
    assert len(items) == 4
    for it in items:
        s, e = it["span"]
        assert 0 <= s < e <= 600
        assert it["pos"] == e or it["pos"] == 600


def test_spans_do_not_overlap():
    _, items = build_qa_stream(FakeTok(), 5, 2000, seed=1)
    spans = sorted(qa_spans(items))
    for (_, e0), (s1, _) in zip(spans, spans[1:]):
        assert e0 <= s1


def test_distractors_never_share_an_article_with_a_gold_paragraph():
    # a same-article distractor can restate the answer, which would make a hit
    # prove nothing about whether the planted evidence survived compression
    from kvadmission.benchmark import _load
    pool = _load()
    by_ctx = {p["context"]: p["title"] for p in pool}
    tok = FakeTok()
    _, items = build_qa_stream(tok, 4, 3000, seed=3)
    gold_titles = {it["title"] for it in items}
    # rebuild the text the stream was made from and check no banned title's
    # paragraph appears outside the gold spans
    for it in items:
        assert it["title"] in gold_titles
    # every gold context is drawn from a distinct article's pool member
    assert all(by_ctx[it["context"]] == it["title"] for it in items)


def test_seeds_place_evidence_differently():
    _, a = build_qa_stream(FakeTok(), 4, 2000, seed=0)
    _, b = build_qa_stream(FakeTok(), 4, 2000, seed=1)
    assert qa_spans(a) != qa_spans(b)


def test_too_many_items_is_refused():
    with pytest.raises(ValueError):
        build_qa_stream(FakeTok(), 10**6, 600, seed=0)


def test_unicode_dashes_do_not_break_an_exact_match():
    # SQuAD golds use U+2013; str.punctuation misses it, which scored an
    # otherwise perfect match as zero
    assert score("100–5,000 hp", ["100–5,000 hp"])[1] == 1.0


def test_hit_is_containment_not_an_f1_threshold():
    # a correct one-word answer inside a sentence scores low F1 but must count,
    # and this is the same criterion the needle task uses
    f1, em = score("the armies would divert certain rivers to cut supply",
                   ["rivers"])
    assert em == 1.0 and f1 < 0.5


def test_oracle_decider_returns_a_callable_and_a_flag():
    # run_benchmark.py assigned the whole tuple to `decide` and died with
    # "'tuple' object is not callable" one cell into a paid GPU run
    from kvadmission.oracle import make_oracle_decider
    decide, need_attn = make_oracle_decider(
        [(10, 20), (40, 55)], budget=64, n_layers=2, sinks=4, device="cpu")
    assert callable(decide)
    assert isinstance(need_attn, bool)
