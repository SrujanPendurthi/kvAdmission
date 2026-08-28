import torch
from transformers import AutoTokenizer

from kvadmission.signal import (
    build_code_stream, code_corpus, long_dependency_positions,
)

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_corpus_finds_local_python_sources():
    paths = code_corpus(max_files=50)
    assert len(paths) > 10
    assert all(p.endswith(".py") for p in paths)


def test_corpus_excludes_the_eval_haystack():
    """Training on the eval corpus would invalidate the whole phase."""
    paths = code_corpus(max_files=400)
    assert not any("haystack" in p or "1342" in p for p in paths)


def test_code_stream_has_exact_length():
    ids = build_code_stream(TOK, target_len=2048)
    assert ids.shape == (1, 2048)


def test_code_stream_is_reproducible_and_seed_varies():
    a = build_code_stream(TOK, 1024, seed=1)
    b = build_code_stream(TOK, 1024, seed=1)
    c = build_code_stream(TOK, 1024, seed=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_long_dependency_positions_found_in_real_code():
    ids = build_code_stream(TOK, 8192, seed=0)
    pos = long_dependency_positions(TOK, ids, min_distance=2048)
    assert len(pos) > 5, f"only {len(pos)} long-dependency positions"
    assert all(0 <= p < 8192 for p in pos)


def test_long_dependency_respects_min_distance():
    """A synthetic stream whose only repeat is close by must yield nothing."""
    text = "alpha_variable = 1\nalpha_variable = 2\n" * 3
    ids = TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert long_dependency_positions(TOK, ids, min_distance=2048) == []
