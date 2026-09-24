import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvadmission.graded import (COST, EVICTED, GradedCache, _probe_logprobs,
                                fake_quant, graded_prefill)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=32,
                      attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval()


def test_fake_quant_error_shrinks_with_bits():
    x = torch.randn(2, 3, 5, 64)
    errs = [(fake_quant(x, b) - x).abs().mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2] > 0
    assert torch.equal(fake_quant(x, 16), x)


def test_masked_probe_equals_real_eviction(tiny):
    """step_kl scores eviction by masking; masking must match evicting."""
    ids = torch.randint(0, 64, (1, 48))
    probe = torch.randint(0, 64, (1, 8))
    cache = GradedCache(budget=10_000, sinks=4)
    graded_prefill(tiny, ids, cache, "cpu", "kl", chunk=16)
    idx = cache.groups(16)[1]
    masked = _probe_logprobs(tiny, cache, probe, 48, "cpu", idx)
    cache.evict(idx)
    evicted = _probe_logprobs(tiny, cache, probe, 48, "cpu")
    assert torch.allclose(masked, evicted, atol=1e-4)


@pytest.mark.parametrize("policy,level", [("uniform", 0), ("uniform", 2),
                                          ("age", 0), ("kl", 0)])
def test_budget_holds_and_sinks_survive(tiny, policy, level):
    ids = torch.randint(0, 64, (1, 160))
    cache = GradedCache(budget=48, sinks=4)
    graded_prefill(tiny, ids, cache, "cpu", policy, chunk=16, probe=8,
                   write_level=level, refresh=2)
    assert cache.used() <= cache.capacity
    assert cache.pos[:4].tolist() == [0, 1, 2, 3]
    assert cache.occupancy(0) == cache.occupancy(1) == cache.level.shape[0]
    assert (cache.level < EVICTED).all()
    if policy != "uniform":
        # the ladder should trade precision for tokens, keeping more than fp16
        assert cache.level.shape[0] > 48


def test_uniform_int4_holds_more_tokens_than_fp16(tiny):
    ids = torch.randint(0, 64, (1, 320))
    n = {}
    for level in (0, 2):
        cache = GradedCache(budget=64, sinks=4)
        graded_prefill(tiny, ids, cache, "cpu", "uniform", chunk=16,
                       write_level=level)
        n[level] = cache.level.shape[0]
    assert n[2] >= n[0] * COST[0] / COST[2] * 0.8


def test_merge_halves_entries_and_keeps_layers_aligned(tiny):
    ids = torch.randint(0, 64, (1, 64))
    cache = GradedCache(budget=10_000, sinks=4)
    graded_prefill(tiny, ids, cache, "cpu", "uniform", chunk=16, merge=2)
    # first chunk (holds the sinks) unmerged, three chunks of 16 -> 8 each
    assert cache.level.shape[0] == 16 + 3 * 8
    assert cache.occupancy(0) == cache.occupancy(1) == 40
    assert cache.pos[16:19].tolist() == [16, 18, 20]


def test_fifo_can_evict_several_chunks_in_one_compaction(tiny):
    """Budget below two chunks forces multi-chunk eviction; indices must not go stale."""
    ids = torch.randint(0, 64, (1, 128))
    cache = GradedCache(budget=20, sinks=4)
    graded_prefill(tiny, ids, cache, "cpu", "uniform", chunk=16)
    assert cache.used() <= cache.capacity
    assert cache.pos[:4].tolist() == [0, 1, 2, 3]


def test_scored_fifo_drops_the_lowest_scoring_tokens(tiny):
    """With scores, uniform eviction keeps the top scorers, not the newest."""
    ids = torch.randint(0, 64, (1, 128))
    scores = torch.zeros(128)
    scores[20:40] = 1.0          # an early, high-scoring span
    cache = GradedCache(budget=48, sinks=4)
    graded_prefill(tiny, ids, cache, "cpu", "uniform", chunk=16, scores=scores)
    kept = set(cache.pos.tolist())
    assert set(range(20, 40)) <= kept
