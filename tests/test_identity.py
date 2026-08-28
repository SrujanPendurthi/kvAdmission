import torch

from kvadmission.cache import AdmissionCache
from kvadmission.runner import load_model, prefill, generate

MODEL, TOK, DEVICE = load_model()


def _ids(n_tokens=600):
    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids[:, :n_tokens]


def _final_logits(cache, chunk):
    ids = _ids()
    prefill(MODEL, ids, cache, DEVICE, chunk=chunk)
    with torch.no_grad():
        out = MODEL(
            ids[:, -1:].to(DEVICE),
            past_key_values=cache,
            position_ids=torch.tensor([[ids.shape[1]]], device=DEVICE),
            attention_mask=torch.ones(
                1, cache.get_seq_length() + 1, device=DEVICE, dtype=torch.long
            ),
            use_cache=True,
        )
    return out.logits[0, -1].float().cpu()


def test_unbounded_cache_is_bit_identical_to_dynamic_cache():
    """The identity test. Same chunking, only the cache class differs."""
    from transformers import DynamicCache

    a = _final_logits(AdmissionCache(budget=None), chunk=128)
    b = _final_logits(DynamicCache(), chunk=128)
    assert torch.equal(a, b), f"max abs diff {(a - b).abs().max().item()}"


def test_chunked_prefill_matches_single_pass():
    """Catches position_ids and attention-mask bugs.

    Tolerance is set from bf16 resolution, not wishfully: one ulp near a logit
    of 13 is 0.0625, and chunked vs single-pass accumulation lands within a few
    of those. atol=0.5 is ~8 ulps - far above float noise, far below what a
    position bug produces (see the test below, which proves that).
    """
    a = _final_logits(AdmissionCache(budget=None), chunk=128)
    b = _final_logits(AdmissionCache(budget=None), chunk=1024)
    assert a.argmax() == b.argmax(), "chunking changed the predicted token"
    assert torch.allclose(a, b, atol=0.5), \
        f"max abs diff {(a - b).abs().max().item()}"


def test_wrong_positions_are_actually_detected():
    """Proof the tolerance above has teeth: corrupt positions must blow past it."""
    ids = _ids()
    cache = AdmissionCache(budget=None)
    prefill(MODEL, ids, cache, DEVICE, chunk=128)
    with torch.no_grad():
        good = MODEL(
            ids[:, -1:].to(DEVICE), past_key_values=cache,
            position_ids=torch.tensor([[ids.shape[1]]], device=DEVICE),
            attention_mask=torch.ones(
                1, cache.get_seq_length() + 1, device=DEVICE, dtype=torch.long
            ),
            use_cache=True,
        ).logits[0, -1].float().cpu()
    cache.crop(ids.shape[1])
    with torch.no_grad():
        bad = MODEL(
            ids[:, -1:].to(DEVICE), past_key_values=cache,
            position_ids=torch.tensor([[0]], device=DEVICE),  # wrong on purpose
            attention_mask=torch.ones(
                1, cache.get_seq_length() + 1, device=DEVICE, dtype=torch.long
            ),
            use_cache=True,
        ).logits[0, -1].float().cpu()
    assert not torch.allclose(good, bad, atol=0.5), \
        "tolerance is too loose - a position bug would slip through"


def test_prefill_returns_stream_length():
    ids = _ids(600)
    cache = AdmissionCache(budget=None)
    assert prefill(MODEL, ids, cache, DEVICE, chunk=128) == 600


def test_generate_restores_cache_length():
    ids = _ids(600)
    cache = AdmissionCache(budget=256, sinks=4)

    def decide(layer_idx, cache, attentions, n_new):
        cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))

    stream_len = prefill(MODEL, ids, cache, DEVICE, chunk=128, decide=decide)
    before = cache.occupancy(0)
    generate(MODEL, TOK, cache, stream_len, "\n\nQuestion: What?\nAnswer:", DEVICE)
    assert cache.occupancy(0) == before


def test_generate_produces_text():
    ids = _ids(600)
    cache = AdmissionCache(budget=None)
    stream_len = prefill(MODEL, ids, cache, DEVICE, chunk=128)
    text = generate(MODEL, TOK, cache, stream_len, "\n\nThe fox is", DEVICE, max_new=8)
    assert isinstance(text, str) and len(text) > 0


def test_base_model_parameters_are_frozen():
    """Nothing in this project may ever backprop into the base model."""
    trainable = [n for n, q in MODEL.named_parameters() if q.requires_grad]
    assert trainable == [], f"{len(trainable)} base params still require grad"


def test_pick_device_returns_something_torch_accepts():
    from kvadmission.runner import pick_device
    assert pick_device() in ("cuda", "mps", "cpu")
