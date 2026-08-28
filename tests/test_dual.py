import torch

from kvadmission.dual import dual_prefill
from kvadmission.runner import load_model

MODEL, TOK, DEVICE = load_model()


def _ids(n=512):
    text = "def alpha(x):\n    return x + 1\n\n" * 200
    return TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :n]


def _admit_all(layer_idx, cache, attentions, n_new):
    cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))


def test_kl_is_zero_when_budget_is_unbounded():
    """No compression means the two caches are identical, so KL must vanish."""
    ids = _ids()
    kl, n = dual_prefill(MODEL, ids, DEVICE, budget=None, decide=None, chunk=128)
    assert n == ids.shape[1]
    assert kl.shape == (ids.shape[1],)
    assert kl.max().item() < 1e-4, f"max KL {kl.max().item()}"


def test_kl_is_positive_when_compressing_hard():
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert kl.max().item() > 0.01, f"max KL {kl.max().item()}"


def test_kl_is_never_negative():
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert kl.min().item() >= -1e-5, f"min KL {kl.min().item()}"


def test_tighter_budget_gives_more_divergence():
    ids = _ids()
    loose, _ = dual_prefill(MODEL, ids, DEVICE, budget=256, decide=_admit_all, chunk=128)
    tight, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert tight.mean().item() > loose.mean().item()


def test_early_positions_have_near_zero_kl():
    """Before the budget is exhausted nothing has been evicted yet."""
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=256, decide=_admit_all, chunk=128)
    assert kl[:120].max().item() < 1e-4
