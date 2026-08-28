import torch

from kvadmission.policy import AdmissionPolicy


def _inputs(n=16, hidden=64):
    torch.manual_seed(0)
    return torch.randn(n, hidden), torch.randn(n, 4)


def test_forward_shapes():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    logits, values = p(h, extra)
    assert logits.shape == (16,)
    assert values.shape == (16,)


def test_act_returns_bool_mask_of_right_length():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    keep, log_prob, value, ent = p.act(h, extra)
    assert keep.shape == (16,)
    assert keep.dtype == torch.bool
    assert log_prob.ndim == 0
    assert value.ndim == 0


def test_act_is_reproducible_under_a_generator():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    a, *_ = p.act(h, extra, generator=torch.Generator().manual_seed(3))
    b, *_ = p.act(h, extra, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_log_prob_matches_bernoulli():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    keep, log_prob, _, _ = p.act(h, extra, generator=torch.Generator().manual_seed(1))
    logits, _ = p(h, extra)
    expected = torch.distributions.Bernoulli(logits=logits).log_prob(
        keep.float()
    ).sum()
    assert torch.allclose(log_prob, expected, atol=1e-5)


def test_gradients_reach_every_parameter():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    _, log_prob, value, _ = p.act(h, extra)
    (log_prob + value).backward()
    missing = [n for n, q in p.named_parameters() if q.grad is None]
    assert missing == [], f"no gradient reached: {missing}"


def test_confident_logits_drive_the_decision():
    """A large positive bias must admit everything; large negative, nothing."""
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    with torch.no_grad():
        p.head[-1].bias.fill_(20.0)
    keep, *_ = p.act(h, extra)
    assert keep.all()
    with torch.no_grad():
        p.head[-1].bias.fill_(-20.0)
    keep, *_ = p.act(h, extra)
    assert not keep.any()


def test_parameter_count_is_small():
    p = AdmissionPolicy(hidden_size=2048)
    total = sum(q.numel() for q in p.parameters())
    assert total < 1_500_000, f"{total} parameters is too large for a frozen-base head"


def test_entropy_is_high_when_undecided_and_low_when_confident():
    """The learner uses this to resist the collapse that killed phase 2b."""
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    with torch.no_grad():
        p.head[-1].bias.fill_(0.0)
    _, _, _, undecided = p.act(h, extra)
    with torch.no_grad():
        p.head[-1].bias.fill_(20.0)
    _, _, _, confident = p.act(h, extra)
    assert undecided > confident
    assert confident < 0.01
