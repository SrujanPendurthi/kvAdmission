import torch

from kvadmission.lagrangian import LagrangianController
from kvadmission.policy import AdmissionPolicy
from kvadmission.runner import load_model
from kvadmission.signal import build_code_stream
from kvadmission.train import rollout, weighted_kl

MODEL, TOK, DEVICE = load_model()


def _setup():
    policy = AdmissionPolicy(hidden_size=MODEL.config.hidden_size).to(DEVICE)
    controller = LagrangianController(target=0.9, eta=0.05)
    ids = build_code_stream(TOK, 1024, seed=0)
    return policy, controller, ids


def test_weighted_kl_equals_mean_when_beta_is_zero():
    kl = torch.tensor([0.1, 0.5, 2.0, 0.0])
    assert torch.allclose(weighted_kl(kl, 0.0), kl.mean())


def test_weighting_lifts_a_skewed_distribution_above_its_mean():
    """The whole point: rare high-KL positions must not be averaged away."""
    kl = torch.cat([torch.full((99,), 0.01), torch.tensor([2.0])])
    assert weighted_kl(kl, 1.0) > 10 * kl.mean()


def test_weighting_is_a_noop_on_a_uniform_distribution():
    kl = torch.full((50,), 0.3)
    assert torch.allclose(weighted_kl(kl, 1.0), kl.mean(), atol=1e-6)


def test_weighted_kl_survives_an_all_zero_chunk():
    kl = torch.zeros(20)
    assert torch.isfinite(weighted_kl(kl, 1.0))


def test_rollout_returns_one_entry_per_chunk():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert len(out["rewards"]) == 8
    assert len(out["log_probs"]) == 8
    assert len(out["values"]) == 8
    assert len(out["kl_mean"]) == 8


def test_rollout_log_probs_carry_gradient():
    """The base model runs under no_grad; the head must still build a graph."""
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert out["log_probs"][0].requires_grad
    out["log_probs"][0].backward()
    assert policy.proj.weight.grad is not None


def test_rollout_respects_the_budget():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert max(out["occupancy"]) <= 256


def test_rewards_are_finite_and_non_positive():
    """Reward is negative KL minus a non-negative occupancy price."""
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert all(r <= 1e-6 for r in out["rewards"])
    assert all(abs(r) < 1e6 for r in out["rewards"])


def test_lambda_moves_during_a_rollout():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert len(set(out["lam"])) > 1, "lambda never updated"


def test_beta_one_reports_higher_kl_than_the_unweighted_mean():
    """On real chunks the KL distribution is skewed, so weighting must lift it."""
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=128, chunk=128,
                  beta=1.0)
    weighted = [k for k, m in zip(out["kl"], out["kl_mean"]) if m > 1e-6]
    plain = [m for m in out["kl_mean"] if m > 1e-6]
    assert weighted, "no chunk had nonzero KL"
    assert sum(weighted) > sum(plain)
