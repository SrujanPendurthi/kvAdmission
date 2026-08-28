import torch

from kvadmission.learner import AverageRewardLearner
from kvadmission.policy import AdmissionPolicy


def _rollout(policy, rewards, gen):
    log_probs, values = [], []
    for _ in rewards:
        h = torch.randn(8, 64)
        extra = torch.zeros(8, 4)
        _, lp, v, _ = policy.act(h, extra, generator=gen)
        log_probs.append(lp)
        values.append(v)
    return log_probs, values


def test_step_returns_diagnostics():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 0.5, 0.2], gen)
    out = learner.step(lp, v, [1.0, 0.5, 0.2])
    for key in ("policy_loss", "value_loss", "r_bar", "mean_delta"):
        assert key in out and isinstance(out[key], float)


def test_r_bar_tracks_the_reward_level():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p, alpha=0.2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        lp, v = _rollout(p, [2.0] * 4, gen)
        learner.step(lp, v, [2.0] * 4)
    assert 0.5 < learner.r_bar < 3.5, f"r_bar drifted to {learner.r_bar}"


def test_parameters_actually_move():
    p = AdmissionPolicy(hidden_size=64)
    before = p.proj.weight.detach().clone()
    learner = AverageRewardLearner(p, lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 1.0], gen)
    learner.step(lp, v, [1.0, 1.0])
    assert not torch.equal(before, p.proj.weight.detach())


def test_learns_a_toy_task_where_admitting_pays():
    """Reward proportional to how much was admitted - policy must trend to admit."""
    torch.manual_seed(0)
    p = AdmissionPolicy(hidden_size=8)
    learner = AverageRewardLearner(p, lr=5e-2, alpha=0.05)
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(32, 8)
    extra = torch.zeros(32, 4)

    def admit_rate():
        with torch.no_grad():
            logits, _ = p(h, extra)
            return torch.sigmoid(logits).mean().item()

    start = admit_rate()
    for _ in range(120):
        log_probs, values, rewards = [], [], []
        for _ in range(4):
            keep, lp, v, _ = p.act(h, extra, generator=gen)
            log_probs.append(lp)
            values.append(v)
            rewards.append(keep.float().mean().item())
        learner.step(log_probs, values, rewards)
    assert admit_rate() > start + 0.05, f"{start:.3f} -> {admit_rate():.3f}"


def test_entropy_bonus_reaches_the_loss():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p, entropy_coef=0.5)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 1.0], gen)
    ents = [torch.tensor(0.5, requires_grad=True) for _ in range(2)]
    out = learner.step(lp, v, [1.0, 1.0], entropies=ents)
    assert out["entropy"] == 0.5


def test_step_still_works_without_entropies():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 1.0], gen)
    assert learner.step(lp, v, [1.0, 1.0])["entropy"] == 0.0
