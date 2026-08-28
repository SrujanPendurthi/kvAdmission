"""Average-reward policy gradient with a value baseline.

Differential TD rather than discounted return. This is a continuing task with
no episode boundary, where a discount factor silently imposes a horizon - and
if that horizon is shorter than the distance at which retention pays off, the
optimal discounted policy is to drop everything.

Plain policy gradient per spec 4.3. No PPO until this demonstrably fails.
"""

import torch


class AverageRewardLearner:
    def __init__(self, policy, lr=3e-4, alpha=0.01, value_coef=0.5,
                 grad_clip=1.0, entropy_coef=0.01):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.alpha = alpha
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef
        # None until the first step. Starting at 0.0 when rewards sit near 0.5
        # makes every early delta strongly positive, which reinforces whatever
        # the policy happened to do first - it collapses to a fixed point long
        # before a slow-moving r_bar can catch up and supply any contrast.
        self.r_bar = None

    def step(self, log_probs, values, rewards, entropies=None):
        """One update over a rollout of chunk-level timesteps."""
        n = len(rewards)
        assert len(log_probs) == len(values) == n, "ragged rollout"

        if self.r_bar is None:
            self.r_bar = sum(rewards) / n

        deltas, policy_terms, value_terms = [], [], []
        for k in range(n):
            # bootstrap off the next state; the final step has no successor
            v_next = values[k + 1].detach() if k + 1 < n else torch.zeros(())
            delta = rewards[k] - self.r_bar + v_next - values[k]
            deltas.append(delta.detach())
            policy_terms.append(-log_probs[k] * delta.detach())
            value_terms.append(delta.pow(2))

        mean_delta = torch.stack(deltas).mean()
        self.r_bar += self.alpha * mean_delta.item()

        policy_loss = torch.stack(policy_terms).mean()
        value_loss = torch.stack(value_terms).mean()
        entropy = (torch.stack(entropies).mean() if entropies
                   else torch.zeros((), device=policy_loss.device))
        # subtracted: maximising entropy keeps the policy stochastic enough to
        # escape a deterministic state rather than reinforcing it
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "r_bar": self.r_bar,
            "mean_delta": mean_delta.item(),
            "entropy": entropy.item(),
        }
