"""The learned admission head.

Reads what the frozen model already computed and emits one admit/skip decision
per token, shared across layers (spec 4.2 - per-layer decisions desynchronise
layer lengths and break the single global attention mask).

Small on purpose: the base model is frozen, so this is the only thing carrying
gradients, and it must be cheap enough to run once per chunk per rollout.
"""

import torch
from torch import nn


class AdmissionPolicy(nn.Module):
    """Bernoulli admission policy with a shared value head.

    extra features, in column order: [k_norm, rel_pos, occupancy_frac, lam]
    """

    def __init__(self, hidden_size, proj=128, n_extra=4):
        super().__init__()
        self.proj = nn.Linear(hidden_size, proj)
        self.head = nn.Sequential(
            nn.Linear(proj + n_extra, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(proj + n_extra, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, h, extra):
        z = torch.cat([torch.tanh(self.proj(h)), extra], dim=-1)
        return self.head(z).squeeze(-1), self.value(z).squeeze(-1)

    def act(self, h, extra, generator=None):
        """Sample admission decisions.

        Returns (keep, summed log_prob, mean value, mean entropy). Entropy is
        returned so the learner can penalise premature determinism: phase 2b
        collapsed into rejecting everything and could not recover, because once
        every logit is confidently negative all sampled actions are rejections
        and the gradient only reinforces them.
        """
        logits, values = self(h, extra)
        dist = torch.distributions.Bernoulli(logits=logits)
        probs = torch.sigmoid(logits)
        noise = torch.rand(probs.shape, generator=generator, device=probs.device)
        keep = noise < probs.detach()
        log_prob = dist.log_prob(keep.float()).sum()
        return keep, log_prob, values.mean(), dist.entropy().mean()
