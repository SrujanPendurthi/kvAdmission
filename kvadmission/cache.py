"""Budget-capped KV cache supporting write-time admission and score eviction."""

import torch
from transformers import DynamicCache


class AdmissionCache(DynamicCache):
    """KV cache with a per-layer token budget and a pinned sink prefix.

    The model writes every token through the normal `update()` path. After each
    forward pass the caller applies a decision per layer:

    - `apply_admission(layer_idx, keep)` drops the just-written tokens the policy
      rejected, then enforces the budget by evicting the oldest non-sink
      residents (FIFO). This is the mode our learned policy uses.
    - `apply_eviction(layer_idx, scores)` keeps the top-`budget` residents by
      score. This is the mode published baselines (H2O, SnapKV) use.

    Peak memory is `budget + chunk_size`, not the full stream length: tokens are
    written and then removed within a chunk rather than accumulating.

    `budget=None` disables both, making this behave exactly as DynamicCache. That
    is the identity mode the correctness test relies on.
    """

    def __init__(self, budget=None, sinks=4, **kwargs):
        super().__init__(**kwargs)
        self.budget = budget
        self.sinks = sinks

    def occupancy(self, layer_idx=0):
        """Number of tokens currently resident in a layer."""
        if layer_idx >= len(self.layers):
            return 0
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            return 0
        return layer.keys.shape[-2]

    def _reindex(self, layer_idx, surviving):
        layer = self.layers[layer_idx]
        layer.keys = layer.keys[:, :, surviving, :]
        layer.values = layer.values[:, :, surviving, :]

    def _enforce_budget_fifo(self, surviving):
        """Trim to budget by dropping the oldest non-sink entries."""
        if surviving.numel() <= self.budget:
            return surviving
        n_sink = min(self.sinks, surviving.numel())
        room = self.budget - n_sink
        if room <= 0:
            return surviving[: self.budget]
        return torch.cat([surviving[:n_sink], surviving[n_sink:][-room:]])

    def apply_admission(self, layer_idx, keep):
        """Drop rejected new tokens, then FIFO down to budget.

        `keep` is a bool tensor over the tokens written by the last update().
        Returns the surviving positions as indices into the pre-call cache, so
        callers can reindex parallel per-token state with `state[surviving]`.
        """
        n = self.occupancy(layer_idx)
        device = self.layers[layer_idx].keys.device
        idx = torch.arange(n, device=device)

        if self.budget is None:
            return idx

        n_new = keep.shape[0]
        n_old = n - n_new
        keep = keep.to(device)

        # The sink prefix is structural, not a policy choice - force it in.
        if n_old < self.sinks:
            keep = keep.clone()
            keep[: min(self.sinks - n_old, n_new)] = True

        surviving = torch.cat([idx[:n_old], idx[n_old:][keep]])
        surviving = self._enforce_budget_fifo(surviving)
        self._reindex(layer_idx, surviving)
        return surviving
