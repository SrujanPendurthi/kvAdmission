"""Dual ascent on the cache-occupancy constraint.

Treats memory as a constraint rather than a penalty term: lambda is a shadow
price on cache space that rises while occupancy is above target and falls when
it is below, so the trade-off tunes itself instead of being swept by hand.

Not novel - TriRoute (arXiv 2607.06601) applies the same dual ascent to KV
allocation. Used here for the practical benefit.
"""


class LagrangianController:
    """lambda_{t+1} = clip(lambda_t + eta * (occupancy - target), 0, lam_max)."""

    def __init__(self, target, eta=0.01, lam=0.0, lam_max=10.0):
        self.target = target
        self.eta = eta
        self.lam = lam
        self.lam_max = lam_max

    def update(self, occupancy_frac):
        """One dual-ascent step. Returns the new multiplier."""
        self.lam += self.eta * (occupancy_frac - self.target)
        # clipped below at 0 (a negative price would pay to fill the cache) and
        # above at lam_max, which stops runaway when the target is unreachable
        self.lam = max(0.0, min(self.lam_max, self.lam))
        return self.lam
