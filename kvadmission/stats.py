"""Interval estimates for the small samples every table in the paper reports."""
import math
import statistics


def wilson(k, n, z=1.96):
    """95% Wilson score interval. Well-behaved at k=0 and k=n, unlike Wald."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def mean_ci(xs, z=1.96):
    """Mean and normal-approximation 95% interval over per-span values."""
    m = statistics.mean(xs)
    h = z * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, max(0.0, m - h), min(1.0, m + h)
