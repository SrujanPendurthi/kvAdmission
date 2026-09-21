from kvadmission.stats import mean_ci, wilson


def test_wilson_matches_known_values():
    lo, hi = wilson(0, 340)
    assert lo == 0.0 and abs(hi - 0.0111) < 1e-3     # z^2 / (n + z^2)
    lo, hi = wilson(13, 24)
    assert abs(lo - 0.351) < 2e-3 and abs(hi - 0.720) < 2e-3


def test_wilson_stays_inside_unit_interval_at_the_extremes():
    for k, n in ((0, 1), (1, 1), (24, 24), (0, 24)):
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_mean_ci_brackets_the_mean_and_collapses_on_constant_data():
    m, lo, hi = mean_ci([0.0, 0.5, 1.0, 0.25])
    assert lo < m < hi
    assert mean_ci([0.3, 0.3, 0.3]) == (0.3, 0.3, 0.3)
