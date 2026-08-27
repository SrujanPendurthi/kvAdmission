from kvadmission.sweep import summarize


def _row(policy, budget, seed, d, hit, stream_len=8192):
    return {"policy": policy, "budget": budget, "seed": seed,
            "topic": "oak", "distance_back": d, "stream_len": stream_len,
            "hit": hit, "gen": ""}


def test_summarize_bands_by_distance_tercile():
    rows = [
        _row("recent", 2048, 0, 100, True),    # near
        _row("recent", 2048, 0, 4000, False),  # middle
        _row("recent", 2048, 0, 7900, False),  # far
    ]
    out = summarize(rows)
    assert out[("recent", 2048, "near")]["mean"] == 1.0
    assert out[("recent", 2048, "middle")]["mean"] == 0.0
    assert out[("recent", 2048, "far")]["mean"] == 0.0


def test_summarize_averages_across_seeds():
    rows = [
        _row("h2o", 1024, 0, 100, True),
        _row("h2o", 1024, 1, 100, False),
    ]
    out = summarize(rows)
    cell = out[("h2o", 1024, "near")]
    assert cell["mean"] == 0.5
    assert cell["n"] == 2


def test_summarize_reports_stderr():
    rows = [_row("h2o", 1024, s, 100, s % 2 == 0) for s in range(4)]
    cell = summarize(rows)[("h2o", 1024, "near")]
    assert cell["n"] == 4
    assert 0.0 < cell["stderr"] < 0.5


def test_stderr_is_zero_when_unanimous():
    rows = [_row("recent", 512, s, 100, True) for s in range(4)]
    assert summarize(rows)[("recent", 512, "near")]["stderr"] == 0.0


def test_zero_hit_cells_report_rule_of_three_bound():
    """stderr is 0.0 for an all-zero cell, which overstates certainty."""
    rows = [_row("h2o", 512, s, 4000, False) for s in range(6)]
    cell = summarize(rows)[("h2o", 512, "middle")]
    assert cell["mean"] == 0.0
    assert cell["stderr"] == 0.0
    assert cell["upper95"] == 0.5  # 3/6


def test_upper_bound_tightens_with_more_trials():
    few = summarize([_row("h2o", 512, s, 4000, False) for s in range(6)])
    many = summarize([_row("h2o", 512, s, 4000, False) for s in range(60)])
    assert many[("h2o", 512, "middle")]["upper95"] < \
        few[("h2o", 512, "middle")]["upper95"]


def test_nonzero_cells_use_normal_approximation():
    rows = [_row("recent", 2048, s, 100, s < 5) for s in range(10)]
    cell = summarize(rows)[("recent", 2048, "near")]
    assert cell["mean"] == 0.5
    assert 0.5 < cell["upper95"] < 1.0
