from kvadmission.lagrangian import LagrangianController


def test_lambda_rises_when_over_target():
    c = LagrangianController(target=0.5, eta=0.1)
    c.update(0.9)
    assert c.lam > 0.0


def test_lambda_falls_when_under_target():
    c = LagrangianController(target=0.5, eta=0.1, lam=1.0)
    before = c.lam
    c.update(0.1)
    assert c.lam < before


def test_lambda_never_goes_negative():
    c = LagrangianController(target=0.5, eta=0.5)
    for _ in range(50):
        c.update(0.0)
    assert c.lam == 0.0


def test_lambda_is_capped():
    c = LagrangianController(target=0.1, eta=1.0, lam_max=3.0)
    for _ in range(100):
        c.update(1.0)
    assert c.lam == 3.0


def test_lambda_drives_occupancy_to_target():
    """Closed loop: occupancy falls as the price of cache space rises."""
    c = LagrangianController(target=0.6, eta=0.2)
    occupancy = 1.0
    for _ in range(300):
        c.update(occupancy)
        occupancy = 1.0 / (1.0 + c.lam)  # more expensive space, less of it used
    assert abs(occupancy - 0.6) < 0.05, f"settled at {occupancy}"


def test_update_returns_the_new_lambda():
    c = LagrangianController(target=0.5, eta=0.1)
    assert c.update(0.9) == c.lam
