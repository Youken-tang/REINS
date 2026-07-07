from app.services.inventory import reconcile


def test_classic_floating_point_pair():
    assert reconcile(0.1 + 0.2, 0.3)

def test_genuinely_different():
    assert not reconcile(1.0, 2.0)
