import pytest

from pricer import black_scholes as bs
from pricer.binomial import american_premium, crr_price

P = dict(S=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_converges_to_black_scholes(kind):
    assert crr_price(kind, **P, steps=2000) == pytest.approx(float(bs.price(kind, **P)), abs=5e-3)


def test_american_call_no_dividend_equals_european():
    # never optimal to exercise a call early without dividends
    assert american_premium("call", **P, steps=500) == pytest.approx(0.0, abs=1e-10)


def test_american_put_has_early_exercise_premium():
    assert american_premium("put", **P, steps=500) > 0.05


def test_american_call_with_dividends_can_exceed_european():
    assert american_premium("call", **P, q=0.08, steps=500) > 0.0
