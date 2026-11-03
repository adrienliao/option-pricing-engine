import itertools

import pytest

from pricer import black_scholes as bs
from pricer.implied_vol import implied_vol

# Deep ITM/OTM short dated options are worth their intrinsic value (or zero) to
# machine precision, so their vol cannot be recovered. Keep only cases where
# vega is large enough for the price to carry information about sigma.
CASES = [
    c for c in itertools.product(["call", "put"], [50, 80, 100, 120, 200], [0.02, 0.5, 3.0], [0.05, 0.3, 1.2])
    if bs.vega(100, c[1], c[2], 0.02, c[3]) > 1e-3
]


@pytest.mark.parametrize("kind,K,T,sigma", CASES)
def test_round_trip(kind, K, T, sigma):
    px = float(bs.price(kind, 100, K, T, 0.02, sigma))
    assert implied_vol(px, kind, 100, K, T, 0.02) == pytest.approx(sigma, abs=1e-6)


def test_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        implied_vol(150.0, "call", 100, 100, 1.0, 0.02)
