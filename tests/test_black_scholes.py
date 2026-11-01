import numpy as np
import pytest

from pricer import black_scholes as bs

BASE = dict(S=100.0, K=95.0, T=0.75, r=0.03, sigma=0.25, q=0.01)


def test_known_value():
    # Hull, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs.price("call", 42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.7594, abs=1e-4)
    assert bs.price("put", 42, 40, 0.5, 0.1, 0.2) == pytest.approx(0.8086, abs=1e-4)


@pytest.mark.parametrize("K", [60, 90, 100, 110, 150])
def test_put_call_parity(K):
    p = {**BASE, "K": K}
    lhs = bs.price("call", **p) - bs.price("put", **p)
    rhs = p["S"] * np.exp(-p["q"] * p["T"]) - K * np.exp(-p["r"] * p["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


def _bump(f, key, h, **p):
    up, dn = {**p, key: p[key] + h}, {**p, key: p[key] - h}
    return (f(**up) - f(**dn)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    f = lambda **p: bs.price(kind, **p)
    assert bs.delta(kind, **BASE) == pytest.approx(_bump(f, "S", 1e-4, **BASE), rel=1e-6)
    assert bs.vega(**BASE) == pytest.approx(_bump(f, "sigma", 1e-5, **BASE), rel=1e-6)
    assert bs.rho(kind, **BASE) == pytest.approx(_bump(f, "r", 1e-6, **BASE), rel=1e-5)
    assert bs.theta(kind, **BASE) == pytest.approx(-_bump(f, "T", 1e-6, **BASE), rel=1e-5)
    d = lambda **p: bs.delta(kind, **p)
    assert bs.gamma(**BASE) == pytest.approx(_bump(d, "S", 1e-4, **BASE), rel=1e-6)


def test_vectorised_strikes():
    K = np.linspace(50, 150, 101)
    prices = bs.price("call", 100, K, 1.0, 0.02, 0.2)
    assert prices.shape == K.shape
    assert np.all(np.diff(prices) < 0)          # decreasing in strike
    assert np.all(np.diff(prices, 2) > 0)       # convex in strike (no butterfly arb)


def test_bad_kind():
    with pytest.raises(ValueError):
        bs.price("straddle", **BASE)
