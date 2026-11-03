import pytest

from pricer import barrier
from pricer import black_scholes as bs
from pricer import monte_carlo as mc

P = dict(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_within_error_bars(kind):
    res = mc.european(kind, **P, n_paths=200_000, seed=1)
    assert abs(res.price - float(bs.price(kind, **P))) < 4 * res.std_err


def test_control_variate_reduces_error():
    plain = mc.european("call", **P, n_paths=100_000, antithetic=False, control_variate=False, seed=2)
    reduced = mc.european("call", **P, n_paths=100_000, antithetic=True, control_variate=True, seed=2)
    # about a 3x variance reduction at the money, more in the wings
    assert reduced.std_err < 0.7 * plain.std_err


def test_pathwise_delta():
    res = mc.pathwise_delta("call", **P, n_paths=200_000, seed=3)
    assert abs(res.price - float(bs.delta("call", **P))) < 4 * res.std_err


@pytest.mark.parametrize("B", [80.0, 95.0])
def test_barrier_bridge_matches_closed_form(B):
    exact = barrier.down_and_out_call(**P, B=B)
    res = mc.down_and_out("call", **P, B=B, n_paths=100_000, n_steps=50, continuous=True, seed=4)
    assert abs(res.price - exact) < 4 * res.std_err


def test_discrete_barrier_matches_bgk_correction():
    B, n_steps = 90.0, 12
    shifted = barrier.bgk_shifted_barrier(B, P["sigma"], P["T"] / n_steps)
    approx = barrier.down_and_out_call(**P, B=shifted)
    res = mc.down_and_out("call", **P, B=B, n_paths=200_000, n_steps=n_steps, seed=5)
    assert abs(res.price - approx) < 4 * res.std_err + 0.02


def test_in_out_parity_and_limits():
    vanilla = float(bs.price("call", **P))
    assert barrier.down_and_out_call(**P, B=1e-6) == pytest.approx(vanilla, rel=1e-9)
    assert barrier.down_and_out_call(**P, B=120.0) == 0.0
    assert barrier.down_and_in_call(**P, B=90.0) + barrier.down_and_out_call(**P, B=90.0) == pytest.approx(vanilla)


def test_barrier_case_above_strike():
    # B > K branch: still monotone, below vanilla, above zero
    v = barrier.down_and_out_call(S=100, K=80, T=1, r=0.03, sigma=0.2, B=90)
    assert 0 < v < float(bs.price("call", 100, 80, 1, 0.03, 0.2))
    res = mc.down_and_out("call", S=100, K=80, T=1, r=0.03, sigma=0.2, B=90,
                          n_paths=100_000, n_steps=50, continuous=True, seed=6)
    assert abs(res.price - v) < 4 * res.std_err
