"""Implied volatility by Newton-Raphson, falling back to Brent when Newton struggles.

Newton converges in a handful of steps near the money, but vega collapses for
deep ITM/OTM options and short maturities, which makes the raw Newton step
blow up. In that case we switch to Brent on a bracket, which always works as
long as the price is inside the no-arbitrage bounds.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

from . import black_scholes as bs

SIGMA_LO, SIGMA_HI = 1e-6, 5.0


def no_arbitrage_bounds(kind, S, K, T, r, q=0.0):
    fwd_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call":
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def implied_vol(target, kind, S, K, T, r, q=0.0, tol=1e-10, max_iter=50) -> float:
    lo, hi = no_arbitrage_bounds(kind, S, K, T, r, q)
    if not lo <= target <= hi:
        raise ValueError(f"price {target:.6g} outside no-arbitrage bounds [{lo:.6g}, {hi:.6g}]")

    # Brenner-Subrahmanyam guess, decent near the money
    sigma = max(math.sqrt(2 * math.pi / T) * target / S, 0.05)
    for _ in range(max_iter):
        diff = float(bs.price(kind, S, K, T, r, sigma, q)) - target
        if abs(diff) < tol:
            return sigma
        v = float(bs.vega(S, K, T, r, sigma, q))
        if v < 1e-8:
            break
        step = diff / v
        sigma -= step
        if not SIGMA_LO < sigma < SIGMA_HI:
            break

    f = lambda s: float(bs.price(kind, S, K, T, r, s, q)) - target
    return brentq(f, SIGMA_LO, SIGMA_HI, xtol=tol, maxiter=500)
