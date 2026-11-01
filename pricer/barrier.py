"""Closed form for a continuously monitored down-and-out call (no rebate).

Used mainly as ground truth for the Monte Carlo engine. Formulas follow
Reiner and Rubinstein (1991), as written in Hull, chapter on exotic options.

Also includes the Broadie, Glasserman and Kou (1997) continuity correction,
which maps a discretely monitored barrier to an equivalent continuous one.
"""
from __future__ import annotations

import math

from scipy.stats import norm

from . import black_scholes as bs

BGK_BETA = 0.5825971579390106  # -zeta(1/2) / sqrt(2 pi)


def down_and_out_call(S, K, T, r, sigma, B, q=0.0) -> float:
    if S <= B:
        return 0.0
    vol_sqrt_t = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    h = B / S

    if B <= K:
        y = math.log(B * B / (S * K)) / vol_sqrt_t + lam * vol_sqrt_t
        down_in = S * df_q * h ** (2 * lam) * norm.cdf(y) - K * df_r * h ** (2 * lam - 2) * norm.cdf(
            y - vol_sqrt_t
        )
        return float(bs.price("call", S, K, T, r, sigma, q)) - down_in

    x1 = math.log(S / B) / vol_sqrt_t + lam * vol_sqrt_t
    y1 = math.log(B / S) / vol_sqrt_t + lam * vol_sqrt_t
    return (
        S * df_q * norm.cdf(x1)
        - K * df_r * norm.cdf(x1 - vol_sqrt_t)
        - S * df_q * h ** (2 * lam) * norm.cdf(y1)
        + K * df_r * h ** (2 * lam - 2) * norm.cdf(y1 - vol_sqrt_t)
    )


def down_and_in_call(S, K, T, r, sigma, B, q=0.0) -> float:
    """In-out parity: down-and-in plus down-and-out equals the vanilla."""
    return float(bs.price("call", S, K, T, r, sigma, q)) - down_and_out_call(S, K, T, r, sigma, B, q)


def bgk_shifted_barrier(B, sigma, dt, direction="down") -> float:
    """Continuous barrier equivalent to monitoring every `dt` years."""
    shift = math.exp(BGK_BETA * sigma * math.sqrt(dt))
    return B / shift if direction == "down" else B * shift
