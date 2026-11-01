"""Cox-Ross-Rubinstein binomial tree, European or American exercise.

The backward induction is vectorised over the nodes of each time slice, so a
2000 step tree runs in a few milliseconds.
"""
from __future__ import annotations

import numpy as np


def crr_price(kind, S, K, T, r, sigma, q=0.0, steps=500, american=False) -> float:
    if kind not in ("call", "put"):
        raise ValueError(kind)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk neutral probability outside (0, 1), use more steps")
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    spot = S * u ** j * d ** (steps - j)
    payoff = (lambda s: np.maximum(s - K, 0.0)) if kind == "call" else (lambda s: np.maximum(K - s, 0.0))
    values = payoff(spot)

    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            spot = S * u ** j[: n + 1] * d ** (n - j[: n + 1])
            values = np.maximum(values, payoff(spot))
    return float(values[0])


def american_premium(kind, S, K, T, r, sigma, q=0.0, steps=500) -> float:
    """Value of the early exercise right: American minus European on the same tree."""
    return crr_price(kind, S, K, T, r, sigma, q, steps, american=True) - crr_price(
        kind, S, K, T, r, sigma, q, steps, american=False
    )
