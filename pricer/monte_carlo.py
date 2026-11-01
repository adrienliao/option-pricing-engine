"""Monte Carlo pricing under GBM with variance reduction.

Every pricer returns an `MCResult` with the estimate and its standard error,
because a Monte Carlo number without an error bar is not worth much.

Variance reduction available:
  * antithetic variates (pair each normal draw with its negative)
  * control variate on the discounted terminal spot, whose mean is known
For barriers, `continuous=True` uses a Brownian bridge to account for the
probability of crossing the barrier between two monitoring dates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    std_err: float
    n_paths: int

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_err, self.price + z * self.std_err


def _normals(rng, n_paths, n_steps, antithetic):
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.vstack([half, -half])
    return rng.standard_normal((n_paths, n_steps))


def _payoff(kind, s, K):
    return np.maximum(s - K, 0.0) if kind == "call" else np.maximum(K - s, 0.0)


def _summarise(samples, n_pairs_antithetic=False) -> MCResult:
    # with antithetic draws, samples i and i + n/2 are dependent, so average the
    # pairs first and compute the error on the (independent) pair means
    if n_pairs_antithetic:
        half = len(samples) // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))), len(samples))


def european(kind, S, K, T, r, sigma, q=0.0, n_paths=100_000, antithetic=True,
             control_variate=True, seed=None) -> MCResult:
    rng = np.random.default_rng(seed)
    z = _normals(rng, n_paths, 1, antithetic)[:, 0]
    s_t = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc = np.exp(-r * T)
    y = disc * _payoff(kind, s_t, K)

    if control_variate:
        x = disc * s_t
        x_mean = S * np.exp(-q * T)
        cov = np.cov(y, x)
        beta = cov[0, 1] / cov[1, 1]
        y = y - beta * (x - x_mean)
    return _summarise(y, antithetic)


def pathwise_delta(kind, S, K, T, r, sigma, q=0.0, n_paths=100_000, seed=None) -> MCResult:
    """Delta from the pathwise derivative of the payoff, no bumping needed."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    s_t = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    itm = (s_t > K) if kind == "call" else (s_t < K)
    sign = 1.0 if kind == "call" else -1.0
    return _summarise(np.exp(-r * T) * sign * itm * s_t / S)


def gbm_paths(S, T, r, sigma, q=0.0, n_paths=10_000, n_steps=252, antithetic=True, seed=None):
    """Log-Euler is exact for GBM, so there is no discretisation bias here."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = _normals(rng, n_paths, n_steps, antithetic)
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.log(S) + np.cumsum(log_inc, axis=1)
    return np.hstack([np.full((len(z), 1), np.log(S)), log_paths])


def down_and_out(kind, S, K, T, r, sigma, B, q=0.0, n_paths=100_000, n_steps=252,
                 continuous=False, antithetic=True, seed=None) -> MCResult:
    """Down-and-out option, barrier monitored at each of the `n_steps` dates.

    With `continuous=True`, the survival probability between two dates comes
    from the Brownian bridge: P(no hit | x0, x1) = 1 - exp(-2 (x0-b)(x1-b) / (sigma^2 dt)).
    Weighting each path by that probability removes the discrete monitoring bias.
    """
    log_s = gbm_paths(S, T, r, sigma, q, n_paths, n_steps, antithetic, seed)
    log_b = np.log(B)
    above = log_s > log_b
    alive = above.all(axis=1)
    weight = alive.astype(float)

    if continuous:
        dt = T / n_steps
        x0, x1 = log_s[:, :-1] - log_b, log_s[:, 1:] - log_b
        hit = np.exp(-2.0 * np.clip(x0, 0, None) * np.clip(x1, 0, None) / (sigma**2 * dt))
        weight *= np.prod(1.0 - hit, axis=1)

    s_t = np.exp(log_s[:, -1])
    samples = np.exp(-r * T) * _payoff(kind, s_t, K) * weight
    return _summarise(samples, antithetic)
