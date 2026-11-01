"""Black-Scholes-Merton prices and Greeks for European options.

Everything is vectorised: any argument can be a scalar or a numpy array and
the usual broadcasting rules apply. A continuous dividend yield `q` is
supported throughout.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(kind, S, K, T, r, sigma, q=0.0):
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def delta(kind, S, K, T, r, sigma, q=0.0):
    _check_kind(kind)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    df_q = np.exp(-q * T)
    return df_q * norm.cdf(d1) if kind == "call" else df_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity to a unit change in sigma (not 1 vol point)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(kind, S, K, T, r, sigma, q=0.0):
    """Derivative with respect to calendar time, per year (usually negative)."""
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    decay = -S * df_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        return decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    return decay + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)


def rho(kind, S, K, T, r, sigma, q=0.0):
    _check_kind(kind)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)
    if kind == "call":
        return K * T * df_r * norm.cdf(d2)
    return -K * T * df_r * norm.cdf(-d2)


def greeks(kind, S, K, T, r, sigma, q=0.0) -> dict:
    return {
        "price": price(kind, S, K, T, r, sigma, q),
        "delta": delta(kind, S, K, T, r, sigma, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(kind, S, K, T, r, sigma, q),
        "rho": rho(kind, S, K, T, r, sigma, q),
    }
