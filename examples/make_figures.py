"""Regenerate the figures in docs/. Run from the repo root: python examples/make_figures.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pricer import barrier, monte_carlo as mc  # noqa: E402
from pricer import black_scholes as bs  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})
OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)
P = dict(S=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)


def convergence():
    ns = np.unique(np.logspace(3, 6, 13).astype(int) // 2 * 2)
    setups = [("plain", dict(antithetic=False, control_variate=False), GREY),
              ("antithetic", dict(antithetic=True, control_variate=False), BLUE),
              ("antithetic + control variate", dict(antithetic=True, control_variate=True), ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, kw, c in setups:
        se = [mc.european("call", **P, n_paths=n, seed=0, **kw).std_err for n in ns]
        ax.loglog(ns, se, "o-", color=c, lw=2, ms=4, label=label)
    ax.set_xlabel("number of paths")
    ax.set_ylabel("standard error of the price")
    ax.set_title("Monte Carlo error on an ATM call, all slopes follow 1/sqrt(N)", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mc_convergence.png", dpi=160)


def barrier_monitoring():
    B = 90.0
    exact = barrier.down_and_out_call(**P, B=B)
    steps = np.array([4, 8, 12, 26, 52, 104, 252, 504])
    est, err = [], []
    for m in steps:
        r = mc.down_and_out("call", **P, B=B, n_paths=200_000, n_steps=int(m), seed=1)
        est.append(r.price); err.append(r.std_err)
    est, err = np.array(est), np.array(err)
    bgk = [barrier.down_and_out_call(**P, B=barrier.bgk_shifted_barrier(B, P["sigma"], P["T"] / m)) for m in steps]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(exact, color=GREY, lw=1.2, ls="--", label="continuous barrier, closed form")
    ax.fill_between(steps, est - 1.96 * err, est + 1.96 * err, color=BLUE, alpha=0.18, lw=0)
    ax.plot(steps, est, "o-", color=BLUE, lw=2, ms=5, label="Monte Carlo, discrete monitoring (95% CI)")
    ax.plot(steps, bgk, "s", color=ORANGE, ms=6, label="Broadie-Glasserman-Kou correction")
    ax.set_xscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("monitoring dates per year")
    ax.set_ylabel("down-and-out call price")
    ax.set_title(f"Discrete vs continuous barrier (S=K=100, B={B:.0f})", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "barrier_monitoring.png", dpi=160)


def greeks_profile():
    S = np.linspace(60, 140, 400)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for T, c in [(0.1, ORANGE), (0.5, BLUE), (1.0, AQUA)]:
        kw = dict(K=100, T=T, r=0.03, sigma=0.2)
        axes[0].plot(S, bs.delta("call", S, **kw), color=c, lw=2, label=f"T = {T}")
        axes[1].plot(S, bs.gamma(S, **kw), color=c, lw=2)
        axes[2].plot(S, bs.vega(S, **kw), color=c, lw=2)
    for ax, name in zip(axes, ["delta", "gamma", "vega"]):
        ax.set_title(name, loc="left")
        ax.set_xlabel("spot")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "greeks.png", dpi=160)


if __name__ == "__main__":
    convergence()
    barrier_monitoring()
    greeks_profile()
    print("figures written to", OUT)
