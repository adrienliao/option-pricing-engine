"""Command line entry point: python -m pricer --help"""
import argparse

from . import barrier, binomial, monte_carlo as mc
from . import black_scholes as bs
from .implied_vol import implied_vol


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pricer", description="Price vanilla and barrier options.")
    ap.add_argument("method", choices=["bs", "crr", "mc", "barrier", "iv"])
    ap.add_argument("--kind", choices=["call", "put"], default="call")
    ap.add_argument("--S", type=float, default=100.0)
    ap.add_argument("--K", type=float, default=100.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.03)
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--B", type=float, default=90.0, help="barrier level")
    ap.add_argument("--price", type=float, help="market price, for iv")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--american", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    args = (a.kind, a.S, a.K, a.T, a.r, a.sigma, a.q)

    if a.method == "bs":
        for name, val in bs.greeks(*args).items():
            print(f"{name:>6}: {float(val): .6f}")
    elif a.method == "crr":
        print(f"{binomial.crr_price(*args, steps=a.steps, american=a.american):.6f}")
    elif a.method == "mc":
        res = mc.european(*args, n_paths=a.paths, seed=a.seed)
        lo, hi = res.ci()
        print(f"{res.price:.6f}  (s.e. {res.std_err:.6f}, 95% CI [{lo:.4f}, {hi:.4f}])")
    elif a.method == "barrier":
        exact = barrier.down_and_out_call(a.S, a.K, a.T, a.r, a.sigma, a.B, a.q)
        res = mc.down_and_out("call", a.S, a.K, a.T, a.r, a.sigma, a.B, a.q,
                              n_paths=a.paths, n_steps=252, continuous=True, seed=a.seed)
        print(f"closed form      {exact:.6f}")
        print(f"MC + bridge      {res.price:.6f}  (s.e. {res.std_err:.6f})")
    elif a.method == "iv":
        if a.price is None:
            ap.error("iv needs --price")
        print(f"{implied_vol(a.price, a.kind, a.S, a.K, a.T, a.r, a.q):.6f}")


if __name__ == "__main__":
    main()
