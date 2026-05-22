"""
Reproduces a slimmed-down version of the Monte-Carlo experiment of Section 5
of Kurita & Shintani (2025).

We generate data from F-DGP-1 (4-d, r = 1, n = 1) and compare the rank
selection of the proposed CNR(n = 1) model with the standard
constant-restricted Johansen model.
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

import fjohansen as fj


def rank_selection_rate(
    *, T_grid, n_reps=400, dgp_kw=None, n_fourier=1, model="CNR", seed=0
):
    rates = []
    for T in T_grid:
        chosen = []
        for rep in range(n_reps):
            X = fj.generate_f_dgp1(T=T, seed=seed * 1_000 + rep)
            try:
                r = fj.JohansenFourier(X, k=2, n=n_fourier, model=model).fit(
                    compute_estimates=False
                ).selected_rank
            except Exception:
                r = None
            chosen.append(r if r is not None else 0)
        chosen = np.asarray(chosen)
        rates.append((chosen == 1).mean())
    return np.asarray(rates)


def main() -> None:
    T_grid = [100, 150, 200, 300]
    n_reps = 200

    t0 = time.time()
    fj_rate = rank_selection_rate(T_grid=T_grid, n_reps=n_reps, n_fourier=1, model="CNR")
    std_rate = rank_selection_rate(T_grid=T_grid, n_reps=n_reps, n_fourier=0, model="constant")
    print(f"Done in {time.time() - t0:.1f}s")

    print(f"\nF-DGP-1: P(select r = 1)  --  {n_reps} replications per T")
    print(f"{'T':>6}  {'CNR(n=1)':>10}  {'Standard':>10}")
    for T, a, b in zip(T_grid, fj_rate, std_rate):
        print(f"{T:>6d}  {a * 100:>9.1f}%  {b * 100:>9.1f}%")

    fig = fj.plot_recursive_rejection(
        {
            "CNR (n=1)": (np.array(T_grid), fj_rate),
            "Standard linear": (np.array(T_grid), std_rate),
        },
        title="F-DGP-1: P(r̂ = 1)",
        nominal=0.95,
        xlabel="T",
        ylabel="%",
    )
    plt.show()


if __name__ == "__main__":
    main()
