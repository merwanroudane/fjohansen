"""
Reproduces a slimmed-down version of Figure 1 of Kurita & Shintani (2025):
limit density of the Johansen-Fourier trace statistic for the CNR model and
several values of n.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import fjohansen as fj


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, p_r in zip(axes.ravel(), [4, 3, 2, 1]):
        for k, n in enumerate([0, 1, 2, 3]):
            model = "CNR" if n > 0 else "CONSTANT"
            draws = fj.simulate_limit_distribution(
                p_r, n if n > 0 else 0, model=model, n_sims=5000, grid_size=300,
                seed=10 + k,
            )
            ax.hist(
                draws,
                bins=80,
                density=True,
                histtype="step",
                label=f"n = {n}" + (" (linear)" if n == 0 else ""),
                linewidth=1.4 if n == 0 else 1.0,
            )
        ax.set_title(f"p - r = {p_r}", loc="left")
        ax.legend()
    fig.suptitle("Simulated limit distributions (CNR vs. standard)", y=0.99)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
