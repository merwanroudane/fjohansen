"""
Publication-quality plotting routines.

Style is calibrated to reproduce the look of Kurita & Shintani (2025):
serif fonts, muted palette, light grid, square-ish panels. Every function
returns a matplotlib ``Figure`` so users can save / customise downstream.

Functions
---------
set_paper_style
    Activate the paper-style rcParams (called automatically).
plot_series
    Multi-panel time-series plot (Fig. 11 of the paper).
plot_limit_density
    Density of the simulated limit distribution (Figs. 1-2).
plot_recursive_rejection
    Recursive rejection frequency curves (Figs. 3-10).
plot_eigenvalues
    Bar / scree plot of the Johansen eigenvalues.
plot_long_run
    Plot of the estimated cointegrating relations.
plot_risk_premium
    Implied risk-premium decomposition (Fig. 13).
plot_residual_diagnostics
    Scaled residuals, ACF, Q-Q plot (Fig. 12).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
except Exception as exc:                            # pragma: no cover
    raise RuntimeError(
        "matplotlib is required for fjohansen plots."
    ) from exc

from .simulation import simulate_limit_distribution

__all__ = [
    "set_paper_style",
    "plot_series",
    "plot_limit_density",
    "plot_recursive_rejection",
    "plot_eigenvalues",
    "plot_long_run",
    "plot_risk_premium",
    "plot_residual_diagnostics",
]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_PAPER_PALETTE = [
    "#1f4e79",  # dark navy
    "#a02828",  # crimson
    "#2e7d32",  # forest
    "#a65900",  # ochre
    "#5b378f",  # plum
    "#0d6e8a",  # teal
]


def set_paper_style() -> None:
    """Apply the rcParams used throughout the library."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.color": "#cccccc",
            "grid.linestyle": ":",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.1,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.prop_cycle": plt.cycler(color=_PAPER_PALETTE),
        }
    )


set_paper_style()


# ---------------------------------------------------------------------------
# Generic series plot (Fig. 11)
# ---------------------------------------------------------------------------
def plot_series(
    data,
    *,
    title: Optional[str] = None,
    ncols: int = 2,
    figsize: Optional[tuple] = None,
    index=None,
) -> Figure:
    """Multi-panel plot of a (T, p) data set (like Fig. 11)."""
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
        idx = data.index if index is None else index
        arr = data.to_numpy()
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        names = [f"x{i + 1}" for i in range(arr.shape[1])]
        idx = index if index is not None else np.arange(arr.shape[0])
    T, p = arr.shape
    nrows = int(np.ceil(p / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 1.8 * nrows + 0.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(p):
        ax = axes[i]
        ax.plot(idx, arr[:, i], color=_PAPER_PALETTE[i % len(_PAPER_PALETTE)])
        ax.set_title(names[i], loc="left")
        ax.tick_params(labelbottom=True)
    for j in range(p, len(axes)):
        axes[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Limit distribution density (Figs. 1, 2)
# ---------------------------------------------------------------------------
def plot_limit_density(
    p_minus_r: int,
    n_values: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    model: str = "CNR",
    n_sims: int = 8_000,
    grid_size: int = 300,
    figsize: tuple = (6, 4),
    title: Optional[str] = None,
) -> Figure:
    r"""
    Density curves of the limiting distribution for several values of ``n``
    (cf. Fig. 1 / Fig. 2 of the paper).
    """
    fig, ax = plt.subplots(figsize=figsize)
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    for k, n in enumerate(n_values):
        draws = simulate_limit_distribution(
            p_minus_r,
            n,
            model=model if n > 0 else ("CONSTANT" if model in ("CNR", "CNU") else "LINEAR"),
            n_sims=n_sims,
            grid_size=grid_size,
            seed=12345 + k,
        )
        hist, edges = np.histogram(draws, bins=120, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # smooth: simple boxcar
        kernel = np.ones(5) / 5
        hist_s = np.convolve(hist, kernel, mode="same")
        ax.plot(
            centres,
            hist_s,
            label=f"n = {n}" + (" (linear)" if n == 0 else ""),
            linewidth=1.2 if n > 0 else 1.6,
            color=_PAPER_PALETTE[k % len(_PAPER_PALETTE)],
            linestyle=styles[k % len(styles)],
        )
    ax.set_xlabel("trace statistic")
    ax.set_ylabel("density")
    ax.set_title(
        title or f"Limit distribution: {model}, p - r = {p_minus_r}", loc="left"
    )
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# Recursive rejection rates (Figs. 3 - 10)
# ---------------------------------------------------------------------------
def plot_recursive_rejection(
    rejection_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: Optional[str] = None,
    nominal: float = 0.05,
    figsize: tuple = (6, 4),
    ylabel: str = "%",
    xlabel: str = "T",
) -> Figure:
    r"""
    Generic plotter for recursive rejection-rate curves.

    Parameters
    ----------
    rejection_curves : dict
        ``{label: (T_grid, rates)}``, where ``rates`` is a 1-D array in [0, 1].
    """
    fig, ax = plt.subplots(figsize=figsize)
    for k, (label, (Ts, rates)) in enumerate(rejection_curves.items()):
        ax.plot(
            Ts,
            np.asarray(rates) * 100,
            label=label,
            color=_PAPER_PALETTE[k % len(_PAPER_PALETTE)],
            linestyle="-" if "CNR" in label or "CNU" in label or "LNR" in label else "--",
            linewidth=1.4,
        )
    ax.axhline(nominal * 100, color="grey", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# Result-bound plots
# ---------------------------------------------------------------------------
def plot_eigenvalues(results, figsize: tuple = (5.5, 3.5)) -> Figure:
    """Bar chart of the ordered eigenvalues from the RR regression."""
    fig, ax = plt.subplots(figsize=figsize)
    eig = results.eigenvalues
    xs = np.arange(1, len(eig) + 1)
    bars = ax.bar(xs, eig, color=_PAPER_PALETTE[0], edgecolor="black", linewidth=0.5)
    if results.selected_rank is not None:
        for j, b in enumerate(bars):
            if j < results.selected_rank:
                b.set_color(_PAPER_PALETTE[1])
    ax.set_xticks(xs)
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"eigenvalue $\hat\lambda_i$")
    ax.set_title(
        f"Johansen eigenvalues  ({results.spec.code}, n = {results.n})",
        loc="left",
    )
    return fig


def plot_long_run(results, figsize: Optional[tuple] = None) -> Figure:
    """Plot of each cointegrating relation beta_j' X_t + delta_j' F_t (+ const)."""
    if results.fitted_long_run is None:
        raise RuntimeError("No fitted long-run relations available; refit with "
                           "compute_estimates=True and a non-zero rank.")
    flr = results.fitted_long_run
    r = flr.shape[1]
    if figsize is None:
        figsize = (6.5, 1.8 * r + 1)
    fig, axes = plt.subplots(r, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    t = results.t_index
    for j in range(r):
        ax = axes[j]
        ax.plot(t, flr[:, j], color=_PAPER_PALETTE[j % len(_PAPER_PALETTE)])
        ax.axhline(np.mean(flr[:, j]), linewidth=0.5, color="grey")
        ax.set_ylabel(rf"$\beta_{{{j + 1}}}'\!X_t$")
    axes[-1].set_xlabel("t")
    fig.suptitle("Estimated cointegrating relations", y=0.995, fontsize=11)
    fig.tight_layout()
    return fig


def plot_risk_premium(
    results,
    *,
    index=None,
    figsize: Optional[tuple] = None,
    titles: Optional[Sequence[str]] = None,
) -> Figure:
    r"""
    Reproduce Figure 13 of Kurita & Shintani (2025): plot, for each
    cointegrating relation, the implied "risk premium" series

        rho_t = - beta' X_{t-1} - delta' F_{t,T} - gamma

    together with its nonlinear component ``delta' F_{t,T}``.
    """
    if results.fitted_long_run is None or results.delta is None:
        raise RuntimeError(
            "plot_risk_premium requires a fitted CNR/LNR model with "
            "non-zero rank and a non-trivial Fourier component."
        )
    fl = -results.fitted_long_run        # rho_t  has the - sign convention
    # nonlinear component delta' F_{t,T}
    n = results.n
    if n == 0:
        raise RuntimeError("No Fourier component to plot (n = 0).")
    # Re-build F on the effective sample to multiply by delta
    T_eff = results.T_eff
    # delta has shape (2n, r); F should have (T_eff, 2n)
    # We reconstruct F using t_index and the original T (rough):
    T_full = T_eff + results.k
    t = results.t_index
    cols = []
    for j in range(1, n + 1):
        ang = 2.0 * np.pi * j * t / T_full
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    F = np.column_stack(cols)
    nonlin = F @ results.delta
    r = fl.shape[1]
    if figsize is None:
        figsize = (6.5, 1.8 * r + 1)
    fig, axes = plt.subplots(r, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    idx = t if index is None else index
    for j in range(r):
        ax = axes[j]
        ax.plot(idx, fl[:, j], color="#a02828", linewidth=1.2, label=r"$\hat\rho_t$")
        ax.plot(
            idx,
            -nonlin[:, j],
            color="#2e7d32",
            linewidth=1.0,
            linestyle="--",
            label="non-linear component",
        )
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_title(
            titles[j] if titles and j < len(titles) else f"Relation {j + 1}",
            loc="left",
        )
        if j == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("t")
    fig.suptitle("Implied risk-premium decomposition", y=0.995, fontsize=11)
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    results,
    figsize: Optional[tuple] = None,
    max_lag: int = 14,
) -> Figure:
    """Reproduce Figure 12 of the paper: residuals, ACF, Q-Q plot per series."""
    if results.residuals is None:
        raise RuntimeError("No residuals available; fit the model first.")
    R = results.residuals
    sigma = np.sqrt(np.diag(results.Sigma)) if results.Sigma is not None else R.std(axis=0)
    Rstd = R / sigma
    p = R.shape[1]
    if figsize is None:
        figsize = (9, 1.6 * p + 0.5)
    fig, axes = plt.subplots(p, 3, figsize=figsize, sharex=False)
    axes = np.atleast_2d(axes)
    t = results.t_index
    for i in range(p):
        # Column 1: scaled residuals
        ax = axes[i, 0]
        ax.plot(t, Rstd[:, i], color=_PAPER_PALETTE[i % len(_PAPER_PALETTE)],
                linewidth=0.6)
        ax.axhline(0.0, color="grey", linewidth=0.5)
        ax.set_ylim(-3.2, 3.2)
        ax.set_title(
            results.series_names[i] if i < len(results.series_names) else f"y_{i+1}",
            loc="left",
        )
        # Column 2: ACF
        ax = axes[i, 1]
        x = Rstd[:, i] - Rstd[:, i].mean()
        denom = (x ** 2).sum()
        acf = np.array(
            [1.0]
            + [
                float((x[k:] * x[:-k]).sum() / denom)
                for k in range(1, max_lag + 1)
            ]
        )
        ax.bar(np.arange(max_lag + 1), acf, color=_PAPER_PALETTE[i % len(_PAPER_PALETTE)],
               width=0.6)
        ax.axhline(1.96 / np.sqrt(len(x)), color="grey", linewidth=0.4, linestyle="--")
        ax.axhline(-1.96 / np.sqrt(len(x)), color="grey", linewidth=0.4, linestyle="--")
        ax.set_ylim(-0.4, 1.05)
        # Column 3: Q-Q
        ax = axes[i, 2]
        sorted_x = np.sort(Rstd[:, i])
        probs = (np.arange(1, sorted_x.size + 1) - 0.5) / sorted_x.size
        from scipy.stats import norm
        ax.plot(norm.ppf(probs), sorted_x, ".", markersize=2,
                color=_PAPER_PALETTE[i % len(_PAPER_PALETTE)])
        lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
        ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    fig.suptitle("Residual diagnostics", y=0.995, fontsize=11)
    fig.tight_layout()
    return fig
