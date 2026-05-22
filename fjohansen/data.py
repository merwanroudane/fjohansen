"""
Data-generating processes from Section 5 of Kurita & Shintani (2025), and a
small synthetic dataset that mimics the Japanese term-structure data of
Section 6.

The DGPs reproduced here are:

* :func:`generate_nf_dgp1`, ..., :func:`generate_nf_dgp4` -- non Fourier-type
  DGPs of Sub-section 5.1 (Section 5.1, eq. 19).
* :func:`generate_f_dgp1`, :func:`generate_f_dgp2` -- Fourier-type DGPs of
  Section 5.2 (4-dimensional systems used to validate the proposed test).
* :func:`sample_jgb_data` -- a deterministic synthetic surrogate of the JGB
  data shown in Figure 11. Useful for documentation and unit tests when the
  real dataset is unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import fourier_basis

__all__ = [
    "generate_nf_dgp1",
    "generate_nf_dgp2",
    "generate_nf_dgp3",
    "generate_nf_dgp4",
    "generate_f_dgp1",
    "generate_f_dgp2",
    "sample_jgb_data",
]


# ---------------------------------------------------------------------------
# Bivariate non-Fourier DGPs
# ---------------------------------------------------------------------------
def _bivariate_skeleton(
    T: int,
    H: Optional[np.ndarray] = None,
    *,
    seed: Optional[int] = None,
    no_cointegration: bool = False,
    s: float = 0.01,
    start: float = np.log(100.0),
) -> np.ndarray:
    r"""Build the bivariate process used by NF-DGP-1..4 (eq. 19 of the paper)."""
    alpha = np.array([[-0.2], [0.1]])
    beta = np.array([[1.0], [-0.5]])
    gamma = -2.3
    rng = np.random.default_rng(seed)
    cov = s * np.array([[1.0, 0.25], [0.25, 1.0]])
    eps = rng.multivariate_normal(np.zeros(2), cov, size=T)
    X = np.empty((T, 2))
    X[0] = start
    if no_cointegration:
        alpha = np.zeros_like(alpha)
    for t in range(1, T):
        h = 0.0 if H is None else float(H[t])
        ec = (beta.T @ X[t - 1]).item() + h + gamma
        X[t] = X[t - 1] + (alpha * ec).ravel() + eps[t]
    return X


def generate_nf_dgp1(T: int = 400, seed: Optional[int] = 0, **kw) -> pd.DataFrame:
    """NF-DGP-1: no cointegration (alpha = 0)."""
    X = _bivariate_skeleton(T, None, seed=seed, no_cointegration=True, **kw)
    return pd.DataFrame(X, columns=["x1", "x2"])


def generate_nf_dgp2(T: int = 400, seed: Optional[int] = 0, **kw) -> pd.DataFrame:
    r"""NF-DGP-2: level shift  H_t = -0.05 * I(t >= 0.4 T)."""
    H = -0.05 * (np.arange(T) >= 0.4 * T).astype(float)
    X = _bivariate_skeleton(T, H, seed=seed, **kw)
    return pd.DataFrame(X, columns=["x1", "x2"])


def generate_nf_dgp3(T: int = 400, seed: Optional[int] = 0, **kw) -> pd.DataFrame:
    r"""NF-DGP-3: exponential smooth transition centred at 0.4 T."""
    u = np.arange(T) / T
    H = -0.1 * (1.0 - np.exp(-10.0 * (u - 0.4) ** 2))
    X = _bivariate_skeleton(T, H, seed=seed, **kw)
    return pd.DataFrame(X, columns=["x1", "x2"])


def generate_nf_dgp4(T: int = 400, seed: Optional[int] = 0, **kw) -> pd.DataFrame:
    r"""NF-DGP-4: sharper exponential smooth transition centred at 0.8 T."""
    u = np.arange(T) / T
    H = -0.2 * (1.0 - np.exp(-30.0 * (u - 0.8) ** 2))
    X = _bivariate_skeleton(T, H, seed=seed, **kw)
    return pd.DataFrame(X, columns=["x1", "x2"])


# ---------------------------------------------------------------------------
# Four-variate Fourier DGPs (Section 5.2)
# ---------------------------------------------------------------------------
def _quad_skeleton(
    T: int,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
    n: int,
    *,
    seed: Optional[int] = None,
    s: float = 0.01,
    start: float = np.log(100.0),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov = s * (np.eye(4) + 0.25 * (np.ones((4, 4)) - np.eye(4)))
    eps = rng.multivariate_normal(np.zeros(4), cov, size=T)
    F = fourier_basis(T, n)
    X = np.full((T, 4), start)
    for t in range(1, T):
        # delta has shape (2n, r); beta has shape (4, r); gamma has shape (r,)
        ec = beta.T @ X[t - 1] + delta.T @ F[t] + gamma
        X[t] = X[t - 1] + alpha @ ec + eps[t]
    return X


def generate_f_dgp1(T: int = 400, seed: Optional[int] = 0) -> pd.DataFrame:
    r"""F-DGP-1: 4-dim system, r = 1, n = 1 (paper eqs. p. 1604)."""
    alpha = np.array([[-0.2], [0.1], [0.0], [0.0]])
    beta = np.array([[1.0], [-1.0], [-1.0], [0.5]])
    delta = np.array([[0.1], [-0.1]])      # 2n x r
    gamma = np.array([2.3])
    X = _quad_skeleton(T, alpha, beta, delta, gamma, n=1, seed=seed)
    return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(4)])


def generate_f_dgp2(T: int = 400, seed: Optional[int] = 0) -> pd.DataFrame:
    r"""F-DGP-2: 4-dim system, r = 2, n = 2 (paper eqs. p. 1604)."""
    alpha = np.array([[-0.4, 0.0], [0.0, -0.3], [-0.2, -0.1], [0.0, 0.0]])
    beta = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )
    delta = np.array(
        [
            [0.08, 0.00],
            [0.06, -0.08],
            [-0.03, 0.04],
            [-0.06, 0.04],
        ]
    )
    gamma = np.array([-4.6, -9.2])
    X = _quad_skeleton(T, alpha, beta, delta, gamma, n=2, seed=seed)
    return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(4)])


# ---------------------------------------------------------------------------
# Synthetic JGB-like dataset
# ---------------------------------------------------------------------------
def sample_jgb_data(T: int = 108, seed: Optional[int] = 7) -> pd.DataFrame:
    r"""
    Synthetic JGB-yield surrogate matching the style of Figure 11 of
    Kurita & Shintani (2025).

    A single common stochastic trend drives six yield series whose long-run
    relations contain a slow Fourier-type movement (intended to imitate the
    time-varying term premium). Returns a (T x 6) DataFrame indexed by
    monthly periods from 1986-12 to 1995-11 (default T = 108 reproduces the
    paper's sample length).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # Stochastic trend (1 common factor, integrated noise + slow cycle)
    common = np.cumsum(rng.standard_normal(T) * 0.07) + 5.0
    common += 1.2 * np.cos(2.0 * np.pi * t / T)
    # Term-premium nonlinearity
    risk_premium = 0.6 * np.sin(2.0 * np.pi * t / T) + 0.3 * np.cos(4.0 * np.pi * t / T)
    # Maturities
    maturities = [20.0, 10.0, 5.0, 3.0, 1.0, 0.083]      # call ~ 1 month
    cols = []
    names = ["i_20yr", "i_10yr", "i_5yr", "i_3yr", "i_1yr", "i_call"]
    for m, name in zip(maturities, names):
        slope = 0.25 * np.log(1.0 + m)
        series = common - slope * np.exp(-0.5 * m) + 0.7 * (1.0 - np.exp(-m / 3.0)) * risk_premium
        series += rng.standard_normal(T) * 0.10
        cols.append(series)
    idx = pd.period_range("1986-12", periods=T, freq="M").to_timestamp()
    return pd.DataFrame(np.column_stack(cols), index=idx, columns=names)
