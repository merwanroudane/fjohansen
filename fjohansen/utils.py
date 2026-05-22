"""
Helper functions for fjohansen.

Mostly: Fourier basis construction, OLS residualisation, small linear-algebra
utilities used throughout the package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "fourier_basis",
    "ensure_array",
    "ols_residuals",
    "log_det_pd",
    "stack_lags",
]


def fourier_basis(T: int, n: int, t_start: int = 1) -> np.ndarray:
    r"""
    Build the Fourier deterministic basis used by Kurita & Shintani (2025):

        F_{t,T} = [sin(2 pi t / T), cos(2 pi t / T),
                   sin(2 pi 2 t / T), cos(2 pi 2 t / T),
                   ...,
                   sin(2 pi n t / T), cos(2 pi n t / T)]'

    Parameters
    ----------
    T : int
        Sample size used in the denominator (paper uses the full sample T).
    n : int
        Number of frequencies in the expansion (n >= 0).
    t_start : int, default 1
        Starting value of t (paper indexes t = 1, ..., T).

    Returns
    -------
    F : ndarray of shape (T, 2 n)
        Empty (T, 0) array if n == 0.
    """
    if n < 0:
        raise ValueError("n must be >= 0.")
    t = np.arange(t_start, t_start + T, dtype=float)
    if n == 0:
        return np.empty((T, 0))
    cols = []
    for j in range(1, n + 1):
        ang = 2.0 * np.pi * j * t / T
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.column_stack(cols)


def ensure_array(data) -> tuple[np.ndarray, list[str]]:
    """Coerce *data* to a 2-D float ndarray and return ``(arr, names)``."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float), list(data.columns)
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=float).reshape(-1, 1), [data.name or "y"]
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, [f"x{i + 1}" for i in range(arr.shape[1])]


def ols_residuals(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    r"""
    Residuals of OLS of each column of Y on X.

    If X has zero columns, returns Y unchanged.
    """
    if X.shape[1] == 0:
        return Y
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def log_det_pd(M: np.ndarray) -> float:
    """Log determinant of a positive-definite matrix via Cholesky."""
    L = np.linalg.cholesky(M)
    return 2.0 * np.sum(np.log(np.diag(L)))


def stack_lags(X: np.ndarray, k: int) -> np.ndarray:
    r"""
    Build the lag matrix [DX_{t-1}, DX_{t-2}, ..., DX_{t-(k-1)}]
    from level series X (T x p). Returns array of shape
    (T - k, p * (k - 1)). Used to construct Z_2 in Johansen's procedure.
    """
    if k < 1:
        raise ValueError("k must be >= 1 (VAR order in levels).")
    dX = np.diff(X, axis=0)  # length T - 1
    T_eff = X.shape[0] - k   # observations after losing k starting values
    if k == 1:
        return np.empty((T_eff, 0))
    cols = []
    for i in range(1, k):
        # Aligned so that row j corresponds to t = k + 1 + j, i.e. DX_{t-i}.
        cols.append(dX[k - 1 - i : k - 1 - i + T_eff, :])
    return np.concatenate(cols, axis=1)
