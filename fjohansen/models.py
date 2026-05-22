"""
Deterministic-component specifications for the Johansen-Fourier procedure.

Six model variants are supported:

==============  ====================================  ============================================================
Code            Z_1 (in cointegrating space)          Z_2 (unrestricted regressors apart from DX lags)
==============  ====================================  ============================================================
``CNR``         ``(X_{t-1}', F_{t,T}', 1)'``         none
``LNR``         ``(X_{t-1}', F_{t,T}', t)'``         constant
``CNU``         ``(X_{t-1}', 1)'``                    ``F_{t,T}``
``LNU``         ``(X_{t-1}', t)'``                    constant, ``F_{t,T}``
``constant``    ``(X_{t-1}', 1)'``                    none                          (standard Johansen)
``linear``      ``(X_{t-1}', t)'``                    constant                      (standard Johansen)
==============  ====================================  ============================================================

The function :func:`build_design_matrices` returns the three matrices
``(Z0, Z1, Z2)`` ready for reduced-rank regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .utils import fourier_basis, stack_lags

__all__ = [
    "ModelSpec",
    "MODELS",
    "build_design_matrices",
]


@dataclass(frozen=True)
class ModelSpec:
    code: str            # canonical code (upper-case)
    label: str           # human-readable label
    z1_constant: bool    # restricted constant in beta-space
    z1_trend: bool       # restricted linear trend in beta-space
    z1_fourier: bool     # restricted Fourier in beta-space
    z2_constant: bool    # unrestricted constant in error-correction
    z2_fourier: bool     # unrestricted Fourier outside beta-space

    @property
    def has_fourier(self) -> bool:
        return self.z1_fourier or self.z2_fourier


MODELS: dict[str, ModelSpec] = {
    "CNR": ModelSpec("CNR", "Constant + Nonlinear restricted",
                     z1_constant=True, z1_trend=False, z1_fourier=True,
                     z2_constant=False, z2_fourier=False),
    "LNR": ModelSpec("LNR", "Linear + Nonlinear restricted",
                     z1_constant=False, z1_trend=True, z1_fourier=True,
                     z2_constant=True, z2_fourier=False),
    "CNU": ModelSpec("CNU", "Constant restricted, Nonlinear unrestricted",
                     z1_constant=True, z1_trend=False, z1_fourier=False,
                     z2_constant=False, z2_fourier=True),
    "LNU": ModelSpec("LNU", "Linear restricted, Nonlinear unrestricted",
                     z1_constant=False, z1_trend=True, z1_fourier=False,
                     z2_constant=True, z2_fourier=True),
    "CONSTANT": ModelSpec("CONSTANT", "Standard Johansen, constant restricted",
                          z1_constant=True, z1_trend=False, z1_fourier=False,
                          z2_constant=False, z2_fourier=False),
    "LINEAR": ModelSpec("LINEAR", "Standard Johansen, linear-trend restricted",
                        z1_constant=False, z1_trend=True, z1_fourier=False,
                        z2_constant=True, z2_fourier=False),
}


def _resolve_model(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    key = model.upper().replace(" ", "_")
    if key not in MODELS:
        raise ValueError(
            f"Unknown model {model!r}; valid codes are {list(MODELS)}."
        )
    return MODELS[key]


def build_design_matrices(
    X: np.ndarray,
    *,
    k: int,
    n: int,
    model: str | ModelSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    r"""
    Build ``Z0``, ``Z1``, ``Z2`` for the requested specification.

    Parameters
    ----------
    X : ndarray, shape (T, p)
        Levels of the multivariate process.
    k : int
        Lag order of the VAR in levels. We use ``k - 1`` lagged differences.
    n : int
        Number of Fourier frequencies (only used for CNR, LNR, CNU, LNU).
    model : str or ModelSpec

    Returns
    -------
    Z0 : ndarray, shape (T_eff, p)
        Difference of the levels for the effective sample.
    Z1 : ndarray, shape (T_eff, q1)
        Restricted regressors (lagged level, restricted constant/trend,
        restricted Fourier). The reduced-rank regression estimates the
        cointegrating space inside ``span(Z1)``.
    Z2 : ndarray, shape (T_eff, q2)
        Unrestricted regressors (lagged differences, plus unrestricted
        constant / Fourier for the relevant models).
    info : dict
        ``{"T_eff": T_eff, "spec": ModelSpec, "z1_names": [...], "z2_names": [...]}``
    """
    spec = _resolve_model(model)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (T, p).")
    T, p = X.shape
    if k < 1:
        raise ValueError("k must be >= 1.")
    if T <= k + 1:
        raise ValueError("Sample size too small for the chosen lag order.")

    # Effective sample: t = k + 1, ..., T
    t_eff_start = k
    T_eff = T - k

    # ----- Z0 = Delta X_t -----
    dX = np.diff(X, axis=0)               # length T - 1
    Z0 = dX[t_eff_start - 1 :, :]         # last T_eff rows

    # ----- X_{t-1} (in Z1) -----
    X_lag1 = X[t_eff_start - 1 : -1, :]   # rows for t = k+1 .. T using t-1

    # ----- DX_{t-i} for i = 1..k-1 (will form Z2) -----
    Z2_dlags = stack_lags(X, k)

    # ----- Fourier basis (full sample, then truncated to effective sample) -----
    if n > 0:
        F_full = fourier_basis(T, n)
        F_eff = F_full[t_eff_start:, :]   # rows for t = k+1 .. T
    else:
        F_eff = np.empty((T_eff, 0))

    # ----- Build Z1 (restricted regressors) -----
    z1_blocks = [X_lag1]
    z1_names = [f"X{i + 1}_lag1" for i in range(p)]
    if spec.z1_fourier and n > 0:
        z1_blocks.append(F_eff)
        for j in range(1, n + 1):
            z1_names += [f"sin({j}.2pit/T)", f"cos({j}.2pit/T)"]
    if spec.z1_constant:
        z1_blocks.append(np.ones((T_eff, 1)))
        z1_names.append("const_in_beta")
    if spec.z1_trend:
        t_vec = np.arange(t_eff_start + 1, T + 1, dtype=float).reshape(-1, 1)
        z1_blocks.append(t_vec)
        z1_names.append("t_in_beta")
    Z1 = np.concatenate(z1_blocks, axis=1)

    # ----- Build Z2 (unrestricted regressors) -----
    z2_blocks = [Z2_dlags]
    z2_names = []
    for i in range(1, k):
        z2_names += [f"DX{j + 1}_lag{i}" for j in range(p)]
    if spec.z2_constant:
        z2_blocks.append(np.ones((T_eff, 1)))
        z2_names.append("const")
    if spec.z2_fourier and n > 0:
        z2_blocks.append(F_eff)
        for j in range(1, n + 1):
            z2_names += [f"sin({j}.2pit/T)", f"cos({j}.2pit/T)"]
    Z2 = (
        np.concatenate(z2_blocks, axis=1)
        if any(b.shape[1] > 0 for b in z2_blocks)
        else np.empty((T_eff, 0))
    )

    info = dict(
        T_eff=T_eff,
        spec=spec,
        z1_names=z1_names,
        z2_names=z2_names,
        p=p,
        n=n,
        k=k,
        t_index=np.arange(t_eff_start + 1, T + 1),
    )
    return Z0, Z1, Z2, info
