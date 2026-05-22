"""
Simulation of the limiting Johansen-Fourier trace distribution.

The limiting distribution from Proposition 3.1 of Kurita & Shintani (2025) is

    tr( int dB G' * (int G G' du)^-1 * int G dB' )

where B is a (p - r)-dimensional standard Brownian motion and

    G_u = (B_u', F_u', 1)'        for the CNR model,
    G_u = (B_u', F_u', u | 1)'    for the LNR model,
    G_u = (B_u' | F_u')'          for the CNU model,
    G_u = (B_u', u | F_u')'       for the LNU model,
    G_u = (B_u', 1)'              for the standard constant restricted model,
    G_u = (B_u', u | 1)'          for the standard linear-trend restricted model.

``F_u = [sin(2 pi u), cos(2 pi u), ..., sin(2 pi n u), cos(2 pi n u)]'``.

Implementation
--------------
The simulation is **fully vectorised** across replications: all ``n_sims``
draws are evaluated in a single set of ``numpy.einsum``/``solve`` calls. A
persistent on-disk cache (under ``~/.fjohansen/cache``) ensures that the
same ``(p - r, n, model, n_sims, grid_size, seed)`` cell is only simulated
once on a given machine.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import pickle
from typing import Optional

import numpy as np

__all__ = [
    "simulate_limit_distribution",
    "simulate_limit_moments",
    "clear_cache",
]


_VALID_MODELS = {"CNR", "LNR", "CNU", "LNU", "CONSTANT", "LINEAR"}


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_dir() -> pathlib.Path:
    base = os.environ.get("FJOHANSEN_CACHE", str(pathlib.Path.home() / ".fjohansen" / "cache"))
    p = pathlib.Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(p_r: int, n: int, model: str, n_sims: int, grid_size: int, seed) -> str:
    raw = f"{p_r}|{n}|{model.upper()}|{n_sims}|{grid_size}|{seed}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def clear_cache() -> int:
    """Delete the on-disk simulation cache. Returns the number of files removed."""
    n = 0
    for f in _cache_dir().glob("*.pkl"):
        f.unlink()
        n += 1
    return n


# ---------------------------------------------------------------------------
# Deterministic-block construction
# ---------------------------------------------------------------------------
def _det_blocks(n: int, model: str, T: int):
    r"""
    Return ``(restricted, unrestricted)`` blocks of shape ``(*, T)``.

    ``restricted`` columns stay inside G_u; ``unrestricted`` columns are
    partialled-out from B and from any restricted deterministic block.
    """
    model = model.upper()
    if model not in _VALID_MODELS:
        raise ValueError(f"Unknown model {model!r}. Use one of {_VALID_MODELS}.")

    u_grid = np.arange(1, T + 1, dtype=float) / T
    rows_F = []
    for j in range(1, n + 1):
        rows_F.append(np.sin(2.0 * np.pi * j * u_grid))
        rows_F.append(np.cos(2.0 * np.pi * j * u_grid))
    F = np.vstack(rows_F) if rows_F else np.zeros((0, T))
    one = np.ones((1, T))
    t_lin = u_grid.reshape(1, T)

    if model == "CNR":
        restricted = np.vstack([F, one]) if F.size else one
        unrestricted = np.zeros((0, T))
    elif model == "LNR":
        restricted = np.vstack([F, t_lin]) if F.size else t_lin
        unrestricted = one
    elif model == "CNU":
        restricted = np.zeros((0, T))
        unrestricted = np.vstack([F, one]) if F.size else one
    elif model == "LNU":
        restricted = t_lin
        unrestricted = np.vstack([F, one]) if F.size else one
    elif model == "CONSTANT":
        restricted = one
        unrestricted = np.zeros((0, T))
    elif model == "LINEAR":
        restricted = t_lin
        unrestricted = one
    else:
        raise AssertionError("unreachable")
    return restricted, unrestricted


# ---------------------------------------------------------------------------
# Vectorised core
# ---------------------------------------------------------------------------
def _simulate_batch(p_r: int, n: int, model: str, T: int, n_sims: int,
                    seed: Optional[int]) -> np.ndarray:
    r"""
    Vectorised simulation of all ``n_sims`` replications at once.

    Returns
    -------
    draws : ndarray, shape (n_sims,)
    """
    rng = np.random.default_rng(seed)
    # Brownian motion: shape (n_sims, p_r, T)
    eps = rng.standard_normal(size=(n_sims, p_r, T)) / np.sqrt(T)
    B = np.cumsum(eps, axis=2)
    dB = eps  # increments == dB (last column is the final increment)

    restricted, unrestricted = _det_blocks(n, model, T)

    # Partial-out unrestricted from B and restricted block.
    if unrestricted.size:
        U = unrestricted.T                              # (T, k_un)
        UU_inv = np.linalg.pinv(U.T @ U)                # (k_un, k_un)
        proj = U @ UU_inv @ U.T                          # (T, T)
        # B - B @ proj  (broadcast across sims)
        B = B - B @ proj
        if restricted.size:
            restricted = restricted - restricted @ proj

    # G_sim has shape (n_sims, q, T) where q = p_r + restricted.shape[0]
    q_det = restricted.shape[0]
    q = p_r + q_det
    # Build by concatenation along axis 1
    if q_det:
        G = np.empty((n_sims, q, T))
        G[:, :p_r, :] = B
        G[:, p_r:, :] = restricted  # broadcast
    else:
        G = B

    # Use batched BLAS matmul -- much faster than einsum for these shapes.
    Gt = np.transpose(G, (0, 2, 1))           # (n_sims, T, q)
    dBt = np.transpose(dB, (0, 2, 1))         # (n_sims, T, p_r)
    int_GG = (G @ Gt) / T                     # (n_sims, q, q)
    int_dB_G = dB @ Gt                        # (n_sims, p_r, q)
    int_G_dB = G @ dBt                        # (n_sims, q, p_r)

    # Solve int_GG @ M = int_G_dB  for each sim
    # Use linalg.solve broadcasted
    try:
        M = np.linalg.solve(int_GG, int_G_dB)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(int_GG) @ int_G_dB

    # trace stat for each sim: tr(int_dB_G @ M)
    # = sum_{i,j} int_dB_G[s,i,j] * M[s,j,i]
    draws = np.einsum("sij,sji->s", int_dB_G, M)
    return draws.astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate_limit_distribution(
    p_minus_r: int,
    n: int,
    *,
    model: str = "CNR",
    n_sims: int = 5_000,
    grid_size: int = 300,
    seed: Optional[int] = 12345,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Monte-Carlo draws from the limiting trace distribution of Proposition 3.1.

    Parameters
    ----------
    p_minus_r : int
        Number of common stochastic trends (>= 1).
    n : int
        Number of Fourier frequencies (>= 0).
    model : {'CNR','LNR','CNU','LNU','CONSTANT','LINEAR'}
    n_sims : int, default 5000
        Number of Monte-Carlo replications.
    grid_size : int, default 300
        Discretisation of the unit interval.
    seed : int or None
        RNG seed (also part of the cache key).
    use_cache : bool, default True
        If True, read from / write to ``~/.fjohansen/cache``.

    Returns
    -------
    draws : ndarray, shape (n_sims,)
    """
    key = _cache_key(p_minus_r, n, model, n_sims, grid_size, seed)
    path = _cache_dir() / f"draws_{key}.pkl" if use_cache else None
    if use_cache and path is not None and path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    draws = _simulate_batch(p_minus_r, n, model, grid_size, n_sims, seed)
    if use_cache and path is not None:
        try:
            with path.open("wb") as fh:
                pickle.dump(draws, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
    return draws


def simulate_limit_moments(
    p_minus_r: int,
    n: int,
    *,
    model: str = "CNR",
    n_sims: int = 5_000,
    grid_size: int = 300,
    seed: Optional[int] = 12345,
    cache: Optional[dict] = None,
    use_cache: bool = True,
):
    r"""
    Return ``(mean, var, draws)`` of the limiting distribution.

    Uses an in-memory cache (the optional ``cache`` dict, or the module
    default) **and** an on-disk cache (in ``~/.fjohansen/cache``).
    """
    key = (model.upper(), int(p_minus_r), int(n), int(n_sims), int(grid_size), seed)
    if cache is None:
        cache = _MOMENT_CACHE
    if key in cache:
        return cache[key]
    draws = simulate_limit_distribution(
        p_minus_r, n, model=model, n_sims=n_sims,
        grid_size=grid_size, seed=seed, use_cache=use_cache,
    )
    out = (float(draws.mean()), float(draws.var(ddof=1)), draws)
    cache[key] = out
    return out


_MOMENT_CACHE: dict = {}
