"""
Critical values and p-values for the Johansen-Fourier trace statistic.

Two routes are supported:

1. **Pre-tabulated quantiles** — Table B1 of Kurita & Shintani (2025) for the
   CNR model. For (p - r) in {2,...,8} and n in {1,...,5}.

2. **Gamma-approximation p-values** — following Doornik (1998) and
   Johansen-Mosconi-Nielsen (2000), the limit distribution is approximated by a
   Gamma distribution matched to its first two moments. Moments are taken from
   Table B1 (for CNR), or simulated on the fly for other models / cells.

The on-the-fly simulator is in ``fjohansen.simulation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import gamma

from .simulation import simulate_limit_moments

__all__ = [
    "CNR_TABLE_B1",
    "quantile",
    "p_value",
    "moments",
]

# ---------------------------------------------------------------------------
# Table B1 of Kurita & Shintani (2025): approximate limit quantiles for CNR.
# Rows: (p - r, n). Cols: 90%, 95%, 97.5%, 99%, mean, var.
# ---------------------------------------------------------------------------
CNR_TABLE_B1: dict[tuple[int, int], dict[str, float]] = {
    # p - r = 8
    (8, 1): {"q90": 215.32, "q95": 222.28, "q975": 228.39, "q99": 235.77, "mean": 192.90, "var": 295.22},
    (8, 2): {"q90": 268.05, "q95": 275.96, "q975": 282.82, "q99": 290.72, "mean": 242.91, "var": 375.90},
    (8, 3): {"q90": 321.21, "q95": 329.61, "q975": 337.26, "q99": 345.96, "mean": 293.23, "var": 460.70},
    (8, 4): {"q90": 374.39, "q95": 383.76, "q975": 391.68, "q99": 401.24, "mean": 344.06, "var": 546.96},
    (8, 5): {"q90": 427.82, "q95": 437.64, "q975": 446.32, "q99": 456.55, "mean": 395.07, "var": 637.83},
    # p - r = 7
    (7, 1): {"q90": 174.61, "q95": 180.95, "q975": 186.85, "q99": 193.44, "mean": 154.54, "var": 239.04},
    (7, 2): {"q90": 220.83, "q95": 227.88, "q975": 234.03, "q99": 241.92, "mean": 197.95, "var": 310.73},
    (7, 3): {"q90": 267.49, "q95": 275.45, "q975": 282.33, "q99": 290.44, "mean": 241.83, "var": 387.45},
    (7, 4): {"q90": 313.83, "q95": 322.43, "q975": 329.86, "q99": 338.80, "mean": 285.85, "var": 463.97},
    (7, 5): {"q90": 360.16, "q95": 369.35, "q975": 377.52, "q99": 387.06, "mean": 330.09, "var": 540.59},
    # p - r = 6
    (6, 1): {"q90": 137.94, "q95": 143.58, "q975": 148.89, "q99": 154.96, "mean": 120.12, "var": 187.99},
    (6, 2): {"q90": 177.63, "q95": 183.99, "q975": 189.77, "q99": 196.59, "mean": 157.05, "var": 249.99},
    (6, 3): {"q90": 217.37, "q95": 224.55, "q975": 230.76, "q99": 238.17, "mean": 194.29, "var": 314.19},
    (6, 4): {"q90": 257.08, "q95": 264.82, "q975": 271.64, "q99": 279.67, "mean": 231.62, "var": 380.54},
    (6, 5): {"q90": 296.54, "q95": 304.89, "q975": 312.21, "q99": 320.98, "mean": 269.15, "var": 445.16},
    # p - r = 5
    (5, 1): {"q90": 105.52, "q95": 110.64, "q975": 115.08, "q99": 120.46, "mean": 89.84, "var": 143.00},
    (5, 2): {"q90": 138.56, "q95": 144.37, "q975": 149.51, "q99": 155.31, "mean": 120.20, "var": 196.05},
    (5, 3): {"q90": 171.38, "q95": 177.87, "q975": 183.62, "q99": 190.44, "mean": 150.75, "var": 250.61},
    (5, 4): {"q90": 204.31, "q95": 211.49, "q975": 217.89, "q99": 225.56, "mean": 181.45, "var": 308.67},
    (5, 5): {"q90": 237.12, "q95": 244.96, "q975": 251.88, "q99": 259.95, "mean": 212.28, "var": 364.95},
    # p - r = 4
    (4, 1): {"q90": 76.78, "q95": 81.19, "q975": 85.07, "q99": 89.90, "mean": 63.50, "var": 102.77},
    (4, 2): {"q90": 103.09, "q95": 108.22, "q975": 112.83, "q99": 118.40, "mean": 87.33, "var": 145.31},
    (4, 3): {"q90": 129.38, "q95": 135.03, "q975": 140.10, "q99": 145.75, "mean": 111.28, "var": 189.19},
    (4, 4): {"q90": 155.48, "q95": 161.64, "q975": 167.20, "q99": 174.07, "mean": 135.44, "var": 234.74},
    (4, 5): {"q90": 181.56, "q95": 188.25, "q975": 194.36, "q99": 201.30, "mean": 159.55, "var": 282.14},
    # p - r = 3
    (3, 1): {"q90": 52.28, "q95": 56.06, "q975": 59.38, "q99": 63.39, "mean": 41.25, "var": 69.30},
    (3, 2): {"q90": 71.88, "q95": 76.30, "q975": 80.22, "q99": 84.92, "mean": 58.64, "var": 101.30},
    (3, 3): {"q90": 91.49, "q95": 96.46, "q975": 100.83, "q99": 106.24, "mean": 76.15, "var": 136.47},
    (3, 4): {"q90": 111.01, "q95": 116.55, "q975": 121.57, "q99": 127.57, "mean": 93.72, "var": 173.72},
    (3, 5): {"q90": 130.39, "q95": 136.70, "q975": 141.97, "q99": 148.50, "mean": 111.27, "var": 211.64},
    # p - r = 2
    (2, 1): {"q90": 31.63, "q95": 34.52, "q975": 37.34, "q99": 40.84, "mean": 23.10, "var": 40.81},
    (2, 2): {"q90": 44.99, "q95": 48.68, "q975": 52.02, "q99": 55.89, "mean": 34.23, "var": 65.36},
    (2, 3): {"q90": 58.11, "q95": 62.44, "q975": 66.39, "q99": 71.07, "mean": 45.40, "var": 92.53},
    (2, 4): {"q90": 71.25, "q95": 76.14, "q975": 80.42, "q99": 85.54, "mean": 56.61, "var": 122.21},
    (2, 5): {"q90": 84.36, "q95": 89.75, "q975": 94.61, "q99": 100.53, "mean": 67.88, "var": 155.42},
}

_QUANTILE_KEYS = {"90%": "q90", "95%": "q95", "97.5%": "q975", "99%": "q99"}


# ---------------------------------------------------------------------------
# Pre-computed moment tables for the non-tabulated models.
# Obtained from simulate_limit_moments(..., n_sims=4000, grid_size=250).
# Keys: (p - r, n). Values: (mean, var). Bundled with the package so the
# very first call for a common cell does not pay the Monte Carlo cost.
# ---------------------------------------------------------------------------
_MOMENT_TABLES: dict[str, dict[tuple[int, int], tuple[float, float]]] = {
    # CNR cells *not* tabulated in the paper (p - r = 1).
    "CNR": {
        (1, 1): (9.018, 17.738), (1, 2): (13.989, 30.841), (1, 3): (18.767, 47.062),
        (1, 4): (23.534, 64.455), (1, 5): (28.293, 85.705),
    },
    "LNR": {
        (1, 1): (12.759, 20.885), (1, 2): (19.246, 31.131), (1, 3): (25.369, 41.657),
        (1, 4): (31.643, 52.566), (1, 5): (37.751, 61.124),
        (2, 1): (29.132, 47.343), (2, 2): (41.975, 67.293), (2, 3): (54.368, 87.798),
        (2, 4): (66.737, 110.222), (2, 5): (78.718, 128.808),
        (3, 1): (49.234, 74.901), (3, 2): (68.303, 107.307), (3, 3): (86.718, 132.605),
        (3, 4): (104.984, 171.875), (3, 5): (122.869, 204.908),
        (4, 1): (72.415, 106.330), (4, 2): (97.671, 148.270), (4, 3): (122.186, 184.340),
        (4, 4): (146.418, 225.682), (4, 5): (170.038, 266.693),
        (5, 1): (99.696, 144.734), (5, 2): (130.636, 192.742), (5, 3): (161.040, 239.781),
        (5, 4): (191.039, 286.182), (5, 5): (220.339, 345.925),
        (6, 1): (129.970, 186.055), (6, 2): (166.898, 250.546), (6, 3): (203.018, 301.259),
        (6, 4): (238.702, 361.059), (6, 5): (273.619, 417.155),
        (7, 1): (163.985, 233.647), (7, 2): (206.531, 297.048), (7, 3): (248.295, 356.012),
        (7, 4): (289.674, 429.830), (7, 5): (329.872, 500.363),
        (8, 1): (201.577, 287.490), (8, 2): (249.824, 369.212), (8, 3): (297.340, 431.904),
        (8, 4): (344.031, 497.661), (8, 5): (389.570, 571.151),
    },
    "CNU": {
        (1, 1): (5.976, 16.404), (1, 2): (8.918, 28.852), (1, 3): (11.759, 43.901),
        (1, 4): (14.502, 59.500), (1, 5): (17.203, 78.512),
        (2, 1): (16.429, 34.293), (2, 2): (22.965, 53.318), (2, 3): (29.221, 71.611),
        (2, 4): (35.316, 90.728), (2, 5): (41.154, 112.818),
        (3, 1): (31.047, 54.353), (3, 2): (41.444, 75.879), (3, 3): (51.374, 94.591),
        (3, 4): (61.051, 116.468), (3, 5): (70.419, 139.674),
        (4, 1): (49.368, 79.644), (4, 2): (63.549, 103.639), (4, 3): (77.364, 125.055),
        (4, 4): (90.710, 150.140), (4, 5): (103.637, 173.944),
        (5, 1): (71.671, 109.003), (5, 2): (89.494, 131.190), (5, 3): (107.094, 157.061),
        (5, 4): (124.229, 185.927), (5, 5): (140.732, 216.954),
        (6, 1): (97.279, 146.312), (6, 2): (119.103, 176.658), (6, 3): (140.273, 204.284),
        (6, 4): (161.018, 238.207), (6, 5): (181.094, 269.520),
        (7, 1): (126.633, 190.775), (7, 2): (151.894, 219.741), (7, 3): (176.744, 248.722),
        (7, 4): (201.185, 291.250), (7, 5): (224.655, 322.529),
        (8, 1): (159.656, 231.902), (8, 2): (188.600, 268.948), (8, 3): (217.078, 295.999),
        (8, 4): (244.944, 336.116), (8, 5): (271.666, 372.556),
    },
    "LNU": {
        (1, 1): (10.725, 16.993), (1, 2): (15.184, 23.552), (1, 3): (19.369, 30.091),
        (1, 4): (23.619, 36.592), (1, 5): (27.670, 40.864),
        (2, 1): (25.075, 38.820), (2, 2): (33.869, 52.187), (2, 3): (42.320, 64.580),
        (2, 4): (50.683, 76.876), (2, 5): (58.659, 87.783),
        (3, 1): (43.158, 64.995), (3, 2): (56.151, 84.880), (3, 3): (68.666, 98.770),
        (3, 4): (80.977, 118.587), (3, 5): (92.855, 137.487),
        (4, 1): (64.381, 92.837), (4, 2): (81.504, 117.611), (4, 3): (98.114, 139.649),
        (4, 4): (114.407, 160.877), (4, 5): (130.021, 181.439),
        (5, 1): (89.621, 127.433), (5, 2): (110.399, 153.922), (5, 3): (130.861, 183.947),
        (5, 4): (150.906, 209.260), (5, 5): (170.161, 241.900),
        (6, 1): (117.889, 165.984), (6, 2): (142.676, 201.927), (6, 3): (166.825, 230.820),
        (6, 4): (190.575, 261.134), (6, 5): (213.441, 290.578),
        (7, 1): (149.899, 213.668), (7, 2): (178.231, 247.867), (7, 3): (205.973, 279.326),
        (7, 4): (233.445, 318.674), (7, 5): (259.626, 355.795),
        (8, 1): (185.492, 263.339), (8, 2): (217.556, 305.168), (8, 3): (249.017, 334.324),
        (8, 4): (279.851, 372.413), (8, 5): (309.332, 410.936),
    },
    "CONSTANT": {
        (1, 0): (3.966, 6.598), (2, 0): (11.986, 20.067), (3, 0): (23.800, 38.162),
        (4, 0): (39.458, 60.840), (5, 0): (58.891, 88.364), (6, 0): (81.868, 120.177),
        (7, 0): (108.737, 162.479), (8, 0): (139.345, 201.169),
    },
    "LINEAR": {
        (1, 0): (6.240, 10.674), (2, 0): (16.271, 26.236), (3, 0): (30.188, 46.763),
        (4, 0): (47.583, 71.064), (5, 0): (68.915, 98.920), (6, 0): (93.454, 135.365),
        (7, 0): (122.094, 177.711), (8, 0): (154.234, 218.781),
    },
}


@dataclass(frozen=True)
class _GammaParams:
    shape: float
    scale: float


def _gamma_from_moments(mean: float, var: float) -> _GammaParams:
    """Match a two-parameter Gamma distribution to a given (mean, var)."""
    scale = var / mean
    shape = mean / scale
    return _GammaParams(shape=shape, scale=scale)


def moments(
    p_minus_r: int,
    n: int,
    model: str = "CNR",
    *,
    n_sims: int = 50_000,
    grid_size: int = 400,
    seed: Optional[int] = 12345,
    cache: dict | None = None,
) -> tuple[float, float]:
    """
    Return ``(mean, variance)`` of the limiting Johansen-Fourier trace
    distribution for the given combination.

    For ``model="CNR"`` and tabulated cells, uses Table B1 of Kurita &
    Shintani (2025). Otherwise simulates the limiting distribution.
    """
    model = model.upper()
    if model == "CNR" and (p_minus_r, n) in CNR_TABLE_B1:
        row = CNR_TABLE_B1[(p_minus_r, n)]
        return row["mean"], row["var"]
    if model in _MOMENT_TABLES and (p_minus_r, n) in _MOMENT_TABLES[model]:
        return _MOMENT_TABLES[model][(p_minus_r, n)]
    m, v, _ = simulate_limit_moments(
        p_minus_r,
        n,
        model=model,
        n_sims=n_sims,
        grid_size=grid_size,
        seed=seed,
        cache=cache,
    )
    return m, v


def quantile(
    level: float | str,
    p_minus_r: int,
    n: int,
    model: str = "CNR",
    **kwargs,
) -> float:
    r"""
    Approximate ``level`` quantile (e.g. 0.95 or '95%') of the limiting
    distribution using the Gamma approximation.

    For tabulated CNR cells and the standard 90 / 95 / 97.5 / 99 percentiles,
    the paper's hard-coded value is returned for exact reproducibility.
    """
    if isinstance(level, str):
        key = level if level.endswith("%") else f"{float(level) * 100:g}%"
    else:
        key = f"{level * 100:g}%"
    model = model.upper()
    if model == "CNR" and (p_minus_r, n) in CNR_TABLE_B1 and key in _QUANTILE_KEYS:
        return CNR_TABLE_B1[(p_minus_r, n)][_QUANTILE_KEYS[key]]
    prob = float(key.rstrip("%")) / 100.0
    mean, var = moments(p_minus_r, n, model=model, **kwargs)
    g = _gamma_from_moments(mean, var)
    return float(gamma.ppf(prob, a=g.shape, scale=g.scale))


def p_value(stat: float, p_minus_r: int, n: int, model: str = "CNR", **kwargs) -> float:
    r"""
    Approximate Gamma-based p-value of an observed trace statistic.

    Returns ``P(X >= stat)`` where ``X`` follows the Gamma distribution
    matched to the moments of the limiting distribution.
    """
    if p_minus_r <= 0:
        return float("nan")
    mean, var = moments(p_minus_r, n, model=model, **kwargs)
    g = _gamma_from_moments(mean, var)
    return float(gamma.sf(stat, a=g.shape, scale=g.scale))
