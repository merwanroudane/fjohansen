"""
Frequency selection a la Perron, Shintani & Yabu (2016/2021).

The univariate Wald test of the null

    H_0:  gamma_1 = gamma_2 = ... = gamma_{2n} = 0

in the model

    y_t = sum beta_i t^i + sum_{j=1}^{n} [gamma_{1,j} sin(2 pi k_j t / T)
                                         + gamma_{2,j} cos(2 pi k_j t / T)] + u_t,
    u_t = alpha u_{t-1} + sum a*_i Delta u_{t-i} + e_t,

is implemented exactly as in PSY (2016, equations 12-17), with the
Prais-Winsten FGLS transformation and a super-efficient + Roy-Fuller
bias-corrected estimator of alpha. The limit distribution is chi^2(2 m)
under both I(0) and I(1) noise.

A general-to-specific algorithm selects ``n`` for a single univariate
series; :func:`select_frequencies` extends it to a multivariate
``(T, p)`` panel by taking the maximum frequency across the columns
(this matches the recipe used in Kurita & Shintani, 2025, Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .utils import fourier_basis

__all__ = [
    "psy_wald_test",
    "select_frequencies_univariate",
    "select_frequencies",
    "FrequencySelectionResult",
]


# ---------------------------------------------------------------------------
# Table 1 of PSY: tau_{.50} and tau_{.85} for the bias correction. Used
# inside the Roy-Fuller correction step.
# Rows: number of frequencies n, columns: (p_d, tau_50 / tau_85).
# ---------------------------------------------------------------------------
PSY_TAU_TABLE: dict[tuple[int, int], tuple[float, float]] = {
    # (n, p_d): (tau_50, tau_85)
    (1, 0): (-2.39, -3.26),
    (2, 0): (-2.99, -3.93),
    (3, 0): (-3.51, -4.49),
    (4, 0): (-3.98, -4.99),
    (5, 0): (-4.36, -5.44),
    (1, 1): (-3.09, -3.83),
    (2, 1): (-3.79, -4.50),
    (3, 1): (-4.40, -5.10),
    (4, 1): (-4.92, -5.64),
    (5, 1): (-5.41, -6.11),
}


# ---------------------------------------------------------------------------
# Bias correction (PSY eq. 11)
# ---------------------------------------------------------------------------
def _roy_fuller_correction(
    alpha_hat: float,
    sigma_alpha: float,
    T: int,
    p_d: int,
    n: int,
    p_T: int = 0,
    version: str = "upper-biased",
) -> float:
    r"""Return alpha_M = alpha_hat + C(tau_hat) sigma_alpha (PSY 2016, eq. 11)."""
    tau_hat = (alpha_hat - 1.0) / sigma_alpha if sigma_alpha > 0 else 0.0
    tau_50, tau_85 = PSY_TAU_TABLE.get((n, p_d), (-3.0, -4.0))
    tau_pct = tau_85 if version.startswith("upper") else tau_50

    a = 10.0
    r = p_d + 1 + 2 * n
    c1 = (1.0 + r) * T
    c2_num = (1.0 + r) * T - tau_pct ** 2 * (((p_T + 2) / 2) + T)
    c2_den = tau_pct * (a + tau_pct) * (((p_T + 2) / 2) + T)
    c2 = c2_num / c2_den if c2_den != 0 else 0.0
    if tau_hat > tau_pct:
        C = -tau_hat
    elif tau_hat > -a:
        denom = (p_T + 2) / 2.0 * (1.0 / T) * tau_hat - (1.0 + r) * (tau_hat + c2 * (tau_hat + a))
        C = (1.0 / denom) if denom != 0 else 0.0
    elif tau_hat > -np.sqrt(c1):
        denom = (p_T + 2) / 2.0 * (1.0 / T) * tau_hat - (1.0 + r) * tau_hat
        C = (1.0 / denom) if denom != 0 else 0.0
    else:
        C = 0.0
    return alpha_hat + C * sigma_alpha


def _select_lag_aic(u: np.ndarray, p_max: int) -> int:
    """Pick lag length for the residual AR by AIC (Ng-Perron MAIC-lite)."""
    T = u.size
    best, best_p = np.inf, 0
    for p in range(p_max + 1):
        if T - p - 1 < 5:
            break
        y = u[p + 1 :]
        X = u[p : -1].reshape(-1, 1)
        for j in range(1, p + 1):
            X = np.column_stack([X, np.diff(u)[p - j : -j]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sig2 = (resid ** 2).mean()
        if sig2 <= 0:
            continue
        aic = np.log(sig2) + 2 * (p + 1) / T
        if aic < best:
            best, best_p = aic, p
    return best_p


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
@dataclass
class PSYTestResult:
    n_freq: int
    stat: float
    df: int
    p_value: float
    alpha_hat: float
    alpha_corrected: float
    alpha_S: float
    test_subset: bool


def psy_wald_test(
    y: np.ndarray,
    k_freqs: Sequence[int],
    *,
    p_d: int = 1,
    subset_freq: Optional[int] = None,
    version: str = "upper-biased",
    p_T_max: Optional[int] = None,
) -> PSYTestResult:
    r"""
    PSY (2016) FGLS Wald test for the presence of Fourier-type nonlinear
    components in a univariate series.

    Parameters
    ----------
    y : array_like, shape (T,)
        Univariate time series.
    k_freqs : sequence of int
        Frequencies ``(k_1, ..., k_n)`` entering the regression.
    p_d : int, default 1
        Order of the polynomial trend (0: constant only, 1: constant + t).
    subset_freq : int or None
        If given, test the joint hypothesis that the sin/cos coefficients
        on the specific frequency value ``subset_freq`` are zero (used by
        the general-to-specific algorithm). Otherwise test all 2n.
    version : {'upper-biased', 'median-unbiased'}
        Which quantile to use in the Roy-Fuller correction.
    p_T_max : int or None
        Upper bound for the augmentation lag in the residual AR. Default
        ``floor(12 (T/100)^{1/4})``.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.size
    if T < 30:
        raise ValueError("Sample too small for PSY test.")
    n = len(k_freqs)
    if n == 0:
        raise ValueError("At least one frequency required.")

    # Build the regressor matrix x_t = (1, t, ..., t^{p_d}, sin/cos block)
    t = np.arange(1, T + 1)
    poly = np.column_stack([t ** i for i in range(p_d + 1)])
    sincos = []
    for k in k_freqs:
        ang = 2.0 * np.pi * k * t / T
        sincos.append(np.sin(ang))
        sincos.append(np.cos(ang))
    F = np.column_stack(sincos)
    Xreg = np.column_stack([poly, F])

    # Step 1: OLS, get residuals
    beta_ols, *_ = np.linalg.lstsq(Xreg, y, rcond=None)
    u_hat = y - Xreg @ beta_ols

    # Step 2: estimate alpha from u_hat with augmentation
    if p_T_max is None:
        p_T_max = int(np.floor(12 * (T / 100.0) ** 0.25))
    p_T = _select_lag_aic(u_hat, p_T_max)

    # Build regressors for u_t = alpha u_{t-1} + sum a*_i Du_{t-i} + e_t
    u_lag = u_hat[p_T : -1]
    y_dep = u_hat[p_T + 1 :]
    regs = [u_lag]
    if p_T > 0:
        du = np.diff(u_hat)
        for i in range(1, p_T + 1):
            regs.append(du[p_T - i : -i])
    Xar = np.column_stack(regs)
    coef, *_ = np.linalg.lstsq(Xar, y_dep, rcond=None)
    resid_ar = y_dep - Xar @ coef
    sig2 = (resid_ar ** 2).sum() / (y_dep.size - Xar.shape[1])
    cov_ar = sig2 * np.linalg.pinv(Xar.T @ Xar)
    alpha_hat = float(coef[0])
    sigma_alpha = float(np.sqrt(max(cov_ar[0, 0], 1e-16)))

    # Step 3: Roy-Fuller bias correction
    alpha_M = _roy_fuller_correction(
        alpha_hat, sigma_alpha, T, p_d, n, p_T=p_T, version=version
    )

    # Step 4: super-efficient estimator
    if abs(alpha_M - 1.0) > T ** (-0.5):
        alpha_S = alpha_M
    else:
        alpha_S = 1.0

    # Step 5: Prais-Winsten FGLS transformation
    y_tilde = y[1:] - alpha_S * y[:-1]
    X_tilde = Xreg[1:, :] - alpha_S * Xreg[:-1, :]
    # First observation:
    first_scale = np.sqrt(max(1.0 - alpha_S ** 2, 0.0))
    y0 = first_scale * y[0]
    X0 = first_scale * Xreg[0, :]
    y_pw = np.concatenate([[y0], y_tilde])
    X_pw = np.vstack([X0, X_tilde])

    # FGLS estimate with generalised inverse (column of constant may be 0
    # when alpha_S = 1 -- this is fine, see PSY remark p. 7).
    XtX = X_pw.T @ X_pw
    XtX_pinv = np.linalg.pinv(XtX)
    psi_hat = XtX_pinv @ (X_pw.T @ y_pw)
    v_hat = y_pw - X_pw @ psi_hat

    # Long-run variance: omega^2 = sig2 if alpha_S != 1, else Andrews (1991)
    if alpha_S != 1.0:
        omega2 = float((v_hat ** 2).sum() / max(T - p_T, 1))
    else:
        # Quadratic-spectral kernel
        m_T = max(int(np.floor(1.1447 * T ** (1 / 5))), 1)
        e0 = v_hat - v_hat.mean()
        gamma0 = float((e0 ** 2).mean())
        omega2 = gamma0
        for j in range(1, m_T):
            gj = float((e0[j:] * e0[:-j]).mean())
            x = 6.0 * np.pi * j / (5.0 * m_T)
            w = (25.0 / (12.0 * np.pi ** 2 * x ** 2)) * (
                np.sin(x) / x - np.cos(x)
            ) if x > 0 else 1.0
            omega2 += 2.0 * w * gj

    # Restriction matrix R Psi = 0
    k_poly = p_d + 1
    if subset_freq is None:
        # Test all 2n sin/cos coefficients jointly.
        R = np.zeros((2 * n, Xreg.shape[1]))
        for i in range(2 * n):
            R[i, k_poly + i] = 1.0
        df = 2 * n
        test_subset = False
    else:
        try:
            idx = list(k_freqs).index(int(subset_freq))
        except ValueError as exc:
            raise ValueError(
                f"subset_freq={subset_freq} not in k_freqs={k_freqs}"
            ) from exc
        R = np.zeros((2, Xreg.shape[1]))
        R[0, k_poly + 2 * idx] = 1.0
        R[1, k_poly + 2 * idx + 1] = 1.0
        df = 2
        test_subset = True

    mid = R @ XtX_pinv @ R.T
    rhs = R @ psi_hat
    # W = psi' R' [omega^2 R XtX_pinv R']^{-1} R psi
    try:
        sol = np.linalg.solve(omega2 * mid, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(omega2 * mid) @ rhs
    W = float(rhs @ sol)
    pval = float(chi2.sf(W, df))

    return PSYTestResult(
        n_freq=n,
        stat=W,
        df=df,
        p_value=pval,
        alpha_hat=alpha_hat,
        alpha_corrected=alpha_M,
        alpha_S=alpha_S,
        test_subset=test_subset,
    )


# ---------------------------------------------------------------------------
# General-to-specific frequency selection
# ---------------------------------------------------------------------------
@dataclass
class FrequencySelectionResult:
    n_selected: int
    per_series: Optional[pd.DataFrame]   # one row per series (multivariate)
    detail: pd.DataFrame                  # one row per step

    def __repr__(self) -> str:
        return (
            f"FrequencySelectionResult(n_selected={self.n_selected}, "
            f"steps={len(self.detail)})"
        )


def select_frequencies_univariate(
    y: np.ndarray,
    *,
    n_max: int = 5,
    p_d: int = 1,
    sig_level: float = 0.10,
    version: str = "upper-biased",
) -> FrequencySelectionResult:
    r"""
    Specific-to-general selection of ``n`` for a *single* time series.

    The algorithm starts at ``n_max`` and tests whether the sin/cos
    coefficients of the highest frequency are jointly zero. If the null is
    rejected the procedure stops at the current ``n``. Otherwise ``n``
    is decreased by 1 and the algorithm continues until ``n = 0``.
    """
    detail_rows = []
    n_selected = 0
    for n_try in range(n_max, 0, -1):
        k_list = list(range(1, n_try + 1))
        res = psy_wald_test(
            y, k_list, p_d=p_d, subset_freq=n_try, version=version
        )
        detail_rows.append(
            dict(
                step=n_max - n_try + 1,
                n=n_try,
                tested_freq=n_try,
                stat=res.stat,
                df=res.df,
                p_value=res.p_value,
                reject=res.p_value < sig_level,
            )
        )
        if res.p_value < sig_level:
            n_selected = n_try
            break
    return FrequencySelectionResult(
        n_selected=n_selected,
        per_series=None,
        detail=pd.DataFrame(detail_rows),
    )


def select_frequencies(
    data,
    *,
    n_max: int = 5,
    p_d: int = 1,
    sig_level: float = 0.10,
    version: str = "upper-biased",
) -> FrequencySelectionResult:
    r"""
    Frequency selection for a multivariate panel.

    Each column is tested independently and the final ``n`` is the maximum
    across columns. This mirrors the recipe used by Kurita & Shintani (2025)
    in their JGB-yield application.
    """
    from .utils import ensure_array
    X, names = ensure_array(data)
    rows = []
    final_n = 0
    detail = []
    for j, name in enumerate(names):
        res_j = select_frequencies_univariate(
            X[:, j], n_max=n_max, p_d=p_d, sig_level=sig_level, version=version
        )
        rows.append(dict(series=name, n_selected=res_j.n_selected))
        d = res_j.detail.copy()
        d["series"] = name
        detail.append(d)
        if res_j.n_selected > final_n:
            final_n = res_j.n_selected
    return FrequencySelectionResult(
        n_selected=final_n,
        per_series=pd.DataFrame(rows),
        detail=pd.concat(detail, ignore_index=True) if detail else pd.DataFrame(),
    )
