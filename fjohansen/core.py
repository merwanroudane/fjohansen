"""
``JohansenFourier`` -- the main public class.

Performs the reduced-rank regression of Kurita & Shintani (2025) for a
cointegrated VAR(k) with restricted Fourier-type smooth nonlinear trends in
the cointegrating space.

Usage
-----
>>> from fjohansen import JohansenFourier
>>> res = JohansenFourier(data, k=3, n=5, model='CNR').fit()
>>> print(res.summary())
>>> res.plot_eigenvalues()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .critical_values import p_value, quantile
from .models import MODELS, ModelSpec, build_design_matrices
from .utils import ensure_array, ols_residuals

__all__ = ["JohansenFourier", "JohansenFourierResults"]


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
@dataclass
class JohansenFourierResults:
    """Estimated quantities returned by :meth:`JohansenFourier.fit`."""

    spec: ModelSpec
    p: int
    k: int
    n: int
    T_eff: int

    eigenvalues: np.ndarray          # length p, sorted descending
    eigenvectors: np.ndarray         # raw eigenvectors of S_11^{-1/2} ...

    trace_stats: pd.DataFrame        # one row per r in 0..p-1
    selected_rank: Optional[int] = None

    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None    # X_{t-1} block
    delta: Optional[np.ndarray] = None   # Fourier block (or None)
    gamma_const: Optional[np.ndarray] = None
    gamma_trend: Optional[np.ndarray] = None
    Gamma: Optional[list[np.ndarray]] = None
    Phi_unrestricted: Optional[np.ndarray] = None  # coeffs on Z2
    Sigma: Optional[np.ndarray] = None             # residual covariance

    residuals: Optional[np.ndarray] = None         # T_eff x p
    fitted_long_run: Optional[np.ndarray] = None   # beta' X + delta' F + gamma
    series_names: Sequence[str] = field(default_factory=list)
    t_index: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pretty summary
    # ------------------------------------------------------------------
    def summary(self, sig_level: float = 0.05) -> str:
        """Return a formatted text summary akin to Table 3 of the paper."""
        from .tables import format_trace_table, format_header
        head = format_header(self)
        body = format_trace_table(self.trace_stats, sig_level=sig_level)
        if self.selected_rank is not None:
            tail = (
                f"\nSelected cointegrating rank (sequential 5%-test): "
                f"r = {self.selected_rank}\n"
                f"=> Number of common stochastic trends p - r "
                f"= {self.p - self.selected_rank}.\n"
            )
        else:
            tail = ""
        return head + "\n" + body + tail

    def to_latex(self, **kw) -> str:
        from .tables import format_trace_latex
        return format_trace_latex(self.trace_stats, spec=self.spec, n=self.n, **kw)

    def to_html(self, **kw) -> str:
        from .tables import format_trace_html
        return format_trace_html(self.trace_stats, spec=self.spec, n=self.n, **kw)

    # ------------------------------------------------------------------
    # Convenience plots (forwarders)
    # ------------------------------------------------------------------
    def plot_eigenvalues(self, **kw):
        from .plotting import plot_eigenvalues
        return plot_eigenvalues(self, **kw)

    def plot_long_run(self, **kw):
        from .plotting import plot_long_run
        return plot_long_run(self, **kw)

    def plot_residual_diagnostics(self, **kw):
        from .plotting import plot_residual_diagnostics
        return plot_residual_diagnostics(self, **kw)

    def plot_risk_premium(self, **kw):
        from .plotting import plot_risk_premium
        return plot_risk_premium(self, **kw)


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------
class JohansenFourier:
    """
    Johansen cointegrating-rank test with Fourier-type smooth nonlinear
    deterministic trends restricted to the cointegrating space.

    Parameters
    ----------
    data : array_like or DataFrame, shape (T, p)
        Multivariate I(1) time series in levels.
    k : int
        VAR order in levels (``k - 1`` lagged differences will be used).
    n : int, default 1
        Number of Fourier frequencies. Use ``n = 0`` to recover the standard
        Johansen models (set ``model='constant'`` or ``'linear'`` instead).
    model : {'CNR', 'LNR', 'CNU', 'LNU', 'constant', 'linear'}, default 'CNR'
        Deterministic specification (see :mod:`fjohansen.models`).
    """

    def __init__(
        self,
        data,
        k: int,
        n: int = 1,
        model: str = "CNR",
    ):
        self.X, self.series_names = ensure_array(data)
        self.k = int(k)
        self.n = int(n)
        self.spec = MODELS[model.upper().replace(" ", "_")]
        if not self.spec.has_fourier and self.n > 0:
            # silently zero-out n for standard models so the table look-up
            # uses the right column.
            self.n = 0

    # ------------------------------------------------------------------
    def fit(
        self,
        sig_level: float = 0.05,
        select_rank: bool = True,
        compute_pvalues: bool = True,
        compute_estimates: bool = True,
        n_sims: int = 5_000,
    ) -> JohansenFourierResults:
        """
        Run the reduced-rank regression and the trace test.

        Parameters
        ----------
        sig_level : float, default 0.05
            Significance level used for the sequential rank selection.
        select_rank : bool, default True
            If True, pick rank via Johansen's sequential procedure
            (r = 0, 1, ..., p - 1).
        compute_pvalues : bool, default True
            If True, compute p-values via the Gamma approximation.
        compute_estimates : bool, default True
            If True, return alpha, beta, delta and the residuals at the
            selected rank.
        n_sims : int, default 30_000
            Number of replications used to simulate the limit distribution
            for cells outside Table B1.
        """
        Z0, Z1, Z2, info = build_design_matrices(
            self.X, k=self.k, n=self.n, model=self.spec
        )
        p = info["p"]
        T_eff = info["T_eff"]

        # --- partial-out Z2 from Z0 and Z1 ------------------------------
        R0 = ols_residuals(Z0, Z2)
        R1 = ols_residuals(Z1, Z2)

        # --- product moments -------------------------------------------
        S00 = R0.T @ R0 / T_eff
        S11 = R1.T @ R1 / T_eff
        S01 = R0.T @ R1 / T_eff
        S10 = S01.T

        # --- generalized eigenvalue problem ----------------------------
        # det(lambda * S11 - S10 S00^{-1} S01) = 0
        # Equivalent to standard eig of S11^{-1/2} S10 S00^{-1} S01 S11^{-1/2}.
        try:
            L = np.linalg.cholesky(S11)
            L_inv = np.linalg.solve(L, np.eye(S11.shape[0]))
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "S11 is not positive-definite; the model is probably "
                "mis-specified (e.g. multi-collinear deterministics)."
            ) from exc
        S00_inv = np.linalg.pinv(S00)
        M = L_inv @ S10 @ S00_inv @ S01 @ L_inv.T  # symmetric in theory
        M = (M + M.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)        # ascending
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, 1.0)
        eigvecs = eigvecs[:, order]
        # Transform back to original metric: v_i = L^{-T} u_i, normalised by
        # v' S11 v = I.
        V = L_inv.T @ eigvecs

        # --- trace statistics ------------------------------------------
        rows = []
        for r in range(p):
            tail = eigvals[r:]
            stat = float(-T_eff * np.sum(np.log1p(-tail)))
            p_r = p - r
            cv95 = quantile("95%", p_r, self.n, model=self.spec.code)
            cv99 = quantile("99%", p_r, self.n, model=self.spec.code)
            pval = (
                p_value(stat, p_r, self.n, model=self.spec.code, n_sims=n_sims)
                if compute_pvalues
                else np.nan
            )
            rows.append(
                dict(
                    null=f"r <= {r}",
                    eigenvalue=eigvals[r],
                    trace_stat=stat,
                    cv_95=cv95,
                    cv_99=cv99,
                    p_value=pval,
                )
            )
        trace_df = pd.DataFrame(rows)

        # --- sequential rank selection ---------------------------------
        selected_rank = None
        if select_rank:
            for r in range(p):
                row = trace_df.iloc[r]
                pv = row["p_value"]
                stat = row["trace_stat"]
                cv = quantile(1 - sig_level, p - r, self.n, model=self.spec.code)
                reject = (pv < sig_level) if np.isfinite(pv) else (stat > cv)
                if not reject:
                    selected_rank = r
                    break
            if selected_rank is None:
                selected_rank = p  # full rank, stationary system

        # --- estimates at the chosen rank ------------------------------
        alpha = beta_X = delta_F = gamma_c = gamma_t = None
        Gamma_list = None
        Phi_un = None
        Sigma = None
        residuals = None
        fitted_long_run = None
        if compute_estimates and selected_rank is not None and selected_rank > 0:
            r = selected_rank
            beta_full = V[:, :r]                 # shape (q1, r)
            # alpha = S_{01} beta (beta' S_{11} beta)^{-1}
            mid = beta_full.T @ S11 @ beta_full
            alpha = S01 @ beta_full @ np.linalg.pinv(mid)
            # Split beta_full into (beta on X, delta on F, gamma on const/trend)
            offset = 0
            beta_X = beta_full[offset : offset + p, :]
            offset += p
            if self.spec.z1_fourier and self.n > 0:
                delta_F = beta_full[offset : offset + 2 * self.n, :]
                offset += 2 * self.n
            if self.spec.z1_constant:
                gamma_c = beta_full[offset : offset + 1, :]
                offset += 1
            if self.spec.z1_trend:
                gamma_t = beta_full[offset : offset + 1, :]
                offset += 1

            # Estimate Phi (coefficients on Z2) and residuals.
            if Z2.shape[1] > 0:
                # Conditional on (alpha beta'), regress Z0 - alpha beta' Z1
                # on Z2. Equivalent to regressing (Z0 | Z2, alpha beta' Z1).
                rhs = np.concatenate(
                    [Z2, Z1 @ beta_full], axis=1
                )
                coef, *_ = np.linalg.lstsq(rhs, Z0, rcond=None)
                Phi_un = coef[: Z2.shape[1], :]
                # second block is alpha (it should match the closed-form)
            # Long-run fitted: beta_full' Z1 (without alpha, this is the
            # cointegrating relation).
            fitted_long_run = Z1 @ beta_full
            # Residuals from the full ECM:
            ecm = Z0 - (Z1 @ beta_full) @ alpha.T
            if Z2.shape[1] > 0 and Phi_un is not None:
                ecm = ecm - Z2 @ Phi_un
            residuals = ecm
            Sigma = residuals.T @ residuals / T_eff

            # Pack Gamma_i for convenience
            if Z2.shape[1] > 0 and Phi_un is not None:
                Gamma_list = []
                idx = 0
                for _ in range(self.k - 1):
                    Gamma_list.append(Phi_un[idx : idx + p, :].T)
                    idx += p

        return JohansenFourierResults(
            spec=self.spec,
            p=p,
            k=self.k,
            n=self.n,
            T_eff=T_eff,
            eigenvalues=eigvals,
            eigenvectors=V,
            trace_stats=trace_df,
            selected_rank=selected_rank,
            alpha=alpha,
            beta=beta_X,
            delta=delta_F,
            gamma_const=gamma_c,
            gamma_trend=gamma_t,
            Gamma=Gamma_list,
            Phi_unrestricted=Phi_un,
            Sigma=Sigma,
            residuals=residuals,
            fitted_long_run=fitted_long_run,
            series_names=self.series_names,
            t_index=info["t_index"],
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def fit_johansen_fourier(
    data,
    k: int,
    n: int = 1,
    model: str = "CNR",
    **fit_kwargs,
) -> JohansenFourierResults:
    """One-shot wrapper around :class:`JohansenFourier`."""
    return JohansenFourier(data, k=k, n=n, model=model).fit(**fit_kwargs)
