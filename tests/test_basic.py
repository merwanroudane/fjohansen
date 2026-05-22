"""
Light-weight smoke tests covering the main public API of ``fjohansen``.

Run with::

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fjohansen as fj


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------
def test_fourier_basis_shape_and_periodicity():
    F = fj.fourier_basis(T=120, n=3)
    assert F.shape == (120, 6)
    # First-frequency sin should have period T
    s1 = F[:, 0]
    assert pytest.approx(s1[0], abs=1e-12) == np.sin(2 * np.pi / 120)


def test_fourier_basis_n_zero():
    F = fj.fourier_basis(T=50, n=0)
    assert F.shape == (50, 0)


# ---------------------------------------------------------------------------
# Critical values
# ---------------------------------------------------------------------------
def test_table_b1_lookup():
    cv95 = fj.quantile("95%", p_minus_r=5, n=1, model="CNR")
    assert cv95 == pytest.approx(110.64)


def test_gamma_p_value_is_in_unit_interval():
    p = fj.p_value(50.0, p_minus_r=3, n=2, model="CNR")
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# JohansenFourier on synthetic data
# ---------------------------------------------------------------------------
def test_fit_on_jgb_surrogate():
    data = fj.sample_jgb_data(T=120, seed=11)
    res = fj.JohansenFourier(data, k=2, n=1, model="CNR").fit(
        compute_pvalues=True, n_sims=2_000
    )
    assert res.trace_stats.shape[0] == data.shape[1]
    # selected rank should be an int in [0, p]
    assert res.selected_rank is not None
    assert 0 <= res.selected_rank <= data.shape[1]
    # All p-values in [0, 1]
    pv = res.trace_stats["p_value"].values
    pv = pv[np.isfinite(pv)]
    assert ((0 <= pv) & (pv <= 1)).all()


def test_models_are_callable():
    data = fj.sample_jgb_data(T=80, seed=3)
    for code in ("CNR", "LNR", "CNU", "LNU", "CONSTANT", "LINEAR"):
        res = fj.JohansenFourier(
            data, k=2, n=1 if "N" in code else 0, model=code
        ).fit(compute_estimates=False, compute_pvalues=False, select_rank=False)
        assert res.trace_stats.shape[0] == data.shape[1]


# ---------------------------------------------------------------------------
# PSY frequency selection
# ---------------------------------------------------------------------------
def test_psy_wald_test_runs():
    rng = np.random.default_rng(0)
    T = 200
    t = np.arange(T)
    y = (
        0.5 * np.sin(2 * np.pi * t / T)
        + 0.3 * np.cos(2 * np.pi * t / T)
        + rng.standard_normal(T) * 0.2
    )
    r = fj.psy_wald_test(y, k_freqs=[1, 2], p_d=1)
    assert r.df == 4
    assert 0.0 <= r.p_value <= 1.0


def test_select_frequencies_multivariate():
    data = fj.sample_jgb_data(T=120, seed=5)
    sel = fj.select_frequencies(data, n_max=3, p_d=1, sig_level=0.10)
    assert 0 <= sel.n_selected <= 3
    assert sel.per_series is not None and len(sel.per_series) == data.shape[1]


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn", [
        fj.generate_nf_dgp1,
        fj.generate_nf_dgp2,
        fj.generate_nf_dgp3,
        fj.generate_nf_dgp4,
    ],
)
def test_nonfourier_dgps(fn):
    X = fn(T=120, seed=42)
    assert X.shape == (120, 2)


def test_fourier_dgps():
    X1 = fj.generate_f_dgp1(T=120, seed=0)
    X2 = fj.generate_f_dgp2(T=120, seed=0)
    assert X1.shape == (120, 4)
    assert X2.shape == (120, 4)
