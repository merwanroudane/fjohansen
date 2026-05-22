"""
fjohansen
=========

Python implementation of the Johansen cointegration test with Fourier-type
smooth nonlinear deterministic trends restricted to cointegrating relations,
as developed in:

    Kurita, T. and Shintani, M. (2025), "Johansen test with Fourier-type
    smooth nonlinear trends in cointegrating relations", *Econometric
    Reviews*, 44(10), 1589-1616. https://doi.org/10.1080/07474938.2025.2530640

The library also bundles the FGLS Wald test of

    Perron, P., Shintani, M. and Yabu, T. (2017, 2021), "Testing for flexible
    nonlinear trends with an integrated or stationary noise component",
    *Oxford Bulletin of Economics and Statistics* / Working Paper.

used by Kurita & Shintani as a frequency-selection pre-step.

Package metadata
----------------
:author:  Merwan Roudane
:repo:    https://github.com/merwanroudane/fjohansen
:license: MIT
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Merwan Roudane"
__url__ = "https://github.com/merwanroudane/fjohansen"

from .core import JohansenFourier, JohansenFourierResults, fit_johansen_fourier
from .critical_values import CNR_TABLE_B1, p_value, quantile, moments
from .frequency_select import (
    psy_wald_test,
    select_frequencies,
    select_frequencies_univariate,
    FrequencySelectionResult,
    PSY_TAU_TABLE,
)
from .models import MODELS, ModelSpec, build_design_matrices
from .simulation import (
    simulate_limit_distribution,
    simulate_limit_moments,
    clear_cache,
)
from .data import (
    generate_nf_dgp1,
    generate_nf_dgp2,
    generate_nf_dgp3,
    generate_nf_dgp4,
    generate_f_dgp1,
    generate_f_dgp2,
    sample_jgb_data,
)
from .tables import (
    format_trace_table,
    format_trace_latex,
    format_trace_html,
    rich_print,
)
from .plotting import (
    set_paper_style,
    plot_series,
    plot_limit_density,
    plot_recursive_rejection,
    plot_eigenvalues,
    plot_long_run,
    plot_risk_premium,
    plot_residual_diagnostics,
)
from .utils import fourier_basis

__all__ = [
    "__version__",
    "__author__",
    "__url__",
    # core
    "JohansenFourier",
    "JohansenFourierResults",
    "fit_johansen_fourier",
    # critical values
    "p_value",
    "quantile",
    "moments",
    "CNR_TABLE_B1",
    # frequency selection
    "psy_wald_test",
    "select_frequencies",
    "select_frequencies_univariate",
    "FrequencySelectionResult",
    "PSY_TAU_TABLE",
    # models
    "MODELS",
    "ModelSpec",
    "build_design_matrices",
    # simulation
    "simulate_limit_distribution",
    "simulate_limit_moments",
    "clear_cache",
    # data
    "generate_nf_dgp1",
    "generate_nf_dgp2",
    "generate_nf_dgp3",
    "generate_nf_dgp4",
    "generate_f_dgp1",
    "generate_f_dgp2",
    "sample_jgb_data",
    # tables
    "format_trace_table",
    "format_trace_latex",
    "format_trace_html",
    "rich_print",
    # plotting
    "set_paper_style",
    "plot_series",
    "plot_limit_density",
    "plot_recursive_rejection",
    "plot_eigenvalues",
    "plot_long_run",
    "plot_risk_premium",
    "plot_residual_diagnostics",
    # utils
    "fourier_basis",
]
