# fjohansen

[![PyPI version](https://img.shields.io/pypi/v/fjohansen.svg)](https://pypi.org/project/fjohansen/)
[![Python](https://img.shields.io/pypi/pyversions/fjohansen.svg)](https://pypi.org/project/fjohansen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/repo-merwanroudane%2Ffjohansen-black)](https://github.com/merwanroudane/fjohansen)

**`fjohansen`** is a Python implementation of the **Johansen cointegration test
with Fourier-type smooth nonlinear deterministic trends restricted to
cointegrating relations**, as developed in

> **Kurita, T. & Shintani, M. (2025)**. *Johansen test with Fourier-type
> smooth nonlinear trends in cointegrating relations.* **Econometric Reviews**,
> 44(10), 1589–1616. [DOI: 10.1080/07474938.2025.2530640](https://doi.org/10.1080/07474938.2025.2530640)

The library also bundles the FGLS Wald test of

> **Perron, P., Shintani, M. & Yabu, T. (2017, 2021)**. *Testing for flexible
> nonlinear trends with an integrated or stationary noise component.*

used by Kurita & Shintani as a frequency-selection pre-step.

> **Author:** [Merwan Roudane](https://github.com/merwanroudane)
> **Repository:** <https://github.com/merwanroudane/fjohansen>
> **PyPI:** <https://pypi.org/project/fjohansen/>

---

## 📋 Table of contents

1. [Features](#-features)
2. [Installation](#-installation)
3. [Quick start](#-quick-start)
4. [Mathematical background](#-mathematical-background)
5. [Model variants](#-model-variants)
6. [Full API reference](#-full-api-reference)
   - [Core test class](#core-test-class-fjjohansenfourier)
   - [Frequency selection](#frequency-selection-perron-shintani-yabu-2021)
   - [Critical values & p-values](#critical-values--p-values)
   - [Limit-distribution simulator](#limit-distribution-simulator)
   - [Tables and exports](#tables-and-exports)
   - [Plotting](#plotting)
   - [Data-generating processes](#data-generating-processes)
   - [Low-level utilities](#low-level-utilities)
7. [Reproducing the paper](#-reproducing-the-paper)
8. [Performance & caching](#-performance--caching)
9. [Citation](#-citation)
10. [License](#-license)

---

## ✨ Features

* **Six model variants** — `CNR`, `LNR`, `CNU`, `LNU` (Fourier in / out of the
  cointegrating space) plus standard constant- and linear-trend restricted
  Johansen models for direct comparison.
* **Trace statistics** with hard-coded critical values from Table B1 of the
  paper and Gamma-approximation *p-values* (Doornik 1998).
* **Sequential cointegrating-rank selection** following Johansen's standard
  procedure.
* **PSY (2021) FGLS Wald test** with Prais–Winsten transformation, Roy–Fuller
  bias correction and super-efficient AR(1) estimator + general-to-specific
  algorithm for picking the number of Fourier frequencies.
* **Publication-quality output**
  * ASCII table summary (`.summary()`)
  * LaTeX booktabs export (`.to_latex()`)
  * Styled HTML export (`.to_html()`)
  * Colourised terminal output (`rich_print()`)
  * Paper-style figures (eigenvalues, long-run relations, recursive
    rejection curves, limit-distribution densities, residual diagnostics,
    risk-premium decomposition).
* **All paper DGPs** (NF-DGP-1..4, F-DGP-1, F-DGP-2) plus a synthetic JGB-yield
  surrogate for replication.
* **Fast** — vectorised simulator, bundled moment tables for every common
  cell, persistent disk cache. Cold-start fit on the 6-variable JGB panel
  is < 0.05 s.

---

## 📦 Installation

### From PyPI (recommended)

```bash
pip install fjohansen
```

with optional pretty-printing extras:

```bash
pip install "fjohansen[pretty]"        # rich + tabulate
```

### From GitHub (latest dev)

```bash
git clone https://github.com/merwanroudane/fjohansen.git
cd fjohansen
pip install -e .
```

### Dependencies

* Python ≥ 3.9
* `numpy`, `scipy`, `pandas`, `matplotlib`, `statsmodels`
* optional: `rich`, `tabulate` (for `pretty` console output)

---

## 🚀 Quick start

```python
import fjohansen as fj

# 1. Load (or simulate) a multivariate I(1) panel
data = fj.sample_jgb_data(T=108)

# 2. Pick the number of Fourier frequencies (PSY 2021)
sel = fj.select_frequencies(data, n_max=5, p_d=1, sig_level=0.10)
print(f"Selected n = {sel.n_selected}")

# 3. Run the Johansen-Fourier (CNR) test
res = fj.JohansenFourier(data, k=3, n=sel.n_selected, model="CNR").fit()

# 4. Output
print(res.summary())                    # Table-3 style ASCII
fj.rich_print(res)                      # coloured terminal
open("out.tex", "w").write(res.to_latex())
open("out.html","w").write(res.to_html())

# 5. Figures
res.plot_eigenvalues()
res.plot_long_run()
res.plot_residual_diagnostics()
res.plot_risk_premium()
```

---

## 🔬 Mathematical background

For a *p*-dimensional I(1) process $X_t$ the package estimates

$$
\Delta X_t \;=\; \alpha\!\left(\beta' X_{t-1} + \delta' F_{t,T} + \gamma'\right)
\;+\;\sum_{i=1}^{k-1}\Gamma_i\,\Delta X_{t-i}\;+\;\Phi D_t\;+\;\varepsilon_t,
$$

where the Fourier basis

$$
F_{t,T} = \bigl[\sin\tfrac{2\pi t}{T},\ \cos\tfrac{2\pi t}{T},\ldots,
\sin\tfrac{2\pi n t}{T},\ \cos\tfrac{2\pi n t}{T}\bigr]'
$$

captures slow, smooth nonlinear movement *inside* the cointegrating
relations. The test statistic

$$
-2\log\!\mathit{LR}(r\mid p; n) \;=\; -T\!\!\sum_{i=r+1}^{p}\log\bigl(1-\hat\lambda_i\bigr)
$$

obtained from the Gaussian reduced-rank regression follows the limiting
distribution

$$
\operatorname{tr}\!\left\{\int_0^1\!\mathrm{d}B_u\,G_u'
\!\left(\!\int_0^1 G_u G_u'\,\mathrm{d}u\!\right)^{-1}\!\!\int_0^1 G_u\,\mathrm{d}B_u'\right\}
$$

(Proposition 3.1 of Kurita & Shintani, 2025). `fjohansen` simulates this
distribution, fits a Gamma matching its two leading moments (Doornik 1998),
and returns the resulting *p*-values.

---

## 🧩 Model variants

| Code        | $Z_1$ (cointegrating space)                          | $Z_2$ (unrestricted)                          |
|-------------|-------------------------------------------------------|------------------------------------------------|
| `CNR`       | $(X_{t-1}',\ F_{t,T}',\ 1)'$                          | $\Delta X$-lags only                           |
| `LNR`       | $(X_{t-1}',\ F_{t,T}',\ t)'$                          | $\Delta X$-lags + constant                     |
| `CNU`       | $(X_{t-1}',\ 1)'$                                     | $\Delta X$-lags + $F_{t,T}$                    |
| `LNU`       | $(X_{t-1}',\ t)'$                                     | $\Delta X$-lags + constant + $F_{t,T}$         |
| `constant`  | $(X_{t-1}',\ 1)'$                                     | $\Delta X$-lags only                           |
| `linear`    | $(X_{t-1}',\ t)'$                                     | $\Delta X$-lags + constant                     |

* `CNR` / `LNR` correspond to eqs. (6)–(7) of the paper (the main contribution).
* `CNU` / `LNU` correspond to Section 4 (Fourier unrestricted, no cointegration possible inside it).
* `constant` / `linear` are the classical Johansen specifications.

---

## 📚 Full API reference

### Core test class: `fj.JohansenFourier`

```python
class JohansenFourier(
    data,                 # array-like or DataFrame, shape (T, p)
    k: int,               # VAR order in levels
    n: int = 1,           # number of Fourier frequencies (0 = standard Johansen)
    model: str = "CNR",   # one of CNR | LNR | CNU | LNU | constant | linear
)
```

Methods:

```python
.fit(
    sig_level: float = 0.05,
    select_rank: bool = True,
    compute_pvalues: bool = True,
    compute_estimates: bool = True,
    n_sims: int = 5_000,
) -> JohansenFourierResults
```

`JohansenFourierResults` exposes:

| Attribute                | Type            | Description                                             |
|--------------------------|-----------------|---------------------------------------------------------|
| `eigenvalues`            | `ndarray`       | Generalised eigenvalues $\hat\lambda_i$, descending.    |
| `trace_stats`            | `DataFrame`     | Per-null trace, $\lambda$, p-value, 5% and 1% c.v.      |
| `selected_rank`          | `int`           | Rank chosen by the sequential 5%-rule.                  |
| `alpha`, `beta`          | `ndarray`       | Loading and cointegrating-vector matrices.              |
| `delta`                  | `ndarray`       | Coefficients on the Fourier block ($2n \times r$).      |
| `gamma_const`, `gamma_trend` | `ndarray`   | Restricted intercept / slope inside $\beta$-space.      |
| `Gamma`                  | `list[ndarray]` | Coefficients on $\Delta X_{t-i}$.                       |
| `Phi_unrestricted`       | `ndarray`       | Coefficients on the $Z_2$ block.                        |
| `Sigma`                  | `ndarray`       | Residual covariance matrix.                             |
| `residuals`              | `ndarray`       | $T_{\text{eff}}\times p$ residuals.                     |
| `fitted_long_run`        | `ndarray`       | Estimated cointegrating relations $\beta'X+\delta'F+\gamma$. |
| `t_index`                | `ndarray`       | Effective-sample time index.                            |

```python
.summary(sig_level=0.05) -> str          # ASCII (Table-3 style)
.to_latex(caption=None, label=None) -> str
.to_html(caption=None) -> str
.plot_eigenvalues()
.plot_long_run()
.plot_residual_diagnostics()
.plot_risk_premium(index=None, titles=None)
```

#### Example

```python
res = fj.JohansenFourier(data, k=3, n=5, model="CNR").fit(sig_level=0.05)
print(res.summary())

# r = res.selected_rank  -- number of cointegrating relations
# beta  = res.beta       -- (p, r) cointegrating vectors
# delta = res.delta      -- (2n, r) Fourier loadings
# alpha = res.alpha      -- (p, r) adjustment speeds
```

---

### Frequency selection (Perron-Shintani-Yabu 2021)

```python
fj.psy_wald_test(
    y,                    # univariate series, shape (T,)
    k_freqs,              # sequence of frequency indices, e.g. [1, 2, 3]
    p_d: int = 1,         # polynomial order: 0 = const, 1 = const+t
    subset_freq=None,     # if int, test only this frequency's two coefficients
    version="upper-biased",
    p_T_max=None,
)
```

Returns a `PSYTestResult` with `.stat`, `.df`, `.p_value`,
`.alpha_hat`, `.alpha_corrected`, `.alpha_S`.

```python
fj.select_frequencies_univariate(
    y,                    # univariate
    n_max=5,
    p_d=1,
    sig_level=0.10,
    version="upper-biased",
) -> FrequencySelectionResult

fj.select_frequencies(
    data,                 # multivariate (T, p)
    n_max=5,
    p_d=1,
    sig_level=0.10,
    version="upper-biased",
) -> FrequencySelectionResult
```

`FrequencySelectionResult` attributes: `n_selected`, `per_series` (DataFrame
of per-column n), `detail` (DataFrame of every step's W-stat / p-value).

#### Example

```python
sel = fj.select_frequencies(data, n_max=5, p_d=1, sig_level=0.10)
print(sel.per_series)        # per-series chosen n
print(sel.detail)             # full step-by-step table
print(f"Final n = {sel.n_selected}")
```

---

### Critical values & p-values

```python
fj.quantile(level, p_minus_r, n, model="CNR")
# level can be 0.95 or '95%'
# Uses Table B1 of the paper when available, otherwise Gamma approx.

fj.p_value(stat, p_minus_r, n, model="CNR")
# Gamma-approximation p-value of an observed trace statistic.

fj.moments(p_minus_r, n, model="CNR") -> (mean, var)
# First two moments of the limiting distribution.
```

The full Table B1 is also exposed as `fj.CNR_TABLE_B1[(p_minus_r, n)]`.

#### Example

```python
cv95 = fj.quantile("95%", p_minus_r=5, n=1, model="CNR")   # 110.64
pv   = fj.p_value(120.0, p_minus_r=5, n=1, model="CNR")    # ~0.03
```

---

### Limit-distribution simulator

```python
fj.simulate_limit_distribution(
    p_minus_r,
    n,
    model="CNR",
    n_sims=5_000,
    grid_size=300,
    seed=12345,
    use_cache=True,
) -> ndarray of shape (n_sims,)

fj.simulate_limit_moments(p_minus_r, n, model="CNR", ...) -> (mean, var, draws)

fj.clear_cache() -> int        # wipe ~/.fjohansen/cache
```

The simulator is **fully vectorised**: all `n_sims` replications run in a
single `numpy.matmul` call. Results are cached on disk under
`~/.fjohansen/cache/` (override via the `FJOHANSEN_CACHE` env var) so
identical calls are instant.

#### Example

```python
draws = fj.simulate_limit_distribution(p_minus_r=3, n=2, model="LNR",
                                       n_sims=10_000)
import matplotlib.pyplot as plt
plt.hist(draws, bins=100, density=True)
```

---

### Tables and exports

```python
fj.format_trace_table(df, sig_level=0.05) -> str
fj.format_trace_latex(df, spec, n, caption=None, label=None) -> str
fj.format_trace_html(df, spec, n, caption=None) -> str
fj.rich_print(results) -> None         # Rich-coloured terminal output
```

#### Example

```python
print(res.summary())
# ========================================================================
#   Johansen-Fourier Cointegration Test (Kurita & Shintani, 2025)
# ========================================================================
#   Model               : CNR  (Constant + Nonlinear restricted)
#   Variables (p)       : 6
#   Lag order (k)       : 3
#   ...
#   H0           eigenvalue   trace stat    p-value   5% c.v.  1% c.v.
#   --------------------------------------------------------------------
#   r <= 0          0.567       376.154   <0.001**   224.55   238.17
#   r <= 1          0.534       293.944   <0.001**   177.87   190.44
#   ...
```

---

### Plotting

```python
fj.set_paper_style()                  # called automatically on import

fj.plot_series(data, title=None, ncols=2)            # Fig. 11
fj.plot_limit_density(p_minus_r, n_values=(0,1,2,3,4), model="CNR")  # Figs. 1-2
fj.plot_recursive_rejection({label: (Ts, rates)}, nominal=0.05)      # Figs. 3-10

# Bound on a JohansenFourierResults:
res.plot_eigenvalues()
res.plot_long_run()
res.plot_residual_diagnostics()      # Fig. 12
res.plot_risk_premium(index=data.index)   # Fig. 13
```

Every function returns a `matplotlib.figure.Figure` you can save with
`fig.savefig("out.pdf")`.

---

### Data-generating processes

```python
# Non-Fourier DGPs from Section 5.1 (bivariate, T x 2 DataFrame)
fj.generate_nf_dgp1(T=400, seed=0)          # no cointegration
fj.generate_nf_dgp2(T=400, seed=0)          # level shift
fj.generate_nf_dgp3(T=400, seed=0)          # exponential transition (0.4 T)
fj.generate_nf_dgp4(T=400, seed=0)          # sharper transition (0.8 T)

# Fourier DGPs from Section 5.2 (4-variate)
fj.generate_f_dgp1(T=400, seed=0)           # r=1, n=1
fj.generate_f_dgp2(T=400, seed=0)           # r=2, n=2

# Synthetic JGB-yield surrogate matching Fig. 11
fj.sample_jgb_data(T=108, seed=7)           # 6 yields, monthly 1986-12..1995-11
```

---

### Low-level utilities

```python
fj.fourier_basis(T, n, t_start=1) -> ndarray of shape (T, 2n)

fj.build_design_matrices(X, k, n, model) -> (Z0, Z1, Z2, info)
# Build the matrices used in Johansen's reduced rank regression.

# Inspect or extend the model registry:
fj.MODELS                  # dict[str, ModelSpec]
fj.ModelSpec               # dataclass describing a deterministic spec
```

---

## 🧪 Reproducing the paper

| Paper section                | API entry-point                                            |
|------------------------------|-------------------------------------------------------------|
| §3 / Prop. 3.1 limit dist.   | `simulate_limit_distribution`, `plot_limit_density`         |
| §4 unrestricted Fourier      | `model='CNU'`, `model='LNU'`                                |
| §5 Monte Carlo DGPs          | `generate_nf_dgp1..4`, `generate_f_dgp1..2`                 |
| §6 JGB application           | `sample_jgb_data` + `JohansenFourier(..., 'CNR')`           |
| App. B critical values       | `CNR_TABLE_B1`, `quantile(...)`                             |

The `examples/` directory has three runnable scripts:

| File                                         | What it reproduces                                                       |
|----------------------------------------------|--------------------------------------------------------------------------|
| `examples/example_section6_jgb.py`           | Full §6 workflow on the synthetic JGB data; writes LaTeX + HTML tables.  |
| `examples/example_section5_monte_carlo.py`   | F-DGP-1 rank-selection rates: CNR(n=1) vs. standard linear model.        |
| `examples/example_limit_distributions.py`    | A simplified version of Fig. 1.                                          |

---

## ⚡ Performance & caching

Cold-start times on a laptop (Windows, NumPy with OpenBLAS, 6-variable JGB panel):

| Operation                                  | Cold start | Warm cache |
|--------------------------------------------|------------|------------|
| `JohansenFourier(..., 'CNR', n=3).fit()`   | 0.01 s     | 0.01 s     |
| `JohansenFourier(..., 'LNR', n=3).fit()`   | 0.01 s     | 0.01 s     |
| Any 4-d / 8-d cell in the bundled tables   | 0.01 s     | 0.01 s     |
| `simulate_limit_distribution(.., 5000)`    | 0.5–2 s    | 0.02 s     |
| `select_frequencies(panel, n_max=5)`       | 0.06 s     | 0.06 s     |

The disk cache lives under `~/.fjohansen/cache/`. Wipe it with
`fj.clear_cache()` or set the `FJOHANSEN_CACHE` environment variable to
relocate it.

---

## 📑 Citation

If you use `fjohansen` in academic work, please cite the underlying paper
**and** the package:

```bibtex
@article{kurita_shintani_2025,
  author  = {Takamitsu Kurita and Mototsugu Shintani},
  title   = {Johansen test with Fourier-type smooth nonlinear trends in cointegrating relations},
  journal = {Econometric Reviews},
  year    = {2025},
  volume  = {44},
  number  = {10},
  pages   = {1589--1616},
  doi     = {10.1080/07474938.2025.2530640}
}

@article{perron_shintani_yabu_2017,
  author  = {Pierre Perron and Mototsugu Shintani and Tomoyoshi Yabu},
  title   = {Testing for flexible nonlinear trends with an integrated or stationary noise component},
  journal = {Oxford Bulletin of Economics and Statistics},
  volume  = {79},
  pages   = {822--850},
  year    = {2017},
  doi     = {10.1111/obes.12169}
}

@software{roudane_fjohansen,
  author  = {Merwan Roudane},
  title   = {fjohansen: Johansen test with Fourier-type smooth nonlinear trends (Python)},
  year    = {2025},
  url     = {https://github.com/merwanroudane/fjohansen}
}
```

---

## 🐛 Issues & contributions

Bug reports, questions and pull requests welcome at
<https://github.com/merwanroudane/fjohansen/issues>.

---

## 📜 License

MIT — see [LICENSE](LICENSE).
