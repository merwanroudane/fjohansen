"""
Build the tutorial notebook examples/fjohansen_tutorial.ipynb.

Run once from the repo root:
    python examples/_build_tutorial_notebook.py
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

NB_PATH = pathlib.Path(__file__).parent / "fjohansen_tutorial.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


cells = []

# ---------------------------------------------------------------------------
cells.append(md(
    """# 📈 `fjohansen` — Johansen Cointegration Test with Fourier-Type Nonlinear Trends

**A complete, real-world tutorial.**

This notebook walks through every public feature of the `fjohansen` Python
library implementing the test of

> **Kurita, T. & Shintani, M. (2025).** *Johansen test with Fourier-type smooth
> nonlinear trends in cointegrating relations.* **Econometric Reviews**, 44(10),
> 1589–1616. [DOI](https://doi.org/10.1080/07474938.2025.2530640)

We will use **real US Treasury yield data from FRED** (Federal Reserve Bank of
St. Louis) — the perfect modern analogue of the Japanese government-bond
application of Section 6 of the paper.

---

**Outline**

1. Setup
2. Fetch real US Treasury yields (FRED)
3. Frequency selection (Perron–Shintani–Yabu 2021)
4. Standard Johansen test (baseline)
5. **Johansen–Fourier CNR test** (the main contribution)
6. Side-by-side comparison
7. Long-run cointegrating relations
8. Implied risk-premium decomposition (paper's Figure 13)
9. Residual diagnostics (paper's Figure 12)
10. Limit-distribution figure (paper's Figure 1)
11. Publication-quality table exports (LaTeX, HTML)

---

> **Author:** [Merwan Roudane](https://github.com/merwanroudane)
> **Package:** [`fjohansen` on PyPI](https://pypi.org/project/fjohansen/) ·
> [GitHub](https://github.com/merwanroudane/fjohansen)
"""
))

# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup\n\nInstall the package and import the libraries we'll need."))

cells.append(code("""# In Colab / fresh environments, uncomment the next line:
# !pip install -q fjohansen rich

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fjohansen as fj
print("fjohansen version:", fj.__version__)
print("Python author    :", fj.__author__)
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 2. Fetch real US Treasury yield data (FRED)

We pull seven constant-maturity Treasury yields directly from FRED via its
public CSV endpoint (no API key required) and average them to monthly
frequency over **1990-01 to 2019-12** — a sample of length T = 360 that spans
several easing / tightening cycles, two recessions and the post-crisis
zero-lower-bound period.

| FRED code | Maturity         |
|-----------|------------------|
| `DGS3MO`  | 3-month          |
| `DGS6MO`  | 6-month          |
| `DGS1`    | 1-year           |
| `DGS2`    | 2-year           |
| `DGS5`    | 5-year           |
| `DGS10`   | 10-year          |
| `DGS30`   | 30-year          |
"""))

cells.append(code(r"""
def fetch_fred_series(code: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    df = pd.read_csv(url, na_values=".")
    # FRED's date column has been DATE in the past and observation_date
    # more recently -- detect it dynamically.
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", code: "value"}).dropna()
    return df.set_index("date")["value"].astype(float)

codes = {
    "y_3m":  "DGS3MO",
    "y_6m":  "DGS6MO",
    "y_1y":  "DGS1",
    "y_2y":  "DGS2",
    "y_5y":  "DGS5",
    "y_10y": "DGS10",
    "y_30y": "DGS30",
}

data_daily = pd.concat(
    {name: fetch_fred_series(code) for name, code in codes.items()},
    axis=1,
)

# Monthly mean, intersection of all available series, our sample window
data = (
    data_daily.resample("MS").mean()
              .loc["1990-01":"2019-12"]
              .dropna()
)
print(f"Sample: {data.index.min().date()}  ->  {data.index.max().date()}")
print(f"Shape : {data.shape}   (T x p)")
data.tail()
"""))

cells.append(md("""### Visualise the yield curve over time

These series share an obvious common nonstationary trend; the long–short
spread (term premium) clearly drifts through cycles."""))

cells.append(code("""fig = fj.plot_series(data, title="US Treasury constant-maturity yields, 1990-2019", ncols=2)
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 3. Frequency selection (Perron–Shintani–Yabu 2021)

Before running the rank test we have to pick the number of Fourier frequencies
$n$. We use the FGLS Wald test of Perron, Shintani & Yabu (2021) — robust to
both $I(0)$ and $I(1)$ noise — applied series-by-series in a general-to-specific
loop. The final $n$ is the maximum across the panel."""))

cells.append(code("""sel = fj.select_frequencies(data, n_max=5, p_d=1, sig_level=0.10)

print("Per-series selected n:")
display(sel.per_series.style.bar(subset=["n_selected"], color="#1f4e79", vmin=0, vmax=5))

print(f"\\n=> Multivariate final selection:  n = {sel.n_selected}")
"""))

cells.append(code("""# Per-step diagnostic table for the first series (full p-value path)
sel.detail.head(20).style.format({"stat": "{:.3f}", "p_value": "{:.4f}"})\\
   .background_gradient(subset=["p_value"], cmap="RdYlGn_r")
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 4. Baseline: standard Johansen test

For comparison, run the textbook constant-restricted Johansen test (no
Fourier component) first."""))

cells.append(code("""res_std = fj.JohansenFourier(data, k=3, n=0, model="constant").fit()
print(res_std.summary())
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 5. Main test: **Johansen–Fourier CNR**

Now the paper's headline procedure: the CNR model with the data-driven `n`
chosen above and a constant restricted inside the cointegrating space."""))

cells.append(code("""res = fj.JohansenFourier(
    data,
    k=3,
    n=max(sel.n_selected, 1),
    model="CNR",
).fit()

print(res.summary())
"""))

cells.append(code("""# Colourised terminal-style table (uses rich if installed)
fj.rich_print(res)
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 6. Side-by-side comparison

Term-structure theory implies $p - r = 1$ — i.e. **one single common stochastic
trend** (the short rate). The CNR model should detect more cointegrating
relations than the standard model because it correctly accommodates the
slow-moving term premium."""))

cells.append(code("""compare = pd.DataFrame({
    "H0":           res_std.trace_stats["null"],
    "trace (std.)": res_std.trace_stats["trace_stat"].round(2),
    "p-val (std.)": res_std.trace_stats["p_value"].round(3),
    "trace (CNR)":  res.trace_stats["trace_stat"].round(2),
    "p-val (CNR)":  res.trace_stats["p_value"].round(3),
}).set_index("H0")

def star(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return " **"
    if p < 0.05: return " *"
    if p < 0.10: return " ."
    return ""

styler = (
    compare.style
    .format("{:.3f}")
    .background_gradient(subset=["p-val (std.)", "p-val (CNR)"],
                         cmap="RdYlGn_r", vmin=0, vmax=0.5)
    .set_caption("Trace test: standard Johansen vs Johansen-Fourier (CNR)")
    .set_table_styles([
        {"selector": "caption", "props": [("font-size", "12pt"),
                                          ("font-weight", "bold"),
                                          ("padding", "10px")]},
    ])
)
styler
"""))

cells.append(code("""print(f"Standard model selected rank:        r = {res_std.selected_rank}  "
      f"=> {data.shape[1] - res_std.selected_rank} common stochastic trends")
print(f"Johansen-Fourier CNR selected rank:   r = {res.selected_rank}  "
      f"=> {data.shape[1] - res.selected_rank} common stochastic trends")
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 7. Estimated cointegrating relations

`fjohansen` recovers the cointegrating vectors $\hat\beta$, the Fourier
loadings $\hat\delta$, and the adjustment matrix $\hat\alpha$. The plot below
shows each long-run relation $\hat\beta_j' X_t + \hat\delta_j' F_{t,T} + \hat\gamma_j$."""))

cells.append(code("""res.plot_long_run()
plt.show()
"""))

cells.append(code("""# Numerical estimates -- normalised cointegrating vectors
if res.beta is not None:
    beta_df = pd.DataFrame(
        res.beta,
        index=list(data.columns),
        columns=[f"coint_vec_{j + 1}" for j in range(res.beta.shape[1])],
    )
    display(
        beta_df.style.format("{:.4f}")
        .background_gradient(cmap="RdBu_r", vmin=-2, vmax=2)
        .set_caption("Cointegrating vectors  (cols = relations)")
    )
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 8. Implied risk-premium decomposition (paper's Figure 13)

If we think of each cointegrating relation as a risk-premium series, the
estimated $\hat\delta' F_{t,T}$ block captures its **smooth nonlinear
component** (the business-cycle component), while the residual captures the
short-run dynamics."""))

cells.append(code("""if res.delta is not None and res.n > 0 and res.selected_rank:
    res.plot_risk_premium(index=data.index)
    plt.show()
else:
    print("No Fourier component recovered; skip this plot.")
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 9. Residual diagnostics (paper's Figure 12)

Scaled residuals, autocorrelation functions and Q–Q plots for the fitted
ECM, panel by panel."""))

cells.append(code("""res.plot_residual_diagnostics()
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 10. Eigenvalues of the reduced-rank regression

A bar chart of the $\hat\lambda_i$. Bars below the selected rank are
highlighted."""))

cells.append(code("""res.plot_eigenvalues()
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 11. Limit distribution under the null (paper's Figure 1)

The limiting distribution of $-2\log\!\mathit{LR}(r\mid p; n)$ shifts to the
right and becomes less skewed as $n$ grows. This is exactly the Figure 1
behaviour of Kurita & Shintani (2025)."""))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, p_r in zip(axes.ravel(), [4, 3, 2, 1]):
    for n in [0, 1, 2, 3]:
        model = "CNR" if n > 0 else "CONSTANT"
        draws = fj.simulate_limit_distribution(
            p_r, n if n > 0 else 0, model=model,
            n_sims=4000, grid_size=300, seed=10 + n,
        )
        ax.hist(draws, bins=80, density=True, histtype="step",
                linewidth=1.6 if n == 0 else 1.0,
                label=f"n = {n}" + (" (linear)" if n == 0 else ""))
    ax.set_title(f"p - r = {p_r}", loc="left")
    ax.set_xlabel("trace statistic")
    ax.legend(fontsize=8)
fig.suptitle("Limit distributions: Johansen-Fourier (CNR) vs standard linear",
             y=0.99, fontsize=12)
fig.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 12. Critical values & p-values

The full Table B1 of the paper is bundled in the package, and `fj.quantile()`
falls back to the Gamma approximation outside it."""))

cells.append(code("""print("CNR critical values, n = 1..5,  p - r = 5  (Table B1):")
rows = []
for n in range(1, 6):
    row = {"n": n,
           "90%":   fj.quantile("90%",   5, n, model="CNR"),
           "95%":   fj.quantile("95%",   5, n, model="CNR"),
           "97.5%": fj.quantile("97.5%", 5, n, model="CNR"),
           "99%":   fj.quantile("99%",   5, n, model="CNR")}
    rows.append(row)

df_cv = pd.DataFrame(rows).set_index("n")
df_cv.style.format("{:.2f}").background_gradient(cmap="YlOrBr")\
     .set_caption("Critical values of the Johansen-Fourier trace statistic (CNR, p - r = 5)")
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 13. Publication-quality table exports

Export the trace-test table to LaTeX (booktabs) or HTML for inclusion in a
paper / report."""))

cells.append(code("""print(res.to_latex(caption="US Treasury yields: Johansen-Fourier trace test", label="tab:us_jf"))
"""))

cells.append(code("""from IPython.display import HTML
HTML(res.to_html(caption="US Treasury yields: Johansen-Fourier (CNR) trace test"))
"""))

# ---------------------------------------------------------------------------
cells.append(md(r"""## 14. Take-aways

* `fjohansen` makes the Kurita-Shintani test usable in two lines.
* On real Treasury yield data, ignoring the smooth term-premium drift
  produces fewer detected cointegrating relations than the theory predicts.
* All six model variants (`CNR`, `LNR`, `CNU`, `LNU`, `constant`, `linear`)
  share the same fluent API.

For more examples and reproduction of the paper's Monte Carlo, see
[`examples/`](https://github.com/merwanroudane/fjohansen/tree/main/examples)
in the repository.

---

If `fjohansen` is useful in your research, please cite:

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

@software{roudane_fjohansen,
  author  = {Merwan Roudane},
  title   = {fjohansen: Johansen test with Fourier-type smooth nonlinear trends (Python)},
  year    = {2025},
  url     = {https://github.com/merwanroudane/fjohansen}
}
```
"""))


# ===========================================================================
# Assemble & write
# ===========================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
    "title": "fjohansen tutorial -- Johansen-Fourier cointegration on US Treasury yields",
    "authors": [{"name": "Merwan Roudane"}],
}

NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size:,} bytes)")
