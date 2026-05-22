"""
Build the fjohansen GitHub Pages site under ``docs/``.

This script:
  1. Fetches real US Treasury yield data from FRED.
  2. Runs frequency selection, standard Johansen and Johansen-Fourier (CNR).
  3. Saves all figures (PNG) into ``docs/figures/``.
  4. Saves all tables (HTML fragments) into ``docs/tables/``.
  5. Assembles ``docs/index.html`` using a sunny theme.

Run from the repository root::

    python docs/build_site.py
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import fjohansen as fj  # noqa: E402  (after sys.path tweak)


DOCS = REPO_ROOT / "docs"
FIG = DOCS / "figures"
TBL = DOCS / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 0. Sunny matplotlib palette (override paper-style for the site)
# ===========================================================================
SUNNY = ["#f9a825", "#ef6c00", "#0288d1", "#7cb342", "#ba68c8", "#26a69a", "#ff7043"]
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#f57f17",
    "axes.labelsize": 10,
    "axes.labelcolor": "#3b3a36",
    "axes.edgecolor": "#e6d49a",
    "axes.linewidth": 1.0,
    "axes.facecolor": "#fffef6",
    "figure.facecolor": "#fffef6",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.color": "#f0e6c8",
    "grid.linestyle": ":",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.color": "#6b6a64",
    "ytick.color": "#6b6a64",
    "lines.linewidth": 1.6,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#fffef6",
    "axes.prop_cycle": plt.cycler(color=SUNNY),
})


def savefig(fig, name: str, caption: str = "") -> dict:
    """Save figure and return a dict suitable for the HTML template."""
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return {"name": name, "path": f"figures/{name}.png", "caption": caption}


def savetable(html: str, name: str) -> str:
    """Save table HTML fragment and return its relative URL."""
    path = TBL / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    return f"tables/{name}.html"


# ===========================================================================
# 1. Fetch real US Treasury yield data from FRED
# ===========================================================================
def fetch_fred_series(code: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    df = pd.read_csv(url, na_values=".")
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", code: "value"}).dropna()
    return df.set_index("date")["value"].astype(float)


print("[1/8] Fetching US Treasury yields from FRED ...")
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
    {n: fetch_fred_series(c) for n, c in codes.items()},
    axis=1,
)
data = data_daily.resample("MS").mean().loc["1990-01":"2019-12"].dropna()
T, p = data.shape
sample_start = data.index.min().strftime("%b %Y")
sample_end = data.index.max().strftime("%b %Y")
print(f"        data: {T} monthly obs x {p} maturities, {sample_start} .. {sample_end}")


# ===========================================================================
# 2. Plot the data
# ===========================================================================
print("[2/8] Plotting yields ...")
fig, ax = plt.subplots(figsize=(9.5, 4.5))
for j, (col, color) in enumerate(zip(data.columns, SUNNY)):
    ax.plot(data.index, data[col], label=col.replace("y_", "").upper(),
            color=color, linewidth=1.3)
ax.set_title("US Treasury constant-maturity yields, 1990-2019")
ax.set_xlabel("date")
ax.set_ylabel("yield (%, monthly average)")
ax.legend(ncol=7, loc="upper right", fontsize=8)
fig_yields = savefig(fig, "01_yields",
                     "Monthly average constant-maturity Treasury yields from FRED.")


# ===========================================================================
# 3. PSY frequency selection
# ===========================================================================
print("[3/8] PSY frequency selection ...")
sel = fj.select_frequencies(data, n_max=5, p_d=1, sig_level=0.10)
n_selected = max(sel.n_selected, 1)

# Per-series table
per_series_html = (
    sel.per_series.style
       .background_gradient(subset=["n_selected"], cmap="YlOrBr", vmin=0, vmax=5)
       .format({"n_selected": "{:d}"})
       .set_caption("PSY-2021 selected number of Fourier frequencies (per series)")
       .set_table_attributes('class="dataframe psy"')
       .to_html()
)
url_psy = savetable(per_series_html, "psy_per_series")


# ===========================================================================
# 4. Standard Johansen + Johansen-Fourier CNR
# ===========================================================================
print("[4/8] Fitting standard Johansen and Johansen-Fourier CNR ...")
res_std = fj.JohansenFourier(data, k=3, n=0, model="constant").fit()
res = fj.JohansenFourier(data, k=3, n=n_selected, model="CNR").fit()

# Side-by-side compare DataFrame
compare = pd.DataFrame({
    "H0":           res_std.trace_stats["null"],
    "trace (std.)": res_std.trace_stats["trace_stat"].round(3),
    "p-val (std.)": res_std.trace_stats["p_value"].round(4),
    "trace (CNR)":  res.trace_stats["trace_stat"].round(3),
    "p-val (CNR)":  res.trace_stats["p_value"].round(4),
}).set_index("H0")

compare_html = (
    compare.style
    .format({"trace (std.)": "{:.2f}", "trace (CNR)": "{:.2f}",
             "p-val (std.)": "{:.4f}", "p-val (CNR)": "{:.4f}"})
    .background_gradient(subset=["p-val (std.)", "p-val (CNR)"],
                         cmap="RdYlGn_r", vmin=0, vmax=0.5)
    .set_caption("Trace test: standard Johansen vs Johansen-Fourier (CNR)")
    .set_table_attributes('class="dataframe trace-compare"')
    .to_html()
)
url_compare = savetable(compare_html, "trace_comparison")


# ===========================================================================
# 5. Eigenvalues + Long-run plot + Risk premium + Residual diagnostics
# ===========================================================================
print("[5/8] Generating result plots ...")
fig = res.plot_eigenvalues(figsize=(8, 4))
fig_eig = savefig(fig, "02_eigenvalues",
                  f"Eigenvalues from the reduced-rank regression of the CNR(n={n_selected}) model. "
                  f"Bars are highlighted up to the selected rank r = {res.selected_rank}.")

fig = res.plot_long_run(figsize=(9, 1.7 * res.selected_rank + 1)) if res.selected_rank else None
if fig is not None:
    fig_lr = savefig(fig, "03_long_run",
                     f"Each cointegrating relation beta'X + delta'F + gamma "
                     f"for the {res.selected_rank} retained vectors.")
else:
    fig_lr = None

if res.delta is not None and res.n > 0 and res.selected_rank:
    eff_index = data.index[res.k:]   # effective sample after losing k initial obs
    fig = res.plot_risk_premium(index=eff_index, figsize=(9, 1.6 * res.selected_rank + 1))
    fig_rp = savefig(fig, "04_risk_premium",
                     "Implied risk-premium decomposition (Fig. 13 of the paper): the thick "
                     "line is the full risk-premium series, the dashed line is its smooth "
                     "nonlinear (Fourier) component.")
else:
    fig_rp = None

fig = res.plot_residual_diagnostics(figsize=(11, 1.4 * p + 0.5))
fig_diag = savefig(fig, "05_diagnostics",
                   "Residual diagnostics (Fig. 12 of the paper): standardised residuals, "
                   "ACF and Q-Q plots for each equation.")


# ===========================================================================
# 6. Limit distribution figure (Fig. 1 of the paper)
# ===========================================================================
print("[6/8] Simulating limit distributions ...")
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
for ax, p_r in zip(axes.ravel(), [4, 3, 2, 1]):
    for k, n in enumerate([0, 1, 2, 3]):
        model = "CNR" if n > 0 else "CONSTANT"
        draws = fj.simulate_limit_distribution(
            p_r, n if n > 0 else 0, model=model, n_sims=4000, grid_size=300, seed=10 + n,
        )
        ax.hist(draws, bins=80, density=True, histtype="step",
                linewidth=1.7 if n == 0 else 1.0,
                color=SUNNY[k % len(SUNNY)],
                label=f"n = {n}" + (" (linear)" if n == 0 else ""))
    ax.set_title(f"p - r = {p_r}", loc="left")
    ax.set_xlabel("trace statistic")
    ax.legend(fontsize=8)
fig.suptitle("Limit distributions: Johansen-Fourier (CNR) vs standard linear", y=0.99)
fig.tight_layout()
fig_limit = savefig(fig, "06_limit_density",
                    "Density of the limiting trace statistic for p - r in {4,3,2,1} and "
                    "n in {0,1,2,3}. The distribution shifts right and becomes less skewed "
                    "as n grows -- exactly the Fig. 1 behaviour of Kurita & Shintani (2025).")


# ===========================================================================
# 7. Critical-values table (Table B1 selection)
# ===========================================================================
print("[7/8] Building critical-values table ...")
rows = []
for p_r in (2, 3, 4, 5, 6, 7, 8):
    row = {"p-r": p_r}
    for n in range(1, 6):
        row[f"n={n} 95%"] = fj.quantile("95%", p_r, n, model="CNR")
    rows.append(row)
cv_df = pd.DataFrame(rows).set_index("p-r")
cv_html = (
    cv_df.style
    .format("{:.2f}")
    .background_gradient(cmap="YlOrBr")
    .set_caption("Approximate 95% critical values of the Johansen-Fourier trace statistic (CNR model)")
    .set_table_attributes('class="dataframe critical-values"')
    .to_html()
)
url_cv = savetable(cv_html, "critical_values")

# Cointegrating vectors heatmap
if res.beta is not None:
    beta_df = pd.DataFrame(
        res.beta,
        index=list(data.columns),
        columns=[f"v{j + 1}" for j in range(res.beta.shape[1])],
    )
    beta_html = (
        beta_df.style
        .format("{:.4f}")
        .background_gradient(cmap="RdBu_r", vmin=-2, vmax=2)
        .set_caption("Estimated cointegrating vectors (columns)")
        .set_table_attributes('class="dataframe beta-matrix"')
        .to_html()
    )
    url_beta = savetable(beta_html, "beta_matrix")
else:
    url_beta = None


# ===========================================================================
# 8. Write index.html
# ===========================================================================
print("[8/8] Writing index.html ...")
build_date = dt.datetime.utcnow().strftime("%Y-%m-%d")

def read(path):
    return pathlib.Path(path).read_text(encoding="utf-8")


HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fjohansen &mdash; Johansen-Fourier cointegration test in Python</title>
<meta name="description" content="Python library for the Johansen cointegration test with Fourier-type smooth nonlinear trends in cointegrating relations (Kurita & Shintani, 2025).">
<meta name="author" content="Merwan Roudane">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters:[
            {{left:'$$',right:'$$',display:true}},
            {{left:'$',right:'$',display:false}}
        ]}});"></script>
</head>
<body>

<header class="hero">
  <h1>&#9728; fjohansen</h1>
  <p class="subtitle">
    Johansen cointegration test with <em>Fourier-type smooth nonlinear trends</em>
    in cointegrating relations &mdash; a Python implementation of the test of
    Kurita &amp; Shintani (2025).
  </p>
  <p class="author">
    Developed by <strong>Merwan Roudane</strong> &middot;
    <a href="https://github.com/merwanroudane">github.com/merwanroudane</a>
  </p>
  <div class="badges">
    <a class="badge badge-pypi"   href="https://pypi.org/project/fjohansen/">&#128230; PyPI &middot; v{fj.__version__}</a>
    <a class="badge badge-github" href="https://github.com/merwanroudane/fjohansen">&#128279; GitHub</a>
    <a class="badge badge-doi"    href="https://doi.org/10.1080/07474938.2025.2530640">&#128214; Paper (DOI)</a>
    <a class="badge badge-mit"    href="https://github.com/merwanroudane/fjohansen/blob/main/LICENSE">&#128274; MIT licence</a>
    <a class="badge badge-python" href="https://www.python.org/">&#128013; Python 3.9+</a>
  </div>
</header>

<main>

<!-- ============ TOC ============ -->
<div class="toc">
  <h3>&#9776; Contents</h3>
  <ol>
    <li><a href="#intro">Introduction</a></li>
    <li><a href="#install">Install</a></li>
    <li><a href="#theory">Theory in five paragraphs</a></li>
    <li><a href="#data">Data &mdash; US Treasury yields</a></li>
    <li><a href="#psy">Frequency selection (PSY 2021)</a></li>
    <li><a href="#compare">Trace test &mdash; standard vs CNR</a></li>
    <li><a href="#eigvals">Eigenvalues</a></li>
    <li><a href="#longrun">Cointegrating relations</a></li>
    <li><a href="#risk">Implied risk premium</a></li>
    <li><a href="#diag">Residual diagnostics</a></li>
    <li><a href="#limit">Limit distributions</a></li>
    <li><a href="#cv">Critical values</a></li>
    <li><a href="#cite">Citation</a></li>
  </ol>
</div>

<!-- ============ 1. Intro ============ -->
<section class="block" id="intro">
  <h2><span class="step-num">1</span> Introduction</h2>
  <p>
    <code>fjohansen</code> is a Python package implementing the
    <strong>Johansen cointegration test with Fourier-type smooth nonlinear
    deterministic trends restricted to cointegrating relations</strong>, as
    developed by Kurita &amp; Shintani (2025, <em>Econometric Reviews</em>).
  </p>
  <p>
    The test extends the classical Johansen procedure by allowing a slow,
    smooth nonlinear movement &mdash; a finite Fourier expansion &mdash;
    <em>inside</em> the long-run equilibrium. This makes the test substantially
    more powerful when the equilibrium drifts gradually through cycles
    (a time-varying risk premium, an unobserved policy stance, a slow
    structural transition).
  </p>
  <p>
    The library also bundles the FGLS Wald test of
    <strong>Perron, Shintani &amp; Yabu (2021)</strong>, used as a
    frequency-selection pre-step.
  </p>
  <div class="callout callout-tip">
    <strong>What you will see below.</strong> A complete worked example on
    <em>real</em> US Treasury yields (monthly, 1990&ndash;2019, 7 maturities),
    from frequency selection all the way to the implied risk-premium
    decomposition, with every plot and table generated by the package.
  </div>
</section>

<!-- ============ 2. Install ============ -->
<section class="block" id="install">
  <h2><span class="step-num">2</span> Install</h2>

  <h3>From PyPI</h3>
  <pre><code>pip install fjohansen</code></pre>

  <h3>From GitHub (latest dev)</h3>
  <pre><code>git clone https://github.com/merwanroudane/fjohansen.git
cd fjohansen
pip install -e .</code></pre>

  <h3>Optional pretty-printing extras</h3>
  <pre><code>pip install "fjohansen[pretty]"   # rich + tabulate</code></pre>
</section>

<!-- ============ 3. Theory ============ -->
<section class="block" id="theory">
  <h2><span class="step-num">3</span> Theory in five short paragraphs</h2>

  <p>
    <strong>Cointegration.</strong> A vector of $p$ non-stationary $I(1)$
    series $X_t$ is said to be cointegrated of rank $r$ when there exist
    $r$ linear combinations $\\beta' X_t$ that are stationary &mdash; the
    long-run equilibria of the system. Johansen (1988, 1996) developed the
    likelihood-ratio "trace" test to select $r$ inside a Gaussian VAR.
  </p>

  <p>
    <strong>The problem.</strong> Standard Johansen models assume the
    equilibrium is constant up to a deterministic trend. When the long-run
    relation drifts slowly (a time-varying term premium, a moving real
    exchange rate target), the test loses power and often under-states
    $r$. The classical fix &mdash; allowing a structural break at a known
    date &mdash; is too rigid for smooth, repeated movements.
  </p>

  <p>
    <strong>Kurita &amp; Shintani (2025).</strong> Replace the constant
    equilibrium with a finite Fourier expansion
    $$F_{{t,T}} = [\\sin(2\\pi t/T),\\,\\cos(2\\pi t/T),\\ldots,
                  \\sin(2\\pi n t/T),\\,\\cos(2\\pi n t/T)]'$$
    restricted to the cointegrating space:
    $$\\Delta X_t = \\alpha (\\beta' X_{{t-1}} + \\delta' F_{{t,T}} + \\gamma')
                 + \\sum_{{i=1}}^{{k-1}} \\Gamma_i \\Delta X_{{t-i}} + \\varepsilon_t.$$
    The trace statistic $-T\\sum_{{i=r+1}}^{{p}} \\log(1-\\hat\\lambda_i)$ has a
    nuisance-parameter-free limiting distribution that depends only on
    $p-r$ and the number of frequencies $n$ (Proposition 3.1 of the paper).
  </p>

  <p>
    <strong>Computation.</strong> The reduced-rank regression is unchanged.
    The library
    <em>(i)</em> reads $\\hat\\lambda_i$ from
    $\\det(\\lambda S_{{11}} - S_{{10}} S_{{00}}^{{-1}} S_{{01}}) = 0$,
    <em>(ii)</em> looks up critical values in Table B1 of the paper for the
    common cells, falling back to a Gamma approximation (Doornik 1998)
    elsewhere, and
    <em>(iii)</em> reports the sequential rank choice using the standard
    Johansen rule.
  </p>

  <p>
    <strong>Picking $n$.</strong> The number of Fourier frequencies is chosen
    data-adaptively using the FGLS Wald test of Perron, Shintani &amp; Yabu
    (2021): for each series we test from $n = n_{{\\max}}$ down to $n = 1$
    until the highest-frequency coefficients become significant. The final
    panel-wide $n$ is the maximum across series.
  </p>
</section>

<!-- ============ 4. Data ============ -->
<section class="block" id="data">
  <h2><span class="step-num">4</span> Data &mdash; US Treasury yields</h2>
  <p>
    Seven constant-maturity Treasury yields fetched live from FRED, averaged
    to monthly frequency over <strong>{sample_start} &ndash; {sample_end}</strong>
    (T = {T}, p = {p}).
  </p>

  <div class="chip-row">
    <div class="chip"><div class="label">Observations T</div><div class="value">{T}</div><div class="sub">monthly</div></div>
    <div class="chip"><div class="label">Maturities p</div><div class="value">{p}</div><div class="sub">3m to 30y</div></div>
    <div class="chip"><div class="label">Sample start</div><div class="value">{sample_start}</div></div>
    <div class="chip"><div class="label">Sample end</div><div class="value">{sample_end}</div></div>
  </div>

  <div class="figure">
    <img src="{fig_yields["path"]}" alt="US Treasury yields">
    <div class="caption">Figure 1 &mdash; {fig_yields["caption"]}</div>
  </div>
</section>

<!-- ============ 5. PSY ============ -->
<section class="block" id="psy">
  <h2><span class="step-num">5</span> Frequency selection (Perron-Shintani-Yabu 2021)</h2>
  <p>
    The PSY (2021) FGLS Wald test &mdash; robust to both $I(0)$ and $I(1)$
    noise &mdash; is applied to each yield series via a general-to-specific
    algorithm with significance level $\\alpha = 0.10$.
  </p>

  {read(TBL / "psy_per_series.html")}

  <div class="callout callout-note">
    <strong>Multivariate result:</strong> the final number of Fourier
    frequencies used in the cointegrating-rank test is
    <strong>n = {n_selected}</strong>.
  </div>
</section>

<!-- ============ 6. Trace test compare ============ -->
<section class="block" id="compare">
  <h2><span class="step-num">6</span> Trace test &mdash; standard Johansen vs CNR</h2>
  <p>
    The standard test uses a constant restricted inside the cointegrating
    space (no Fourier). The CNR model adds the $\\delta' F_{{t,T}}$ block.
  </p>

  {read(TBL / "trace_comparison.html")}

  <div class="chip-row">
    <div class="chip">
      <div class="label">Standard rank</div>
      <div class="value">r = {res_std.selected_rank}</div>
      <div class="sub">{p - res_std.selected_rank} common stochastic trends</div>
    </div>
    <div class="chip">
      <div class="label">CNR rank (n = {n_selected})</div>
      <div class="value">r = {res.selected_rank}</div>
      <div class="sub">{p - res.selected_rank} common stochastic trends</div>
    </div>
  </div>

  <div class="callout">
    Term-structure theory implies one common stochastic trend
    ($p - r = 1$). The two specifications can disagree significantly when the
    underlying equilibrium drifts; this is exactly the
    <em>spurious stochastic trend</em> problem flagged by Kurita &amp; Shintani
    (2025).
  </div>
</section>

<!-- ============ 7. Eigenvalues ============ -->
<section class="block" id="eigvals">
  <h2><span class="step-num">7</span> Eigenvalues of the reduced-rank regression</h2>
  <div class="figure">
    <img src="{fig_eig["path"]}" alt="Eigenvalues">
    <div class="caption">Figure 2 &mdash; {fig_eig["caption"]}</div>
  </div>
</section>

<!-- ============ 8. Long-run ============ -->
<section class="block" id="longrun">
  <h2><span class="step-num">8</span> Estimated cointegrating relations</h2>
"""

if url_beta:
    HTML += f"  {read(TBL / 'beta_matrix.html')}\n"

if fig_lr:
    HTML += f"""
  <div class="figure">
    <img src="{fig_lr["path"]}" alt="Long-run relations">
    <div class="caption">Figure 3 &mdash; {fig_lr["caption"]}</div>
  </div>
"""

HTML += """
</section>

<!-- ============ 9. Risk premium ============ -->
<section class="block" id="risk">
  <h2><span class="step-num">9</span> Implied risk-premium decomposition</h2>
"""
if fig_rp:
    HTML += f"""
  <div class="figure">
    <img src="{fig_rp["path"]}" alt="Risk premium decomposition">
    <div class="caption">Figure 4 &mdash; {fig_rp["caption"]}</div>
  </div>
  <p>
    The thick line is the full implied risk-premium series
    $\\hat\\rho_t = -\\hat\\beta' X_{{t-1}} - \\hat\\delta' F_{{t,T}} - \\hat\\gamma$.
    The dashed line isolates the smooth nonlinear component
    $\\hat\\delta' F_{{t,T}}$ &mdash; the slow, cyclical part of the premium.
  </p>
"""
else:
    HTML += "<p>No Fourier component to decompose for this sample.</p>"

HTML += f"""
</section>

<!-- ============ 10. Diagnostics ============ -->
<section class="block" id="diag">
  <h2><span class="step-num">10</span> Residual diagnostics</h2>
  <div class="figure">
    <img src="{fig_diag["path"]}" alt="Residual diagnostics">
    <div class="caption">Figure 5 &mdash; {fig_diag["caption"]}</div>
  </div>
</section>

<!-- ============ 11. Limit distribution ============ -->
<section class="block" id="limit">
  <h2><span class="step-num">11</span> Limit distributions under the null</h2>
  <div class="figure">
    <img src="{fig_limit["path"]}" alt="Limit distributions">
    <div class="caption">Figure 6 &mdash; {fig_limit["caption"]}</div>
  </div>
</section>

<!-- ============ 12. Critical values ============ -->
<section class="block" id="cv">
  <h2><span class="step-num">12</span> Critical values (Table B1 of the paper)</h2>
  {read(TBL / "critical_values.html")}
</section>

<!-- ============ 13. Citation ============ -->
<section class="block" id="cite">
  <h2><span class="step-num">13</span> Citation</h2>
  <p>If you use <code>fjohansen</code> in academic work, please cite both
     the underlying paper and the package:</p>

  <div class="bibtex">@article{{kurita_shintani_2025,
  author  = {{Takamitsu Kurita and Mototsugu Shintani}},
  title   = {{Johansen test with Fourier-type smooth nonlinear trends in cointegrating relations}},
  journal = {{Econometric Reviews}},
  year    = {{2025}},
  volume  = {{44}},
  number  = {{10}},
  pages   = {{1589--1616}},
  doi     = {{10.1080/07474938.2025.2530640}}
}}

@article{{perron_shintani_yabu_2017,
  author  = {{Pierre Perron and Mototsugu Shintani and Tomoyoshi Yabu}},
  title   = {{Testing for flexible nonlinear trends with an integrated or stationary noise component}},
  journal = {{Oxford Bulletin of Economics and Statistics}},
  volume  = {{79}}, pages = {{822--850}}, year = {{2017}},
  doi     = {{10.1111/obes.12169}}
}}

@software{{roudane_fjohansen,
  author  = {{Merwan Roudane}},
  title   = {{fjohansen: Johansen test with Fourier-type smooth nonlinear trends (Python)}},
  year    = {{2025}},
  url     = {{https://github.com/merwanroudane/fjohansen}}
}}</div>
</section>

</main>

<footer>
  <p>
    Built with <span class="heart">&#9829;</span> by
    <strong><a href="https://github.com/merwanroudane">Merwan Roudane</a></strong> &middot;
    Page generated on {build_date} &middot;
    <a href="https://pypi.org/project/fjohansen/">PyPI</a> &middot;
    <a href="https://github.com/merwanroudane/fjohansen">Source</a> &middot;
    <a href="https://github.com/merwanroudane/fjohansen/blob/main/examples/fjohansen_tutorial.ipynb">Notebook</a> &middot;
    <a href="https://doi.org/10.1080/07474938.2025.2530640">Paper</a>
  </p>
  <p style="font-size:0.85em;">&copy; 2025 Merwan Roudane. Released under the MIT License.</p>
</footer>

</body>
</html>
"""

(DOCS / "index.html").write_text(HTML, encoding="utf-8")
print(f"        Wrote {DOCS / 'index.html'}")

# A small Jekyll config so GitHub Pages does not try to process the site.
(DOCS / "_config.yml").write_text(
    textwrap.dedent("""\
        # GitHub Pages config -- bypass Jekyll, serve the static index.html.
        plugins: []
        include: ['_*']
    """),
    encoding="utf-8",
)
# Disable Jekyll entirely:
(DOCS / ".nojekyll").write_text("", encoding="utf-8")

print()
print("Done. Site root:", DOCS)
print("Open docs/index.html in a browser, or push docs/ to GitHub Pages.")
