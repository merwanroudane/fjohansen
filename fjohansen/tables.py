"""
Publication-quality table formatters.

Three output formats are supported:

* Plain-text console output (with optional `rich` styling) -- looks like
  Table 3 of Kurita & Shintani (2025).
* LaTeX booktabs source -- ready to drop into a journal manuscript.
* HTML -- styled with light shading for use in notebooks or reports.

Significance stars follow the paper:

    *** : p < 0.01
    **  : p < 0.05
    *   : p < 0.10
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import ModelSpec

__all__ = [
    "format_header",
    "format_trace_table",
    "format_trace_latex",
    "format_trace_html",
    "rich_print",
]


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return " "


def format_header(results) -> str:
    """Build the first block of the summary (sample, model, lags, ...)."""
    lines = [
        "=" * 72,
        "  Johansen-Fourier Cointegration Test (Kurita & Shintani, 2025)",
        "=" * 72,
        f"  Model               : {results.spec.code}  ({results.spec.label})",
        f"  Variables (p)       : {results.p}",
        f"  Lag order (k)       : {results.k}",
        f"  Fourier freqs (n)   : {results.n}",
        f"  Effective sample    : {results.T_eff}",
        "-" * 72,
    ]
    return "\n".join(lines)


def _format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "   .   "
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_trace_table(df: pd.DataFrame, sig_level: float = 0.05) -> str:
    """ASCII table for the trace test."""
    header = (
        f"  {'H0':10s} {'eigenvalue':>12s} {'trace stat':>12s} "
        f"{'p-value':>10s}  {'5% c.v.':>8s} {'1% c.v.':>8s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for _, row in df.iterrows():
        stars = _stars(row["p_value"])
        lines.append(
            f"  {row['null']:10s} "
            f"{row['eigenvalue']:12.5f} "
            f"{row['trace_stat']:12.3f} "
            f"{_format_pvalue(row['p_value']):>8s}{stars:<2s} "
            f"{row['cv_95']:8.2f} "
            f"{row['cv_99']:8.2f}"
        )
    lines.append(sep)
    lines.append("  Stars: ** p<0.01,  * p<0.05,  . p<0.10")
    return "\n".join(lines)


def format_trace_latex(
    df: pd.DataFrame,
    *,
    spec: ModelSpec,
    n: int,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    sig_level: float = 0.05,
) -> str:
    """Return a booktabs LaTeX table, journal style."""
    cap = caption or (
        f"Johansen-Fourier trace test ({spec.code} model, n = {n} frequencies)."
    )
    lab = label or f"tab:johansen_fourier_{spec.code.lower()}_n{n}"
    out = []
    out.append(r"\begin{table}[!htbp]")
    out.append(r"\centering")
    out.append(r"\caption{" + cap + "}")
    out.append(r"\label{" + lab + "}")
    out.append(r"\begin{tabular}{lrrrrr}")
    out.append(r"\toprule")
    out.append(
        r"$H_0$ & $\hat\lambda$ & $-2\log LR$ & $p$-value & "
        r"5\% c.v. & 1\% c.v. \\"
    )
    out.append(r"\midrule")
    for _, row in df.iterrows():
        pstr = _format_pvalue(row["p_value"])
        stars = {".": r"$^{\dagger}$", "*": r"$^{*}$", "**": r"$^{**}$", " ": ""}[_stars(row["p_value"])]
        h0 = row["null"].replace("<=", r"\leq")
        out.append(
            f"$ {h0} $ & {row['eigenvalue']:.5f} & "
            f"{row['trace_stat']:.3f}{stars} & "
            f"[{pstr}] & {row['cv_95']:.2f} & {row['cv_99']:.2f} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(
        r"\\[2pt]\footnotesize Note: $p$-values from Gamma-approximation; "
        r"$^{*} p<0.05$, $^{**} p<0.01$, $^{\dagger} p<0.10$."
    )
    out.append(r"\end{table}")
    return "\n".join(out)


def format_trace_html(
    df: pd.DataFrame,
    *,
    spec: ModelSpec,
    n: int,
    caption: Optional[str] = None,
) -> str:
    """Return a styled HTML table for notebooks / web reports."""
    cap = caption or (
        f"Johansen-Fourier trace test &mdash; <b>{spec.code}</b> model, "
        f"n = {n} frequencies."
    )
    css = """
    <style>
      table.fjoh { border-collapse: collapse; font-family: 'Georgia',serif;
                   font-size: 11pt; margin: 14px auto; }
      table.fjoh th { background: #f5f5f5; border-bottom: 1.5pt solid #222;
                       padding: 6px 14px; text-align: center; }
      table.fjoh td { border-bottom: 0.5pt solid #ddd;
                       padding: 5px 14px; text-align: right; }
      table.fjoh td.h0 { text-align: left; font-style: italic; }
      table.fjoh tr:last-child td { border-bottom: 1.5pt solid #222; }
      table.fjoh caption { caption-side: top; padding: 6px; font-size: 11pt; }
      .star { color: #b00020; font-weight: bold; }
    </style>
    """
    rows_html = []
    for _, row in df.iterrows():
        pstr = _format_pvalue(row["p_value"])
        stars = _stars(row["p_value"]).strip()
        rows_html.append(
            "<tr>"
            f"<td class='h0'>{row['null']}</td>"
            f"<td>{row['eigenvalue']:.5f}</td>"
            f"<td>{row['trace_stat']:.3f}</td>"
            f"<td>[{pstr}] <span class='star'>{stars}</span></td>"
            f"<td>{row['cv_95']:.2f}</td>"
            f"<td>{row['cv_99']:.2f}</td>"
            "</tr>"
        )
    body = "\n".join(rows_html)
    return css + (
        f"<table class='fjoh'>"
        f"<caption>{cap}</caption>"
        "<thead><tr>"
        "<th>H&#8320;</th><th>&lambda;&#770;</th>"
        "<th>&minus;2 log&nbsp;LR</th><th>p-value</th>"
        "<th>5% c.v.</th><th>1% c.v.</th>"
        "</tr></thead><tbody>"
        f"{body}"
        "</tbody></table>"
        "<p style='font-size:10pt; text-align:center;'>"
        "&#42;&#42; p&lt;0.01, &#42; p&lt;0.05, &#46; p&lt;0.10."
        "</p>"
    )


def rich_print(results) -> None:
    """Render a colourful Rich table to the console if ``rich`` is installed."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:                              # pragma: no cover
        print(results.summary())
        return
    console = Console()
    spec = results.spec
    console.rule(
        f"[bold]Johansen-Fourier Test[/bold] | "
        f"model={spec.code} | k={results.k} | n={results.n} | "
        f"p={results.p} | T={results.T_eff}"
    )
    tbl = Table(box=box.SIMPLE_HEAVY, header_style="bold", show_lines=False)
    tbl.add_column("H0", justify="left", style="italic")
    tbl.add_column("eigenvalue", justify="right")
    tbl.add_column("trace stat", justify="right")
    tbl.add_column("p-value", justify="right")
    tbl.add_column("5% c.v.", justify="right")
    tbl.add_column("1% c.v.", justify="right")
    for _, row in results.trace_stats.iterrows():
        stars = _stars(row["p_value"]).strip()
        pv = _format_pvalue(row["p_value"])
        sty = "red" if row["p_value"] < 0.05 else ""
        tbl.add_row(
            row["null"],
            f"{row['eigenvalue']:.5f}",
            f"{row['trace_stat']:.3f}",
            f"[{sty}]{pv}{stars}[/]",
            f"{row['cv_95']:.2f}",
            f"{row['cv_99']:.2f}",
        )
    console.print(tbl)
    if results.selected_rank is not None:
        console.print(
            f"[bold green]Selected rank:[/] r = {results.selected_rank}  "
            f"(=> {results.p - results.selected_rank} common stochastic trends)"
        )
