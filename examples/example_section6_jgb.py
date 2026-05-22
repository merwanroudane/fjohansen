"""
Replicates the empirical application of Section 6 of Kurita & Shintani (2025)
using a synthetic JGB-style dataset bundled with the package.

Steps:
    1. Load the data and plot it (Fig. 11).
    2. Run PSY (2021) general-to-specific selection of the Fourier frequencies.
    3. Estimate the CNR model with the selected n.
    4. Print the Table-3-style summary and export LaTeX/HTML.
    5. Produce the eigenvalue plot, long-run plot, residual diagnostics
       (Fig. 12) and implied risk premium (Fig. 13).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import fjohansen as fj


def main() -> None:
    # 1. Data
    data = fj.sample_jgb_data(T=108)
    fj.plot_series(data, title="Japanese yields (synthetic surrogate of Fig. 11)")
    plt.show()

    # 2. Frequency selection
    sel = fj.select_frequencies(data, n_max=5, p_d=1, sig_level=0.10)
    print("Per-series selected n:")
    print(sel.per_series.to_string(index=False))
    print(f"\nFinal n_selected = {sel.n_selected}")

    # 3. Estimate CNR model with k = 3 (as in the paper)
    res = fj.JohansenFourier(
        data, k=3, n=max(sel.n_selected, 1), model="CNR"
    ).fit()

    # 4. Output
    print("\n" + res.summary())
    fj.rich_print(res)

    with open("section6_table.tex", "w", encoding="utf-8") as fh:
        fh.write(res.to_latex(caption="JGB Johansen-Fourier (CNR) test"))
    with open("section6_table.html", "w", encoding="utf-8") as fh:
        fh.write(res.to_html())
    print("\nWrote section6_table.tex and section6_table.html.")

    # 5. Plots
    res.plot_eigenvalues()
    res.plot_long_run()
    res.plot_residual_diagnostics()
    if res.delta is not None and res.n > 0 and res.selected_rank:
        res.plot_risk_premium(index=data.index)
    plt.show()


if __name__ == "__main__":
    main()
