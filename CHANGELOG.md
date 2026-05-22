# Changelog

All notable changes to **fjohansen** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-22

### Added
- Initial public release.
- Johansen cointegration test with Fourier-type smooth nonlinear trends
  restricted to cointegrating relations (Kurita & Shintani 2025).
- Six model variants: `CNR`, `LNR`, `CNU`, `LNU`, `constant`, `linear`.
- Bundled Table B1 critical values (CNR) and pre-computed moment tables
  for the other model variants.
- Vectorised limit-distribution simulator with persistent disk cache.
- Gamma-approximation p-values (Doornik 1998).
- Perron-Shintani-Yabu (2021) FGLS Wald test with Prais-Winsten
  transformation, Roy-Fuller bias correction and super-efficient AR(1)
  estimator.
- General-to-specific frequency selection for univariate and multivariate
  panels.
- Publication-quality output: ASCII summary, LaTeX (booktabs), HTML, and
  Rich-coloured terminal tables.
- Paper-style figures: eigenvalues, long-run relations, recursive
  rejection curves, limit-distribution densities, residual diagnostics,
  risk-premium decomposition.
- Section 5 DGPs (NF-DGP-1..4, F-DGP-1, F-DGP-2) and a synthetic JGB-yield
  surrogate.
- Three runnable example scripts reproducing Sections 5, 6 and the
  limit-distribution figures of the paper.
- Test suite (13 tests).
