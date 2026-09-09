# Fixed-Income Yield Curve & Relative-Value Engine

A Python 3.11+ project for explicit fixed-income mathematics, yield-curve
modelling, interest-rate risk, historical factor analysis, and relative-value
research.

The repository is an educational research implementation. Core mathematics,
data validation, PCA, relative-value analysis and the sensitivity backtest are
implemented with explicit conventions. This README makes no performance claims.

## Current capabilities

- Fixed-rate bond cash flows, accrued interest, clean and dirty pricing
- Yield-to-maturity, duration, convexity, and DV01
- Tagged curve representations, zero-rate interpolation, and explicit
  continuous, periodic, or simple-compounding discount factors
- Spot-curve bond pricing, parallel and shaped full-revaluation scenarios, and
  localized 2Y/5Y/10Y/30Y key-rate DV01 reports
- Explicit Nelson-Siegel-Svensson evaluation and bounded multi-start
  calibration with diagnostics
- Offline historical Treasury constant-maturity yields and an optional official
  FRED/H.15 online adapter with a documented fallback
- Historical Treasury-yield PCA
- Curve residuals, rolling z-scores, butterfly analytics, and hedging
- A historical 5Y NSS-residual research backtest with lagged signals,
  time-varying DV01-neutral hedge weights, and cost sensitivity
- Reproducible charts and tables written to `outputs/`

## Yield-curve data and semantics

`fixed_income/sample_data/sample_treasury_yields.csv` contains 124 complete daily observations from
2 January through 28 June 2024 for the 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y,
and 30Y Treasury constant-maturity series. The values are a bundled snapshot of
official Federal Reserve H.15 series retrieved through FRED and remain in their
published percent units in the CSV. `fixed_income.data` converts them once to
decimal annual rates when loading.

Treasury constant-maturity yields are par-yield-like statistical observations.
They are not labelled or used as bootstrapped zero rates. NSS fitting produces
a smooth fit retaining the meaning of its input series. Discounting requires a
separately constructed curve explicitly tagged as zero rates, together with an
explicit compounding convention.

Missing observations are never filled implicitly. Callers select `raise`,
`drop`, or `keep`; the online loader defaults to complete-case rows and can
fall back to the bundled sample when connectivity is unavailable.

## Installation

Create and activate a virtual environment, then install the package and its
development tools:

```bash
python -m pip install -e ".[dev]"
```

## Command-line interface

The root script provides composable workflows. Add `--offline` to every
market-data command to guarantee use of the bundled sample:

```bash
python fixed_income_engine.py demo --offline
python fixed_income_engine.py curve --offline
python fixed_income_engine.py bond
python fixed_income_engine.py risk --offline
python fixed_income_engine.py pca --offline
python fixed_income_engine.py relative-value --offline
python fixed_income_engine.py backtest --offline
python fixed_income_engine.py backtest --offline --lookback 40 --entry-z 1.75 \
  --exit-z 0.25 --transaction-cost-bp 0.02
```

The backtest is a first-order yield-change P&L study on non-tradable Treasury
constant-maturity observations. It is not a percentage-return series, a
deployable strategy, or evidence of live profitability. It omits true security
prices, cash/futures bid-ask data, carry, roll, convexity, financing, funding,
and market impact; its configurable bp-of-face transaction costs are only a
simple sensitivity assumption.

At observation date t, the current residual is standardized with preceding
observations only (`ddof=1` by default). Positive z means above its trailing
mean; this can differ from the raw residual's rich/cheap classification. The
target is held over t to t+1, and P&L at t uses prior-date notionals and DV01s.
Costs are charged at t on absolute changes in face, including reversals, daily
hedge rebalancing and final liquidation. A long exits at z <= exit_z and a short
at z >= -exit_z, including jumps across the exit band; a jump to the opposite
entry threshold reverses directly. Missing signals close the target after that
date's held-interval P&L.

Each date receives an independent bounded multistart NSS fit. A low RMSE does
not establish stable betas/taus; conditioning and active bounds are reported.
No strategy parameters are optimized against P&L. Annualization uses 252 equally
weighted observation intervals, including warm-up zeros, with sample P&L
volatility. The P&L mean/volatility ratio is not an excess-return Sharpe and is
not adjusted for serial correlation. Complete-case gaps, later data revisions,
publication delays and hypothetical same-observation execution remain material
limitations.

`demo --offline` fits the latest CMT cross-section with NSS, reports a
representative bond's YTM analytics, runs full-revaluation curve shocks and
key-rate DV01, estimates historical yield-change PCA, ranks current fitted-curve
residuals, and constructs a DV01-neutral 2Y/5Y/10Y butterfly. Curve-risk output
clearly labels its direct CMT-to-zero-rate proxy as an illustrative assumption,
not a bootstrap. This workflow explicitly uses flat endpoint zero rates for
coupon times outside the node range, including payments before 3M. The generic
curve-pricing API rejects extrapolation by default. The demo finishes with a concise historical relative-value
backtest summary and writes `outputs/rv_backtest.csv` and
`outputs/rv_backtest.png`.

Additional artifacts include `nss_calibration_diagnostics.csv`,
`rv_backtest_metrics.csv`, and `rv_backtest_metadata.json`, with fit failures,
configuration, provenance, missing dates and methodological limitations.

Bond calculations use regular annual or semiannual schedules, maturity-anchored
month-end rolls, Actual/Actual coupon-period accrual and fractional-period nominal
YTM compounding throughout. Settlement-date payments are excluded. There are no
stubs, calendars, business-day adjustments, settlement lags or ex-coupon rules.
This is a generic convention, not a complete implementation of all U.S. Treasury
security pricing rules (including final-period special conventions).

Spot pricing uses the same coupon-period time basis, explicit compounding and
linear interpolation of zero rates. Linear zeros are a documented choice
differing from the specification's initial preference for log discount factors;
they do not guarantee monotone discount factors or non-negative forwards.
Key-rate hats partition a parallel shock, but finite-bump risk reconciliation
is only approximate, with second-order truncation error in the bump size.

## Development checks

```bash
pytest -q
pytest --cov=fixed_income
ruff check .
ruff format --check .
python fixed_income_engine.py demo --offline
python fixed_income_engine.py backtest --offline
python -m compileall fixed_income fixed_income_engine.py
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the planned mathematics,
conventions, numerical considerations, and validation strategy.
See [docs/AUDIT.md](docs/AUDIT.md) for the independent audit, corrections,
verification evidence and remaining limitations.
