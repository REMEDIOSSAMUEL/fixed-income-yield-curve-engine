# Fixed-Income Yield Curve & Relative-Value Engine

A Python 3.11+ project for explicit fixed-income mathematics, yield-curve
modelling, interest-rate risk, historical factor analysis, and relative-value
research.

The repository is under active implementation. It currently includes explicit
fixed-rate bond mathematics and a typed yield-curve modelling/data layer. This
README makes no empirical or performance claims.

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
- Reproducible charts and tables written to `outputs/`

## Planned capabilities

- A historical relative-value backtest with explicit timing and costs

## Yield-curve data and semantics

`data/sample_treasury_yields.csv` contains 124 complete daily observations from
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
```

`demo --offline` fits the latest CMT cross-section with NSS, reports a
representative bond's YTM analytics, runs full-revaluation curve shocks and
key-rate DV01, estimates historical yield-change PCA, ranks current fitted-curve
residuals, and constructs a DV01-neutral 2Y/5Y/10Y butterfly. Curve-risk output
clearly labels its direct CMT-to-zero-rate proxy as an illustrative assumption,
not a bootstrap.

## Development checks

```bash
pytest
ruff check .
ruff format --check .
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the planned mathematics,
conventions, numerical considerations, and validation strategy.
