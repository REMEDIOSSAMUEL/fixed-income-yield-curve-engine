# Fixed-Income Yield Curve & Relative-Value Engine

A Python 3.11+ project for explicit fixed-income mathematics, yield-curve
modelling, interest-rate risk, historical factor analysis, and relative-value
research.

The repository is currently at the architecture and scaffolding stage. The
financial analytics described in [PROJECT_SPEC.md](PROJECT_SPEC.md) have not
yet been implemented, and this README makes no empirical or performance
claims.

## Planned capabilities

- Fixed-rate bond cash flows, accrued interest, clean and dirty pricing
- Yield-to-maturity, duration, convexity, DV01, and key-rate risk
- Zero-curve discounting and Nelson-Siegel-Svensson fitting
- Historical Treasury-yield analysis and principal component analysis
- Curve residuals, rolling z-scores, butterfly analytics, and hedging
- A historical relative-value backtest with explicit timing and costs
- Reproducible charts and tables written to `outputs/`

## Installation

Create and activate a virtual environment, then install the package and its
development tools:

```bash
python -m pip install -e ".[dev]"
```

## Command-line interface

The root script currently provides command placeholders only:

```bash
python fixed_income_engine.py demo --offline
python fixed_income_engine.py curve
python fixed_income_engine.py bond
python fixed_income_engine.py risk
python fixed_income_engine.py pca
python fixed_income_engine.py relative-value
python fixed_income_engine.py backtest
```

## Development checks

```bash
pytest
ruff check .
ruff format --check .
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the planned mathematics,
conventions, numerical considerations, and validation strategy.
