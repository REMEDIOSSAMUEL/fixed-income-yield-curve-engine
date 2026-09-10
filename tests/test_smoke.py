"""Smoke tests for the package and command-line application."""

import os
import subprocess
import sys
from pathlib import Path

import fixed_income


def test_package_imports() -> None:
    """The package version agrees with the release metadata."""
    import tomllib

    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert fixed_income.__version__ == metadata["project"]["version"] == "1.0.0"


def test_cli_help_runs() -> None:
    """The root help is discoverable and lists the implemented workflows."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "fixed_income_engine.py", "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "relative-value" in result.stdout
    assert "backtest" in result.stdout


def test_offline_backtest_cli_reports_metrics_and_writes_artifacts() -> None:
    """The configurable offline CLI reports metrics and saves outputs."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "fixed_income_engine.py",
            "backtest",
            "--offline",
            "--lookback",
            "40",
            "--entry-z",
            "1.75",
            "--exit-z",
            "0.25",
            "--transaction-cost-bp",
            "0.02",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        # Each day now uses a full multistart calibration; coverage also traces
        # this subprocess. Keep a bounded runtime without relaxing math checks.
        timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, result.stderr
    for label in (
        "Sample date range:",
        "Observations:",
        "Methodology summary:",
        "Gross cumulative P&L:",
        "Net cumulative P&L:",
        "Annualised volatility (gross / net):",
        "Sharpe on approximate P&L (gross / net):",
        "Maximum drawdown (gross / net):",
        "Turnover:",
        "Hit rate on active intervals (gross / net):",
        "Number of trades (entries):",
        "Transaction-cost assumption:",
    ):
        assert label in result.stdout
    assert "Transaction-cost assumption: 0.02 bp" in result.stdout
    for filename in ("rv_backtest.csv", "rv_backtest.png"):
        artifact = project_root / "outputs" / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 0


def test_offline_demo_runs_end_to_end() -> None:
    """The offline demo runs without a network call and writes useful artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "fixed_income_engine.py", "demo", "--offline"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, result.stderr
    assert "Latest observed Treasury CMT curve" in result.stdout
    assert "Fit RMSE" in result.stdout
    assert "Representative fixed-rate bond" in result.stdout
    assert "Historical yield-change PCA" in result.stdout
    assert "Current relative value" in result.stdout
    assert "Historical relative-value research backtest" in result.stdout
    for filename in (
        "yield_curve.png",
        "pca_loadings.png",
        "bond_risk_report.csv",
        "key_rate_dv01.csv",
        "relative_value_report.csv",
        "rv_backtest.csv",
        "rv_backtest.png",
    ):
        artifact = project_root / "outputs" / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
