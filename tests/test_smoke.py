"""Smoke tests for the package and command-line application."""

import os
import subprocess
import sys
from pathlib import Path

import fixed_income


def test_package_imports() -> None:
    """The package exposes its initial version metadata."""
    assert fixed_income.__version__ == "0.1.0"


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


def test_offline_demo_runs_end_to_end() -> None:
    """The offline demo runs without a network call and writes useful artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "fixed_income_engine.py", "demo", "--offline"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, result.stderr
    assert "Latest observed Treasury CMT curve" in result.stdout
    assert "Fit RMSE" in result.stdout
    assert "Representative fixed-rate bond" in result.stdout
    assert "Historical yield-change PCA" in result.stdout
    assert "Current relative value" in result.stdout
    for filename in (
        "yield_curve.png",
        "pca_loadings.png",
        "bond_risk_report.csv",
        "key_rate_dv01.csv",
        "relative_value_report.csv",
    ):
        artifact = project_root / "outputs" / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
