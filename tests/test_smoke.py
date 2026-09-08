"""Smoke tests for the initial project scaffold."""

import subprocess
import sys
from pathlib import Path

import fixed_income


def test_package_imports() -> None:
    """The package exposes its initial version metadata."""
    assert fixed_income.__version__ == "0.1.0"


def test_offline_demo_placeholder_runs() -> None:
    """The offline CLI placeholder parses and exits successfully."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "fixed_income_engine.py", "demo", "--offline"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "not implemented yet" in result.stdout
