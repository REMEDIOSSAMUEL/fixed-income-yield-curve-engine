"""Deterministic yield-curve plotting with no embedded financial formulas."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fixed_income.curves import (
    CurveRepresentation,
    NSSCalibrationResult,
    nss_yields,
)
from fixed_income.pca import PCAResult, interpret_pca_factors


def plot_observed_and_fitted_curve(
    observed_maturities_years: Sequence[float] | np.ndarray,
    observed_yields: Sequence[float] | np.ndarray,
    fitted_maturities_years: Sequence[float] | np.ndarray,
    fitted_yields: Sequence[float] | np.ndarray,
    *,
    yield_unit: str = "percent",
    title: str = "Observed and fitted yield curve",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot observed points and an already evaluated fitted curve.

    Args:
        observed_maturities_years: Observed maturities in years.
        observed_yields: Observed decimal annual yields.
        fitted_maturities_years: Fitted-curve maturities in years.
        fitted_yields: Fitted decimal annual yields.
        yield_unit: Display unit, either ``"percent"`` or ``"decimal"``.
            This affects labels and display scaling only.
        title: Figure title. Callers should identify the fitted input type when
            the generic default would be ambiguous.
        ax: Optional Matplotlib axes on which to draw.

    Returns:
        The Matplotlib figure and axes. No file is written automatically.

    Notes:
        All financial modelling is performed before this function is called.
        The function does not reinterpret observed yields as zero rates.
    """
    observed_maturities = _plot_vector(
        observed_maturities_years, "observed_maturities_years"
    )
    observations = _plot_vector(observed_yields, "observed_yields")
    fitted_maturities = _plot_vector(fitted_maturities_years, "fitted_maturities_years")
    fit = _plot_vector(fitted_yields, "fitted_yields")
    if observed_maturities.size != observations.size:
        raise ValueError("observed maturities and yields must have equal length")
    if fitted_maturities.size != fit.size:
        raise ValueError("fitted maturities and yields must have equal length")
    if np.any(observed_maturities < 0.0) or np.any(fitted_maturities < 0.0):
        raise ValueError("plot maturities must be non-negative years")
    if yield_unit == "percent":
        scale = 100.0
        y_label = "Annual yield (%)"
    elif yield_unit == "decimal":
        scale = 1.0
        y_label = "Annual yield (decimal)"
    else:
        raise ValueError("yield_unit must be 'percent' or 'decimal'")

    if ax is None:
        figure, axes = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    else:
        axes = ax
        figure = axes.figure
    axes.scatter(
        observed_maturities,
        observations * scale,
        color="tab:blue",
        marker="o",
        label="Observed maturities",
        zorder=3,
    )
    axes.plot(
        fitted_maturities,
        fit * scale,
        color="tab:orange",
        linewidth=2.0,
        label="Fitted NSS curve",
    )
    axes.set_xlabel("Maturity (years)")
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.grid(visible=True, alpha=0.25)
    axes.legend()
    return figure, axes


def plot_nss_curve(
    calibration: NSSCalibrationResult,
    *,
    maturity_grid_years: Sequence[float] | np.ndarray | None = None,
    yield_unit: str = "percent",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot observed decimal yields and their fitted NSS curve over years.

    Args:
        calibration: Structured NSS result whose yields are decimal annual
            rates and maturities are years.
        maturity_grid_years: Optional non-negative plotting grid in years.
            Defaults to 250 points spanning zero through the longest observed
            maturity.
        yield_unit: Display as ``"percent"`` or internal ``"decimal"`` units.
        ax: Optional existing Matplotlib axes.

    Returns:
        The Matplotlib figure and axes. Model evaluation delegates to
        :func:`fixed_income.curves.nss_yields`; no curve mathematics is
        implemented in this plotting module.
    """
    if not isinstance(calibration, NSSCalibrationResult):
        raise TypeError("calibration must be an NSSCalibrationResult")
    if maturity_grid_years is None:
        grid = np.linspace(0.0, float(calibration.maturities_years[-1]), 250)
    else:
        grid = _plot_vector(maturity_grid_years, "maturity_grid_years")
        if np.any(grid < 0.0):
            raise ValueError("maturity_grid_years must be non-negative")
    fitted = nss_yields(grid, calibration.parameters)
    title = _fit_title(calibration.input_representation)
    return plot_observed_and_fitted_curve(
        calibration.maturities_years,
        calibration.observed_yields,
        grid,
        fitted,
        yield_unit=yield_unit,
        title=title,
        ax=ax,
    )


def plot_pca_loadings(
    result: PCAResult,
    *,
    n_components: int = 3,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot leading dimensionless PCA loadings against maturity in years.

    Args:
        result: Historical yield-change PCA result. Its input yields and changes
            use decimal annual rates; plotted loadings are dimensionless.
        n_components: Number of leading components to draw, from one through
            three and no more than the number of available maturities.
        ax: Optional Matplotlib axes on which to draw.

    Returns:
        The Matplotlib figure and axes. Suggested level/slope/curvature labels
        are displayed only when the quantitative shape diagnostic supports
        them. No file is written automatically; a later CLI workflow can call
        :func:`save_figure` with ``outputs/pca_loadings.png``.
    """
    if not isinstance(result, PCAResult):
        raise TypeError("result must be a PCAResult")
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("n_components must be an integer")
    available = result.loadings.shape[1]
    if n_components < 1 or n_components > min(3, available):
        raise ValueError(f"n_components must be between one and {min(3, available)}")
    if ax is None:
        figure, axes = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    else:
        axes = ax
        figure = axes.figure

    diagnostics = (
        interpret_pca_factors(result, max_components=n_components)
        if len(result.maturities_years) >= 3
        else None
    )
    for component in range(n_components):
        label = f"PC{component + 1}"
        if (
            diagnostics is not None
            and diagnostics[component].suggested_label is not None
        ):
            label += f" ({diagnostics[component].suggested_label})"
        axes.plot(
            result.maturities_years,
            result.loadings[:, component],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes.set_xlabel("Maturity (years)")
    axes.set_ylabel("Loading (dimensionless)")
    axes.set_title("Historical yield-change PCA loadings")
    axes.grid(visible=True, alpha=0.25)
    axes.legend()
    return figure, axes


def save_figure(figure: Figure, output_path: str | Path) -> Path:
    """Save a figure below ``outputs/`` and return its resolved filesystem path.

    ``output_path`` is a repository-relative or absolute image path; generated
    output outside the repository's ``outputs`` directory is rejected. Parent
    directories below ``outputs`` are created as needed.
    """
    if not isinstance(figure, Figure):
        raise TypeError("figure must be a matplotlib Figure")
    outputs_root = (Path(__file__).resolve().parents[1] / "outputs").resolve()
    requested = Path(output_path)
    if not requested.is_absolute():
        requested = Path(__file__).resolve().parents[1] / requested
    resolved = requested.resolve()
    if outputs_root not in resolved.parents:
        raise ValueError("generated figures must be saved below outputs/")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved, dpi=150, bbox_inches="tight")
    return resolved


def _plot_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric one-dimensional sequence") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _fit_title(representation: CurveRepresentation) -> str:
    if representation is CurveRepresentation.TREASURY_CMT:
        return "Treasury constant-maturity yields with fitted NSS curve"
    if representation is CurveRepresentation.PAR_YIELD:
        return "Par yields with fitted NSS curve"
    return "Zero rates with fitted NSS curve"
