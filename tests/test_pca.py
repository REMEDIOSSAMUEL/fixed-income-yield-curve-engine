"""Synthetic offline tests for historical yield-change PCA."""

import numpy as np
import pandas as pd
import pytest

from fixed_income.pca import (
    BASIS_POINT_DECIMAL,
    PCAMethod,
    calculate_yield_changes,
    fit_yield_change_pca,
    interpret_loading_shapes,
    interpret_pca_factors,
    normalize_loading_signs,
)


@pytest.fixture
def synthetic_yield_levels() -> pd.DataFrame:
    """Build seeded decimal yield levels driven by three curve factors."""
    random = np.random.default_rng(20260908)
    labels = np.array(["3M", "2Y", "5Y", "10Y", "20Y", "30Y"])
    maturity_count = len(labels)
    ranks = np.linspace(-1.0, 1.0, maturity_count)
    candidates = np.column_stack((np.ones(maturity_count), ranks, -(ranks**2)))
    shapes, _ = np.linalg.qr(candidates)
    factor_changes_bp = random.normal(scale=np.array([4.0, 2.0, 0.8]), size=(400, 3))
    noise_bp = random.normal(scale=0.05, size=(400, maturity_count))
    changes_decimal = (factor_changes_bp @ shapes.T + noise_bp) * BASIS_POINT_DECIMAL
    base_curve = np.array([0.035, 0.036, 0.037, 0.039, 0.041, 0.042])
    levels = np.vstack((base_curve, base_curve + np.cumsum(changes_decimal, axis=0)))
    dates = pd.bdate_range("2024-01-02", periods=len(levels))
    ordered = pd.DataFrame(levels, index=dates, columns=labels)
    return ordered.loc[:, ["10Y", "3M", "30Y", "2Y", "20Y", "5Y"]]


def test_pca_dimensions_ordering_and_eigen_properties(
    synthetic_yield_levels: pd.DataFrame,
) -> None:
    """Outputs have full dimensions, ordered tenors, and valid eigenpairs."""
    result = fit_yield_change_pca(synthetic_yield_levels)

    assert result.yield_changes_decimal.shape == (400, 6)
    assert result.factor_scores.shape == (400, 6)
    assert result.loadings.shape == (6, 6)
    assert result.eigenvalues.shape == (6,)
    assert result.maturity_labels == ("3M", "2Y", "5Y", "10Y", "20Y", "30Y")
    np.testing.assert_allclose(
        result.maturities_years, [0.25, 2.0, 5.0, 10.0, 20.0, 30.0]
    )
    assert np.all(np.diff(result.eigenvalues) <= 0.0)
    np.testing.assert_allclose(
        result.loadings.T @ result.loadings, np.eye(6), atol=1e-12
    )
    assert result.explained_variance_ratios.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["covariance", "correlation"])
def test_full_factor_reconstruction(
    synthetic_yield_levels: pd.DataFrame, method: str
) -> None:
    """All factors reconstruct complete decimal changes under both methods."""
    result = fit_yield_change_pca(synthetic_yield_levels, method=method)

    pd.testing.assert_frame_equal(
        result.reconstruct(), result.yield_changes_decimal, atol=1e-15, rtol=1e-12
    )


def test_sign_convention_is_deterministic_and_shape_specific() -> None:
    """Level, slope, and curvature columns receive documented orientations."""
    ranks = np.linspace(-1.0, 1.0, 7)
    candidates = np.column_stack((np.ones(7), ranks, -(ranks**2)))
    shapes, _ = np.linalg.qr(candidates)

    first = normalize_loading_signs(-shapes)
    second = normalize_loading_signs(-shapes)

    np.testing.assert_array_equal(first, second)
    assert first[:, 0].mean() > 0.0
    assert first[-1, 1] - first[0, 1] > 0.0
    assert first[3, 2] > np.mean(first[[0, -1], 2])


def test_known_synthetic_level_factor_is_identified(
    synthetic_yield_levels: pd.DataFrame,
) -> None:
    """The dominant constant co-movement is diagnosed as level, not assumed."""
    result = fit_yield_change_pca(synthetic_yield_levels)
    diagnostics = interpret_pca_factors(result)

    assert diagnostics[0].suggested_label == "level"
    assert diagnostics[0].level_similarity > 0.98


def test_ambiguous_factor_shape_remains_unlabelled() -> None:
    """A mixed loading does not receive an overconfident conventional label."""
    maturities = np.array([0.25, 2.0, 5.0, 10.0, 20.0, 30.0])
    ranks = np.linspace(-1.0, 1.0, len(maturities))
    level = np.ones(len(maturities)) / np.sqrt(len(maturities))
    slope = ranks / np.linalg.norm(ranks)
    mixed = (level + slope)[:, np.newaxis]

    diagnostic = interpret_loading_shapes(maturities, mixed)[0]

    assert diagnostic.suggested_label is None
    assert diagnostic.confidence_margin < 0.10


def test_estimation_uses_changes_not_cross_sectional_yield_levels(
    synthetic_yield_levels: pd.DataFrame,
) -> None:
    """A large fixed curve shape has no effect because PCA uses changes."""
    shifted = synthetic_yield_levels.copy()
    fixed_shift = {
        "3M": -0.025,
        "2Y": -0.015,
        "5Y": -0.005,
        "10Y": 0.005,
        "20Y": 0.015,
        "30Y": 0.025,
    }
    shifted = shifted.add(pd.Series(fixed_shift), axis="columns")

    original_result = fit_yield_change_pca(synthetic_yield_levels)
    shifted_result = fit_yield_change_pca(shifted)

    np.testing.assert_allclose(shifted_result.eigenvalues, original_result.eigenvalues)
    np.testing.assert_allclose(shifted_result.loadings, original_result.loadings)


def test_basis_point_conversion_is_explicit_and_exact() -> None:
    """One decimal change of 0.0001 is reported as exactly one basis point."""
    dates = pd.date_range("2025-01-01", periods=4)
    levels = pd.DataFrame(
        {
            "1Y": [0.04, 0.0401, 0.0403, 0.0402],
            "2Y": [0.05, 0.0499, 0.0500, 0.0502],
            "5Y": [0.06, 0.0603, 0.0601, 0.0600],
        },
        index=dates,
    )

    result = fit_yield_change_pca(levels)

    np.testing.assert_allclose(
        result.yield_changes_bp.to_numpy(),
        result.yield_changes_decimal.to_numpy() / 0.0001,
        rtol=0.0,
        atol=1e-13,
    )
    assert result.yield_changes_bp.iloc[0, 0] == pytest.approx(1.0)


def test_missing_levels_never_create_filled_zero_changes() -> None:
    """A missing level invalidates changes on both sides of the gap."""
    dates = pd.date_range("2025-01-01", periods=4)
    levels = pd.DataFrame(
        {
            "1Y": [0.040, np.nan, 0.041, 0.042],
            "2Y": [0.050, 0.051, 0.052, 0.053],
            "5Y": [0.060, 0.061, 0.062, 0.063],
        },
        index=dates,
    )

    changes = calculate_yield_changes(levels, missing="drop")

    assert list(changes.index) == [dates[-1]]
    assert changes.loc[dates[-1], "1Y"] == pytest.approx(0.001)
    with pytest.raises(ValueError, match="missing adjacent"):
        calculate_yield_changes(levels, missing="raise")


def test_correlation_mode_is_explicitly_standardized(
    synthetic_yield_levels: pd.DataFrame,
) -> None:
    """Correlation PCA decomposes a unit-diagonal standardized matrix."""
    result = fit_yield_change_pca(synthetic_yield_levels, method="correlation")

    assert result.method is PCAMethod.CORRELATION
    np.testing.assert_allclose(np.diag(result.decomposition_matrix), 1.0)
    assert not np.allclose(result.scales_decimal, 1.0)
