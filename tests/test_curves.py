"""Tests for typed curve representations, NSS, and zero-rate discounting."""

import numpy as np
import pytest

from fixed_income.curves import (
    CompoundingConvention,
    CurveRepresentation,
    NSSParameters,
    YieldCurve,
    calibrate_nss,
    discount_cash_flows,
    interpolate_zero_rates,
    nelson_siegel_svensson,
    nss_loadings,
    nss_yields,
    validate_nss_parameters,
    zero_rates_to_discount_factors,
)

MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
KNOWN_PARAMETERS = NSSParameters(0.042, -0.025, 0.018, -0.012, 1.4, 6.0)


def test_nss_dimensions_and_explicit_parameter_entry_point() -> None:
    """Loadings and yields retain the expected observation dimensions."""
    loadings = nss_loadings(MATURITIES, 1.4, 6.0)
    direct = nss_yields(MATURITIES, KNOWN_PARAMETERS)
    explicit = nelson_siegel_svensson(
        MATURITIES, 0.042, -0.025, 0.018, -0.012, 1.4, 6.0
    )

    assert loadings.shape == (10, 4)
    assert direct.shape == (10,)
    np.testing.assert_allclose(direct, explicit)


def test_nss_is_finite_and_has_analytic_limit_at_short_maturities() -> None:
    """Taylor handling avoids cancellation at zero and tiny maturities."""
    maturities = np.array([0.0, 1e-14, 1e-9, 1e-6])
    loadings = nss_loadings(maturities, 1.5, 5.0)
    yields = nss_yields(maturities, KNOWN_PARAMETERS)

    assert np.all(np.isfinite(loadings))
    assert np.all(np.isfinite(yields))
    np.testing.assert_allclose(loadings[0], [1.0, 1.0, 0.0, 0.0])
    assert yields[0] == pytest.approx(KNOWN_PARAMETERS.beta0 + KNOWN_PARAMETERS.beta1)


@pytest.mark.parametrize(
    "parameters",
    [
        [0.04, -0.02, 0.01, 0.0, 0.0, 2.0],
        [0.04, -0.02, 0.01, 0.0, 1.0, -2.0],
        [0.04, -0.02, float("nan"), 0.0, 1.0, 2.0],
        [0.04, -0.02, 0.01],
    ],
)
def test_nss_parameter_validation_rejects_invalid_values(
    parameters: list[float],
) -> None:
    """Non-finite betas, non-positive taus, and wrong dimensions fail."""
    with pytest.raises(ValueError):
        validate_nss_parameters(parameters)


def test_calibration_recovers_exact_synthetic_nss_curve() -> None:
    """Deterministic multi-start fitting recovers an identified exact curve."""
    synthetic_yields = nss_yields(MATURITIES, KNOWN_PARAMETERS)

    result = calibrate_nss(MATURITIES, synthetic_yields)

    assert result.success
    assert result.rmse < 1e-10
    assert result.diagnostics.starts_attempted > 1
    np.testing.assert_allclose(
        result.parameters.as_array(), KNOWN_PARAMETERS.as_array(), atol=1e-6
    )
    np.testing.assert_allclose(
        result.residuals, result.observed_yields - result.fitted_yields
    )


def test_calibration_rejects_percentage_inputs_and_bad_dimensions() -> None:
    """Ambiguous percent units and underdetermined inputs are rejected."""
    with pytest.raises(ValueError, match="decimal"):
        calibrate_nss(MATURITIES, np.full(MATURITIES.size, 4.0))
    with pytest.raises(ValueError, match="six"):
        calibrate_nss([1.0, 2.0, 3.0], [0.02, 0.025, 0.03])


def test_linear_zero_rate_interpolation_recovers_nodes_and_midpoint() -> None:
    """Decimal zero rates interpolate directly in maturity-year space."""
    interpolated = interpolate_zero_rates(
        [1.0, 2.0, 5.0], [0.03, 0.04, 0.05], [1.0, 1.5, 5.0]
    )

    np.testing.assert_allclose(interpolated, [0.03, 0.035, 0.05])
    with pytest.raises(ValueError, match="outside"):
        interpolate_zero_rates([1.0, 2.0], [0.03, 0.04], [0.5])


def test_positive_continuous_zero_curve_has_decreasing_discount_factors() -> None:
    """A normal positive zero curve produces positive declining discounts."""
    maturities = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.full(maturities.size, 0.04)

    factors = zero_rates_to_discount_factors(
        maturities, rates, compounding=CompoundingConvention.CONTINUOUS
    )

    assert factors[0] == 1.0
    assert np.all(factors > 0.0)
    assert np.all(np.diff(factors) < 0.0)
    np.testing.assert_allclose(factors, np.exp(-0.04 * maturities))


def test_periodic_discounting_requires_explicit_frequency() -> None:
    """Periodic zero-rate conversion cannot infer periods per year."""
    with pytest.raises(ValueError, match="periodic_frequency"):
        zero_rates_to_discount_factors(
            [1.0], [0.04], compounding=CompoundingConvention.PERIODIC
        )


def test_cash_flow_discounting_is_auditable_and_uses_currency_units() -> None:
    """Spot present values reconcile exactly to their structured total."""
    curve = YieldCurve(
        maturities_years=(0.5, 1.0, 2.0),
        values=(0.04, 0.04, 0.04),
        representation=CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.CONTINUOUS,
        source="test zero curve",
    )

    result = discount_cash_flows([0.5, 1.0, 2.0], [2.0, 2.0, 102.0], curve)

    expected = np.array([2.0, 2.0, 102.0]) * np.exp(-0.04 * np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose(result.present_values, expected)
    assert result.total_present_value == pytest.approx(expected.sum())


def test_cmt_curve_cannot_be_silently_used_as_zero_curve() -> None:
    """Observed CMT values are never implicitly promoted to spot rates."""
    cmt_curve = YieldCurve(
        maturities_years=(1.0, 2.0, 5.0),
        values=(0.04, 0.041, 0.042),
        representation=CurveRepresentation.TREASURY_CMT,
        source="H.15",
    )

    with pytest.raises(ValueError, match="zero-rate"):
        cmt_curve.discount_factors([1.0, 2.0])


@pytest.mark.parametrize(
    "maturities, values",
    [
        ((1.0, 1.0), (0.03, 0.04)),
        ((2.0, 1.0), (0.03, 0.04)),
        ((-1.0, 2.0), (0.03, 0.04)),
        ((1.0, 2.0), (0.03, float("nan"))),
    ],
)
def test_curve_node_validation_rejects_invalid_input(
    maturities: tuple[float, ...], values: tuple[float, ...]
) -> None:
    """Nodes must be finite, unique, positive, and increasing."""
    with pytest.raises(ValueError):
        YieldCurve(
            maturities,
            values,
            CurveRepresentation.ZERO_RATE,
            compounding=CompoundingConvention.CONTINUOUS,
        )
