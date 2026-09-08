"""Independent tests for spot-curve pricing and curve risk analytics."""

from datetime import date

import numpy as np
import pytest

from fixed_income.bonds import FixedRateBond, PriceType, price_from_ytm
from fixed_income.curves import (
    CompoundingConvention,
    CurveRepresentation,
    YieldCurve,
)
from fixed_income.risk import (
    ShapeScenarioParameters,
    bond_risk_report,
    curve_price_details,
    curve_risk_measures,
    curve_shape_shock,
    key_rate_bump_function,
    key_rate_dv01_report,
    parallel_curve_shock,
    price_from_curve,
    shape_shock_function,
)

SETTLEMENT = date(2025, 1, 15)


@pytest.fixture
def ten_year_bond() -> FixedRateBond:
    """Return a regular ten-year 5% semiannual test bond."""
    return FixedRateBond(
        accrual_start_date=SETTLEMENT,
        maturity_date=date(2035, 1, 15),
        coupon_rate=0.05,
        face_value=100.0,
        frequency=2,
    )


@pytest.fixture
def zero_curve() -> YieldCurve:
    """Return a normal upward-sloping continuously compounded zero curve."""
    return YieldCurve(
        maturities_years=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
        values=(0.0300, 0.0310, 0.0320, 0.0340, 0.0360, 0.0380, 0.0390),
        representation=CurveRepresentation.ZERO_RATE,
        valuation_date=SETTLEMENT,
        compounding=CompoundingConvention.CONTINUOUS,
        source="synthetic test zero curve",
    )


def test_spot_curve_price_discounts_every_cash_flow_independently(
    ten_year_bond: FixedRateBond,
) -> None:
    """Spot pricing matches a manual continuous-compounding DCF calculation."""
    curve = YieldCurve(
        maturities_years=(0.5, 5.0, 10.0),
        values=(0.03, 0.04, 0.05),
        representation=CurveRepresentation.ZERO_RATE,
        valuation_date=SETTLEMENT,
        compounding=CompoundingConvention.CONTINUOUS,
    )

    result = curve_price_details(ten_year_bond, SETTLEMENT, curve)
    times = np.arange(0.5, 10.0 + 0.5, 0.5)
    amounts = np.full(20, 2.5)
    amounts[-1] += 100.0
    interpolated_rates = np.interp(times, [0.5, 5.0, 10.0], [0.03, 0.04, 0.05])
    expected = np.sum(amounts * np.exp(-interpolated_rates * times))

    assert result.dirty_price_currency == pytest.approx(expected)
    assert result.clean_price_currency == pytest.approx(expected)
    assert result.accrued_interest_currency == 0.0


def test_curve_and_ytm_apis_require_explicit_price_basis(
    ten_year_bond: FixedRateBond, zero_curve: YieldCurve
) -> None:
    """YTM and curve entry points remain separate and clean/dirty is explicit."""
    curve_price = price_from_curve(
        ten_year_bond, SETTLEMENT, zero_curve, price_type=PriceType.DIRTY
    )
    ytm_price = price_from_ytm(
        ten_year_bond, SETTLEMENT, 0.036, price_type=PriceType.DIRTY
    )

    assert curve_price > 0.0
    assert ytm_price > 0.0
    with pytest.raises(TypeError):
        price_from_curve(ten_year_bond, SETTLEMENT, zero_curve)  # type: ignore[call-arg]


@pytest.mark.parametrize("shock_bp", [-100.0, -50.0, -25.0, 25.0, 50.0, 100.0])
def test_parallel_shocks_have_exact_bp_scale_and_expected_price_direction(
    ten_year_bond: FixedRateBond, zero_curve: YieldCurve, shock_bp: float
) -> None:
    """Every node moves by bp times 0.0001 and long-bond price moves inversely."""
    shocked = parallel_curve_shock(zero_curve, shock_bp)
    np.testing.assert_allclose(
        np.asarray(shocked.values) - np.asarray(zero_curve.values),
        shock_bp * 0.0001,
        rtol=0.0,
        atol=1e-15,
    )
    base_price = price_from_curve(
        ten_year_bond, SETTLEMENT, zero_curve, price_type=PriceType.DIRTY
    )
    shocked_price = price_from_curve(
        ten_year_bond, SETTLEMENT, shocked, price_type=PriceType.DIRTY
    )
    assert (shocked_price < base_price) is (shock_bp > 0.0)


def test_curve_dv01_matches_independent_one_bp_central_difference(
    ten_year_bond: FixedRateBond, zero_curve: YieldCurve
) -> None:
    """Reported positive DV01 reconciles with independent full repricing."""
    measures = curve_risk_measures(ten_year_bond, SETTLEMENT, zero_curve)
    price_down = price_from_curve(
        ten_year_bond,
        SETTLEMENT,
        parallel_curve_shock(zero_curve, -1.0),
        price_type=PriceType.DIRTY,
    )
    price_up = price_from_curve(
        ten_year_bond,
        SETTLEMENT,
        parallel_curve_shock(zero_curve, 1.0),
        price_type=PriceType.DIRTY,
    )

    assert measures.dv01_currency_per_bp == pytest.approx(
        (price_down - price_up) / 2.0, rel=1e-12
    )
    assert measures.dv01_currency_per_bp > 0.0
    assert abs((price_down + price_up) / 2.0 - measures.base_price_currency) < 1e-4


def test_duration_is_accurate_small_and_convexity_improves_large_shocks(
    ten_year_bond: FixedRateBond, zero_curve: YieldCurve
) -> None:
    """Taylor approximations behave as expected under normal curve conditions."""
    report = bond_risk_report(
        ten_year_bond,
        SETTLEMENT,
        zero_curve,
        parallel_shocks_bp=(1.0, 100.0),
        include_shape_scenarios=False,
    ).set_index("parallel_shock_bp")

    small = report.loc[1.0]
    large = report.loc[100.0]
    assert abs(small["first_order_approximation_error_currency"]) < 1e-4
    assert abs(large["approximation_error_currency"]) < abs(
        large["first_order_approximation_error_currency"]
    )


def test_shape_scenarios_are_continuous_parameterized_mirrors() -> None:
    """Anchor values, pivot, interpolation, and flattener sign are exact."""
    parameters = ShapeScenarioParameters(
        short_maturity_years=1.0,
        pivot_maturity_years=7.0,
        long_maturity_years=20.0,
        short_end_shock_bp=-15.0,
        long_end_shock_bp=35.0,
    )
    maturities = np.array([0.5, 1.0, 4.0, 7.0, 13.5, 20.0, 30.0])
    steepener = shape_shock_function(maturities, "steepener", parameters=parameters)
    flattener = shape_shock_function(maturities, "flattener", parameters=parameters)

    assert steepener[0] == pytest.approx(-15.0 * 0.0001)
    assert steepener[1] == pytest.approx(-15.0 * 0.0001)
    assert steepener[3] == 0.0
    assert steepener[-1] == pytest.approx(35.0 * 0.0001)
    np.testing.assert_allclose(flattener, -steepener)


def test_key_rate_bump_is_local_and_uses_exact_peak_size() -> None:
    """An interior triangular bump is zero outside adjacent keys."""
    maturities = np.array([1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0, 40.0])
    bump = key_rate_bump_function(maturities, 5.0, bump_size_bp=2.0)

    assert bump[0] == 0.0
    assert bump[1] == 0.0
    assert bump[3] == pytest.approx(2.0 * 0.0001)
    assert bump[5] == 0.0
    assert np.all(bump[6:] == 0.0)


def test_key_rate_basis_partitions_parallel_bump() -> None:
    """Summed piecewise-linear key bumps equal exactly one bump at all tenors."""
    maturities = np.linspace(0.5, 35.0, 70)
    bumps = np.vstack(
        [key_rate_bump_function(maturities, key) for key in (2.0, 5.0, 10.0, 30.0)]
    )

    np.testing.assert_allclose(bumps.sum(axis=0), 0.0001, atol=1e-15, rtol=0.0)


def test_key_rate_dv01_reconciles_and_is_not_a_parallel_relabel(
    ten_year_bond: FixedRateBond, zero_curve: YieldCurve
) -> None:
    """Localized risks differ by tenor while their first-order sum reconciles."""
    report = key_rate_dv01_report(ten_year_bond, SETTLEMENT, zero_curve)

    assert list(report["key_tenor"]) == ["2Y", "5Y", "10Y", "30Y"]
    assert np.all(report["bump_size_bp"] == 1.0)
    assert np.all(report["bump_size_decimal"] == 0.0001)
    assert report["key_rate_dv01_currency_per_bp"].nunique() > 1
    assert abs(report["reconciliation_error_currency_per_bp"].iloc[0]) < 1e-6
    assert (
        report.loc[report["key_tenor"] == "10Y", "key_rate_dv01_currency_per_bp"].iloc[
            0
        ]
        > 0.0
    )


def test_curve_shape_application_moves_each_node_by_documented_function(
    zero_curve: YieldCurve,
) -> None:
    """Applied shape shocks equal the public continuous shock function."""
    shocked = curve_shape_shock(zero_curve, "steepener")
    expected = shape_shock_function(zero_curve.maturities_years, "steepener")

    np.testing.assert_allclose(
        np.asarray(shocked.values) - np.asarray(zero_curve.values), expected
    )
