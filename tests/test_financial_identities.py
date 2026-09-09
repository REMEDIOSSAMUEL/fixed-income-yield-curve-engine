"""Pricing identities and independent finite differences, with decimal yield units."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fixed_income.bonds import (
    FixedRateBond,
    clean_price_from_ytm,
    convexity,
    dirty_price_from_ytm,
    dv01,
    macaulay_duration,
    modified_duration,
    yield_to_maturity,
)
from fixed_income.curves import CompoundingConvention, CurveRepresentation, YieldCurve
from fixed_income.relative_value import construct_dv01_neutral_butterfly
from fixed_income.risk import (
    curve_risk_measures,
    key_rate_dv01_report,
    price_from_curve,
)


@pytest.mark.parametrize("frequency", [1, 2])
@pytest.mark.parametrize("ytm", [-0.02, 0.0, 0.08])
@pytest.mark.parametrize("settlement", [date(2025, 1, 15), date(2025, 4, 10)])
def test_duration_convexity_and_dv01_from_price_derivatives(
    frequency: int,
    ytm: float,
    settlement: date,
) -> None:
    """Price derivatives independently verify duration, convexity and currency DV01."""
    bond = FixedRateBond(
        date(2025, 1, 15), date(2032, 1, 15), 0.05, frequency=frequency
    )
    bump = 0.00005
    base = dirty_price_from_ytm(bond, settlement, ytm)
    up = dirty_price_from_ytm(bond, settlement, ytm + bump)
    down = dirty_price_from_ytm(bond, settlement, ytm - bump)
    derivative = (up - down) / (2 * bump)
    assert modified_duration(bond, settlement, ytm) == pytest.approx(
        -derivative / base, rel=4e-8
    )
    assert dv01(bond, settlement, ytm) == pytest.approx(-derivative * 0.0001, rel=4e-8)
    assert convexity(bond, settlement, ytm) == pytest.approx(
        (up - 2 * base + down) / (base * bump * bump), rel=2e-7
    )
    scale = 10_000
    large = replace(bond, face_value=bond.face_value * scale)
    assert dv01(large, settlement, ytm) == pytest.approx(
        scale * dv01(bond, settlement, ytm)
    )
    assert macaulay_duration(large, settlement, ytm) == pytest.approx(
        macaulay_duration(bond, settlement, ytm)
    )


@pytest.mark.parametrize("frequency", [1, 2])
def test_flat_periodic_spot_curve_matches_ytm_between_coupons(frequency: int) -> None:
    """Separate pricing APIs agree only when compounding and time conventions match."""
    settlement = date(2025, 4, 10)
    bond = FixedRateBond(
        date(2025, 1, 15), date(2030, 1, 15), 0.045, frequency=frequency
    )
    curve = YieldCurve(
        (0.001, 30.0),
        (0.04, 0.04),
        CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.PERIODIC,
        periodic_frequency=frequency,
        valuation_date=settlement,
    )
    for basis, price in (
        ("clean", clean_price_from_ytm),
        ("dirty", dirty_price_from_ytm),
    ):
        assert price_from_curve(
            bond, settlement, curve, price_type=basis
        ) == pytest.approx(price(bond, settlement, 0.04), rel=1e-13)


@pytest.mark.parametrize("bump_bp", [0.25, 1.0, 2.0])
def test_continuous_zero_coupon_risk_has_known_derivatives(bump_bp: float) -> None:
    """A payment at T has limiting duration T years and convexity T squared."""
    settlement = date(2025, 1, 15)
    bond = FixedRateBond(settlement, date(2030, 1, 15), 0.0)
    curve = YieldCurve(
        (0.5, 30.0),
        (0.04, 0.04),
        CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.CONTINUOUS,
    )
    risk = curve_risk_measures(bond, settlement, curve, bump_size_bp=bump_bp)
    expected_price = 100 * np.exp(-0.2)
    assert risk.base_price_currency == pytest.approx(expected_price)
    assert risk.effective_duration_years == pytest.approx(5.0, rel=2e-7)
    assert risk.effective_convexity_years_squared == pytest.approx(25.0, rel=2e-7)
    assert risk.dv01_currency_per_bp == pytest.approx(
        5 * expected_price * 0.0001, rel=2e-7
    )


def test_key_rate_reconciliation_has_quadratic_finite_bump_error() -> None:
    """A 7.5Y sole payment has half of its first-order exposure at 5Y and 10Y."""
    settlement = date(2025, 1, 15)
    bond = FixedRateBond(settlement, date(2032, 7, 15), 0.0)
    curve = YieldCurve(
        (0.5, 30.0),
        (0.04, 0.04),
        CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.CONTINUOUS,
    )
    errors = []
    for bump_bp in (1.0, 2.0):
        report = key_rate_dv01_report(bond, settlement, curve, bump_size_bp=bump_bp)
        risks = report["key_rate_dv01_currency_per_bp"].to_numpy()
        # Exact central response of a single exponential payment under a half hat.
        exact_half = (
            100 * np.exp(-0.04 * 7.5) * np.sinh(7.5 * bump_bp * 0.0001 / 2) / bump_bp
        )
        np.testing.assert_allclose(
            risks, [0.0, exact_half, exact_half, 0.0], atol=1e-12
        )
        errors.append(report["reconciliation_error_currency_per_bp"].iloc[0])
    assert errors[0] < 0
    assert errors[1] / errors[0] == pytest.approx(4.0, rel=2e-5)


def test_dv01_neutral_fly_reprices_neutrally_to_first_order() -> None:
    """Portfolio repricing checks hedge weights using actual bond sensitivities."""
    settlement = date(2025, 1, 15)
    bonds = {
        f"{years}Y": FixedRateBond(
            settlement, date(2025 + years, 1, 15), 0.04, face_value=1.0
        )
        for years in (2, 5, 10)
    }
    units = {label: dv01(bond, settlement, 0.04) for label, bond in bonds.items()}
    hedge = construct_dv01_neutral_butterfly(units, belly_notional=1_000_000)
    notionals = hedge.set_index("maturity_years")["notional"]
    bump = 0.00001
    difference = sum(
        notionals[int(label[:-1])]
        * (
            dirty_price_from_ytm(bond, settlement, 0.04 - bump)
            - dirty_price_from_ytm(bond, settlement, 0.04 + bump)
        )
        for label, bond in bonds.items()
    )
    assert abs(difference / (2 * bump) * 0.0001) < 0.000001


@pytest.mark.parametrize("ytm", [-0.9, -0.05, 0.0, 0.3, 3.0])
def test_single_payment_ytm_matches_algebraic_inverse(ytm: float) -> None:
    """Price from a closed-form three-year zero is inverted without using the pricer."""
    bond = FixedRateBond(date(2025, 1, 15), date(2028, 1, 15), 0.0, frequency=1)
    market = 100 / (1 + ytm) ** 3
    assert yield_to_maturity(
        bond, bond.accrual_start_date, market, price_type="dirty"
    ) == pytest.approx(ytm, abs=2e-12)
