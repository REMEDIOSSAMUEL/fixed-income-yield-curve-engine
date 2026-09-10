"""Independent tests for fixed-rate bond cash-flow and YTM analytics."""

from datetime import date, datetime

import pytest

from fixed_income.bonds import (
    BASIS_POINT,
    DayCountConvention,
    FixedRateBond,
    PriceType,
    accrued_interest,
    clean_price_from_ytm,
    convexity,
    day_count_fraction,
    dirty_price_from_ytm,
    dv01,
    generate_coupon_schedule,
    generate_fixed_rate_cash_flows,
    macaulay_duration,
    modified_duration,
    yield_to_maturity,
)


@pytest.fixture
def five_year_bond() -> FixedRateBond:
    """Return a regular semiannual 5% bond with 100 currency face value."""
    return FixedRateBond(
        accrual_start_date=date(2024, 8, 15),
        maturity_date=date(2029, 8, 15),
        coupon_rate=0.05,
        face_value=100.0,
        frequency=2,
    )


def test_semiannual_cash_flow_schedule_is_auditable(
    five_year_bond: FixedRateBond,
) -> None:
    """Regular cash flows have exact coupons and one principal repayment."""
    cash_flows = generate_fixed_rate_cash_flows(five_year_bond)

    assert len(cash_flows) == 10
    assert [cash_flow.payment_date for cash_flow in cash_flows] == sorted(
        cash_flow.payment_date for cash_flow in cash_flows
    )
    assert all(
        cash_flow.coupon_amount == pytest.approx(2.5) for cash_flow in cash_flows
    )
    assert sum(cash_flow.coupon_amount for cash_flow in cash_flows) == pytest.approx(
        25.0
    )
    assert [cash_flow.principal_amount for cash_flow in cash_flows[:-1]] == [0.0] * 9
    assert cash_flows[-1].principal_amount == 100.0
    assert cash_flows[-1].total_amount == 102.5


def test_month_end_schedule_preserves_roll_through_leap_year() -> None:
    """Backward generation retains explicit end-of-month behavior."""
    bond = FixedRateBond(
        accrual_start_date=date(2023, 8, 31),
        maturity_date=date(2025, 8, 31),
        coupon_rate=0.04,
    )

    payment_dates = [period.payment_date for period in generate_coupon_schedule(bond)]

    assert payment_dates == [
        date(2024, 2, 29),
        date(2024, 8, 31),
        date(2025, 2, 28),
        date(2025, 8, 31),
    ]


def test_actual_actual_fraction_uses_actual_reference_period_days() -> None:
    """Leap-day inclusion affects the independently counted denominator."""
    fraction = day_count_fraction(
        date(2023, 8, 31),
        date(2023, 11, 30),
        reference_start_date=date(2023, 8, 31),
        reference_end_date=date(2024, 2, 29),
        frequency=2,
    )

    assert fraction == pytest.approx(91 / 182 / 2)


@pytest.mark.parametrize("frequency, periods", [(1, 5), (2, 10)])
def test_coupon_frequency_controls_schedule_and_coupon(
    frequency: int, periods: int
) -> None:
    """Annual and semiannual contracts use their declared payment frequency."""
    bond = FixedRateBond(
        accrual_start_date=date(2024, 6, 30),
        maturity_date=date(2029, 6, 30),
        coupon_rate=0.06,
        frequency=frequency,
    )

    cash_flows = generate_fixed_rate_cash_flows(bond)

    assert len(cash_flows) == periods
    assert cash_flows[0].coupon_amount == pytest.approx(6.0 / frequency)
    assert all(
        cash_flow.accrual_fraction == pytest.approx(1.0 / frequency)
        for cash_flow in cash_flows
    )


def test_zero_coupon_price_matches_closed_form() -> None:
    """A zero coupon on a coupon date reduces to one elementary discount factor."""
    bond = FixedRateBond(
        accrual_start_date=date(2025, 1, 15),
        maturity_date=date(2030, 1, 15),
        coupon_rate=0.0,
        face_value=100.0,
        frequency=2,
    )
    settlement = date(2025, 1, 15)
    ytm = 0.04
    expected = 100.0 / (1.0 + ytm / 2) ** 10

    assert dirty_price_from_ytm(bond, settlement, ytm) == pytest.approx(
        expected, rel=1e-14
    )


@pytest.mark.parametrize("frequency", [1, 2])
def test_coupon_equals_ytm_gives_par_on_coupon_date(frequency: int) -> None:
    """Compatible coupon and compounding assumptions produce par."""
    bond = FixedRateBond(
        accrual_start_date=date(2020, 5, 31),
        maturity_date=date(2030, 5, 31),
        coupon_rate=0.0475,
        frequency=frequency,
    )
    settlement = date(2025, 5, 31)

    assert dirty_price_from_ytm(bond, settlement, 0.0475) == pytest.approx(
        bond.face_value, abs=1e-12
    )
    assert accrued_interest(bond, settlement) == 0.0


def test_price_moves_inversely_with_yield(five_year_bond: FixedRateBond) -> None:
    """The same positive cash flows rise on a yield fall and fall on a rise."""
    settlement = date(2025, 3, 4)
    low_yield_price = dirty_price_from_ytm(five_year_bond, settlement, 0.02)
    base_yield_price = dirty_price_from_ytm(five_year_bond, settlement, 0.05)
    high_yield_price = dirty_price_from_ytm(five_year_bond, settlement, 0.08)

    assert low_yield_price > base_yield_price > high_yield_price


def test_clean_price_approaches_face_immediately_before_maturity() -> None:
    """Accrued coupon removal leaves approximately face near final payment."""
    bond = FixedRateBond(
        accrual_start_date=date(2024, 9, 30),
        maturity_date=date(2026, 9, 30),
        coupon_rate=0.06,
        frequency=2,
    )
    settlement = date(2026, 9, 29)

    clean_price = clean_price_from_ytm(bond, settlement, 0.06)

    assert clean_price == pytest.approx(100.0, abs=0.001)


@pytest.mark.parametrize("chosen_yield", [-0.0125, 0.0, 0.03725, 0.19])
@pytest.mark.parametrize("price_type", [PriceType.CLEAN, PriceType.DIRTY])
def test_ytm_round_trip_at_fractional_settlement(
    five_year_bond: FixedRateBond,
    chosen_yield: float,
    price_type: PriceType,
) -> None:
    """Brent inversion recovers positive, zero, and negative decimal yields."""
    settlement = date(2025, 4, 7)
    if price_type is PriceType.CLEAN:
        price = clean_price_from_ytm(five_year_bond, settlement, chosen_yield)
    else:
        price = dirty_price_from_ytm(five_year_bond, settlement, chosen_yield)

    solved_yield = yield_to_maturity(
        five_year_bond,
        settlement,
        price,
        price_type=price_type,
    )

    assert solved_yield == pytest.approx(chosen_yield, abs=2e-12)


def test_bad_ytm_bracket_reports_failure(five_year_bond: FixedRateBond) -> None:
    """A bracket with same-sign price residuals is rejected clearly."""
    settlement = date(2025, 4, 7)
    market_price = dirty_price_from_ytm(five_year_bond, settlement, 0.04)

    with pytest.raises(ValueError, match="does not contain"):
        yield_to_maturity(
            five_year_bond,
            settlement,
            market_price,
            price_type="dirty",
            bracket=(0.10, 0.20),
        )


def test_clean_dirty_accrued_identity(five_year_bond: FixedRateBond) -> None:
    """Clean and dirty prices reconcile through separately calculated accrual."""
    settlement = date(2025, 5, 2)
    ytm = 0.042
    dirty = dirty_price_from_ytm(five_year_bond, settlement, ytm)
    clean = clean_price_from_ytm(five_year_bond, settlement, ytm)
    accrued = accrued_interest(five_year_bond, settlement)

    assert dirty == pytest.approx(clean + accrued, abs=1e-14)


@pytest.mark.parametrize(
    "coupon_date",
    [date(2024, 8, 15), date(2025, 2, 15), date(2025, 8, 15)],
)
def test_accrued_interest_is_zero_on_coupon_dates(
    five_year_bond: FixedRateBond, coupon_date: date
) -> None:
    """Coupon boundaries reset accrued interest to exactly zero."""
    assert accrued_interest(five_year_bond, coupon_date) == 0.0


def test_accrued_interest_increases_sensibly_within_period(
    five_year_bond: FixedRateBond,
) -> None:
    """Actual/Actual accrual is monotone and matches a calendar-day ratio."""
    first = accrued_interest(five_year_bond, date(2025, 3, 1))
    second = accrued_interest(five_year_bond, date(2025, 5, 1))
    expected_second = 2.5 * 75 / 181

    assert 0.0 < first < second < 2.5
    assert second == pytest.approx(expected_second)


def test_duration_relationship_and_positive_values(
    five_year_bond: FixedRateBond,
) -> None:
    """Modified duration applies the nominal-compounding adjustment."""
    settlement = date(2025, 4, 7)
    ytm = 0.043
    macaulay = macaulay_duration(five_year_bond, settlement, ytm)
    modified = modified_duration(five_year_bond, settlement, ytm)

    assert macaulay > 0.0
    assert modified > 0.0
    assert modified == pytest.approx(macaulay / (1.0 + ytm / 2), rel=1e-14)


def test_zero_coupon_macaulay_duration_matches_fractional_period_time() -> None:
    """A sole principal flow has duration equal to its discounting time."""
    bond = FixedRateBond(
        accrual_start_date=date(2024, 8, 31),
        maturity_date=date(2026, 8, 31),
        coupon_rate=0.0,
        frequency=2,
    )
    settlement = date(2025, 11, 30)
    # Nov 30 to Feb 28 is 90/181 of the current period, then one whole period.
    expected_years = (90 / 181 + 1) / 2

    assert macaulay_duration(bond, settlement, 0.03) == pytest.approx(
        expected_years, rel=1e-14
    )


def test_dv01_agrees_with_independent_central_difference(
    five_year_bond: FixedRateBond,
) -> None:
    """Analytical positive DV01 matches symmetric repricing at plus/minus 1 bp."""
    settlement = date(2025, 4, 7)
    ytm = 0.043
    price_down_one_bp = dirty_price_from_ytm(
        five_year_bond, settlement, ytm - BASIS_POINT
    )
    price_up_one_bp = dirty_price_from_ytm(
        five_year_bond, settlement, ytm + BASIS_POINT
    )
    finite_difference_dv01 = (price_down_one_bp - price_up_one_bp) / 2.0

    assert dv01(five_year_bond, settlement, ytm) == pytest.approx(
        finite_difference_dv01, rel=2e-7
    )


def test_convexity_is_positive_and_matches_finite_difference(
    five_year_bond: FixedRateBond,
) -> None:
    """Analytical curvature reconciles to an independent second difference."""
    settlement = date(2025, 4, 7)
    ytm = 0.043
    bump = 0.0005
    base_price = dirty_price_from_ytm(five_year_bond, settlement, ytm)
    price_down = dirty_price_from_ytm(five_year_bond, settlement, ytm - bump)
    price_up = dirty_price_from_ytm(five_year_bond, settlement, ytm + bump)
    finite_difference_convexity = (
        (price_down - 2.0 * base_price + price_up) / bump**2 / base_price
    )
    analytical_convexity = convexity(five_year_bond, settlement, ytm)

    assert analytical_convexity > 0.0
    assert analytical_convexity == pytest.approx(finite_difference_convexity, rel=2e-6)


def test_convexity_improves_moderate_shock_price_estimate(
    five_year_bond: FixedRateBond,
) -> None:
    """Adding second-order curvature improves on duration for a 50 bp move."""
    settlement = date(2025, 4, 7)
    ytm = 0.043
    shock = 0.005
    base_price = dirty_price_from_ytm(five_year_bond, settlement, ytm)
    shocked_price = dirty_price_from_ytm(five_year_bond, settlement, ytm + shock)
    actual_fractional_change = shocked_price / base_price - 1.0
    duration_only = -modified_duration(five_year_bond, settlement, ytm) * shock
    duration_and_convexity = (
        duration_only + 0.5 * convexity(five_year_bond, settlement, ytm) * shock**2
    )

    assert abs(duration_and_convexity - actual_fractional_change) < abs(
        duration_only - actual_fractional_change
    )


def test_edge_case_one_day_before_maturity_has_small_positive_risk() -> None:
    """Near maturity, valid prices and risk measures remain finite and positive."""
    bond = FixedRateBond(
        accrual_start_date=date(2025, 9, 30),
        maturity_date=date(2026, 9, 30),
        coupon_rate=0.04,
        frequency=2,
    )
    settlement = date(2026, 9, 29)

    assert dirty_price_from_ytm(bond, settlement, -0.01) > 100.0
    assert 0.0 < macaulay_duration(bond, settlement, -0.01) < 0.01
    assert dv01(bond, settlement, -0.01) > 0.0
    assert convexity(bond, settlement, -0.01) > 0.0


@pytest.mark.parametrize("frequency", [0, 3, 4, -2])
def test_invalid_frequency_fails_clearly(frequency: int) -> None:
    """Unsupported frequencies are not silently approximated."""
    with pytest.raises(ValueError, match="frequency"):
        FixedRateBond(
            accrual_start_date=date(2025, 1, 15),
            maturity_date=date(2030, 1, 15),
            coupon_rate=0.04,
            frequency=frequency,
        )


@pytest.mark.parametrize("face_value", [0.0, -100.0, float("nan"), float("inf")])
def test_invalid_face_value_fails_clearly(face_value: float) -> None:
    """Face value must be positive and finite currency."""
    with pytest.raises(ValueError, match="face_value"):
        FixedRateBond(
            accrual_start_date=date(2025, 1, 15),
            maturity_date=date(2030, 1, 15),
            coupon_rate=0.04,
            face_value=face_value,
        )


@pytest.mark.parametrize("coupon_rate", [-0.01, 1.01, float("nan")])
def test_invalid_or_ambiguous_coupon_rate_fails_clearly(coupon_rate: float) -> None:
    """Coupon rates use explicit decimal units and a documented supported range."""
    with pytest.raises(ValueError, match="coupon_rate"):
        FixedRateBond(
            accrual_start_date=date(2025, 1, 15),
            maturity_date=date(2030, 1, 15),
            coupon_rate=coupon_rate,
        )


def test_invalid_dates_and_stub_schedule_fail_clearly() -> None:
    """Reversed dates, datetimes, and unsupported stub terms are rejected."""
    with pytest.raises(ValueError, match="maturity_date"):
        FixedRateBond(date(2025, 1, 15), date(2025, 1, 15), 0.04)
    with pytest.raises(TypeError, match="datetime.date"):
        FixedRateBond(datetime(2025, 1, 15), date(2030, 1, 15), 0.04)
    with pytest.raises(ValueError, match="stub"):
        FixedRateBond(date(2025, 2, 1), date(2030, 1, 15), 0.04)


def test_invalid_settlement_ytm_and_price_fail_clearly(
    five_year_bond: FixedRateBond,
) -> None:
    """Invalid valuation inputs do not return misleading numerical values."""
    with pytest.raises(ValueError, match="settlement_date"):
        dirty_price_from_ytm(five_year_bond, date(2030, 1, 1), 0.04)
    with pytest.raises(ValueError, match="ytm"):
        dirty_price_from_ytm(five_year_bond, date(2025, 1, 15), -2.0)
    with pytest.raises(ValueError, match="market_price"):
        yield_to_maturity(
            five_year_bond,
            date(2025, 1, 15),
            0.0,
            price_type="dirty",
        )
    with pytest.raises(ValueError, match="price_type"):
        yield_to_maturity(
            five_year_bond,
            date(2025, 1, 15),
            100.0,
            price_type="unspecified",
        )


def test_day_count_rejects_segment_outside_reference_period() -> None:
    """Accrual fractions cannot silently use an inconsistent coupon period."""
    with pytest.raises(ValueError, match="inside"):
        day_count_fraction(
            date(2025, 1, 1),
            date(2025, 8, 1),
            reference_start_date=date(2025, 2, 1),
            reference_end_date=date(2025, 8, 1),
            frequency=2,
            convention=DayCountConvention.ACTUAL_ACTUAL_TREASURY,
        )


def test_price_underflow_is_rejected_instead_of_reported_as_zero() -> None:
    """A positive distant cash flow cannot have an exactly zero mathematical price."""
    bond = FixedRateBond(date(2025, 1, 15), date(3025, 1, 15), 0.0)
    with pytest.raises(ArithmeticError, match="numeric range"):
        dirty_price_from_ytm(bond, bond.accrual_start_date, 10.0)


def test_negative_clean_price_round_trip_with_positive_dirty_price() -> None:
    """Accrued interest can exceed PV at extreme yields without invalidating YTM."""
    bond = FixedRateBond(date(2025, 1, 15), date(2026, 1, 15), 0.1)
    settlement = date(2025, 4, 15)
    clean = clean_price_from_ytm(bond, settlement, 100.0)
    assert clean < 0.0
    assert clean + accrued_interest(bond, settlement) > 0.0
    assert yield_to_maturity(
        bond, settlement, clean, price_type="clean"
    ) == pytest.approx(100.0, rel=1e-12)


@pytest.mark.parametrize("dirty_target", [0.0, -1.0])
def test_clean_price_requires_positive_total_dirty_price(dirty_target: float) -> None:
    """A non-positive full price cannot match positive future cash flows."""
    bond = FixedRateBond(date(2025, 1, 15), date(2026, 1, 15), 0.1)
    settlement = date(2025, 4, 15)
    clean = dirty_target - accrued_interest(bond, settlement)
    with pytest.raises(ValueError, match="dirty price must be greater than zero"):
        yield_to_maturity(bond, settlement, clean, price_type="clean")
