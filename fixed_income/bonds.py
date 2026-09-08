"""Transparent analytics for nominal fixed-rate bullet bonds.

The module deliberately implements the financial mathematics directly. It
supports regular annual and semiannual schedules with no business-day
adjustment and Actual/Actual coupon-period accrual (a convention suitable for
simple US Treasury-style examples).

Yield to maturity (YTM) is a nominal annual decimal yield compounded at the
bond's coupon frequency. A value of ``0.05`` means 5%. Cash flows are
discounted using fractional coupon periods when settlement falls between
coupon dates. This is YTM pricing, not spot-curve discounting.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from scipy.optimize import brentq

BASIS_POINT = 0.0001
"""One basis point expressed as a decimal annual yield change."""


class DayCountConvention(StrEnum):
    """Supported day-count conventions.

    ``ACTUAL_ACTUAL_TREASURY`` measures an accrual segment using actual
    calendar days divided by the actual days in its surrounding regular coupon
    period. Dividing that ratio by coupon frequency gives a year fraction.
    """

    ACTUAL_ACTUAL_TREASURY = "actual_actual_treasury"


class PriceType(StrEnum):
    """Market-price basis; both values are currency amounts for the bond face."""

    CLEAN = "clean"
    DIRTY = "dirty"


@dataclass(frozen=True)
class FixedRateBond:
    """Contract terms for a nominal fixed-rate bullet bond.

    Args:
        accrual_start_date: Calendar date on which the first coupon starts
            accruing. It must lie on the regular schedule implied by maturity.
        maturity_date: Calendar date of the final coupon and principal payment.
        coupon_rate: Decimal nominal annual coupon rate; ``0.05`` means 5%.
            The supported range is 0 through 1 (0% through 100%).
        face_value: Positive principal and currency-price scale, in currency.
        frequency: Coupon payments and YTM compounding periods per year. The
            initial implementation supports 1 (annual) and 2 (semiannual).
        day_count: Accrual convention. Only Actual/Actual Treasury-style
            coupon-period accrual is currently supported.

    Notes:
        Dates are unadjusted calendar dates: no holiday calendar, settlement
        lag, ex-coupon period, or business-day adjustment is applied.
    """

    accrual_start_date: date
    maturity_date: date
    coupon_rate: float
    face_value: float = 100.0
    frequency: int = 2
    day_count: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL_TREASURY

    def __post_init__(self) -> None:
        """Validate contract terms and normalize the day-count enum."""
        _validate_date(self.accrual_start_date, "accrual_start_date")
        _validate_date(self.maturity_date, "maturity_date")
        if self.maturity_date <= self.accrual_start_date:
            raise ValueError("maturity_date must be after accrual_start_date")
        _validate_frequency(self.frequency)
        _validate_finite_number(self.face_value, "face_value")
        if self.face_value <= 0.0:
            raise ValueError("face_value must be greater than zero")
        _validate_finite_number(self.coupon_rate, "coupon_rate")
        if not 0.0 <= self.coupon_rate <= 1.0:
            raise ValueError(
                "coupon_rate must be a decimal rate between 0 and 1; use 0.05 for 5%"
            )
        try:
            convention = DayCountConvention(self.day_count)
        except ValueError as exc:
            raise ValueError(
                f"unsupported day-count convention: {self.day_count!r}"
            ) from exc
        object.__setattr__(self, "day_count", convention)

        # Constructing the schedule here rejects irregular/stub terms early.
        _regular_schedule_dates(
            self.accrual_start_date, self.maturity_date, self.frequency
        )


@dataclass(frozen=True)
class CouponPeriod:
    """One regular accrual period, with dates and year fraction.

    ``accrual_fraction`` is in years. ``payment_date`` is unadjusted and is
    equal to ``accrual_end_date`` in the supported no-calendar convention.
    """

    accrual_start_date: date
    accrual_end_date: date
    payment_date: date
    accrual_fraction: float


@dataclass(frozen=True)
class BondCashFlow:
    """A contractual cash flow in currency units.

    ``coupon_amount``, ``principal_amount``, and ``total_amount`` are currency
    amounts for the bond's supplied face value. Dates are unadjusted calendar
    dates and ``accrual_fraction`` is in years.
    """

    accrual_start_date: date
    accrual_end_date: date
    payment_date: date
    accrual_fraction: float
    coupon_amount: float
    principal_amount: float
    total_amount: float


def generate_coupon_schedule(bond: FixedRateBond) -> tuple[CouponPeriod, ...]:
    """Generate the bond's ordered regular coupon schedule.

    Args:
        bond: Bond terms. Dates are calendar dates and frequency is payments
            per year.

    Returns:
        Coupon periods whose accrual fractions are in years. Payment dates are
        unadjusted and equal period end dates.

    Notes:
        The maturity date fixes the day-of-month roll. If maturity is a month
        end, all possible schedule dates use month ends. Stub periods are not
        supported and fail explicitly.
    """
    if not isinstance(bond, FixedRateBond):
        raise TypeError("bond must be a FixedRateBond")
    dates = _regular_schedule_dates(
        bond.accrual_start_date, bond.maturity_date, bond.frequency
    )
    return tuple(
        CouponPeriod(
            accrual_start_date=start,
            accrual_end_date=end,
            payment_date=end,
            accrual_fraction=day_count_fraction(
                start,
                end,
                reference_start_date=start,
                reference_end_date=end,
                frequency=bond.frequency,
                convention=bond.day_count,
            ),
        )
        for start, end in zip(dates[:-1], dates[1:], strict=True)
    )


def generate_fixed_rate_cash_flows(
    bond: FixedRateBond,
) -> tuple[BondCashFlow, ...]:
    """Generate coupon and bullet-principal cash flows in currency units.

    Args:
        bond: Bond terms. ``coupon_rate`` is a decimal annual rate,
            ``face_value`` is currency, and frequency is payments per year.

    Returns:
        Ordered cash flows in currency units. Each regular coupon is
        ``face_value * coupon_rate / frequency``; principal is paid once at
        maturity.
    """
    coupon_amount = bond.face_value * bond.coupon_rate / bond.frequency
    periods = generate_coupon_schedule(bond)
    cash_flows: list[BondCashFlow] = []
    for period in periods:
        principal_amount = (
            bond.face_value if period.payment_date == bond.maturity_date else 0.0
        )
        cash_flows.append(
            BondCashFlow(
                accrual_start_date=period.accrual_start_date,
                accrual_end_date=period.accrual_end_date,
                payment_date=period.payment_date,
                accrual_fraction=period.accrual_fraction,
                coupon_amount=coupon_amount,
                principal_amount=principal_amount,
                total_amount=coupon_amount + principal_amount,
            )
        )
    return tuple(cash_flows)


def day_count_fraction(
    start_date: date,
    end_date: date,
    *,
    reference_start_date: date,
    reference_end_date: date,
    frequency: int,
    convention: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL_TREASURY,
) -> float:
    """Calculate an accrual year fraction under a supported convention.

    Args:
        start_date: Inclusive start of the measured calendar-date interval.
        end_date: Exclusive end of the measured calendar-date interval.
        reference_start_date: Start of the containing regular coupon period.
        reference_end_date: End of the containing regular coupon period.
        frequency: Coupon periods per year, currently 1 or 2.
        convention: Day-count convention.

    Returns:
        Dimensionless year fraction. For Actual/Actual Treasury-style accrual,
        this is ``actual segment days / actual coupon-period days / frequency``.
    """
    for value, name in (
        (start_date, "start_date"),
        (end_date, "end_date"),
        (reference_start_date, "reference_start_date"),
        (reference_end_date, "reference_end_date"),
    ):
        _validate_date(value, name)
    _validate_frequency(frequency)
    try:
        normalized_convention = DayCountConvention(convention)
    except ValueError as exc:
        raise ValueError(f"unsupported day-count convention: {convention!r}") from exc
    if normalized_convention is not DayCountConvention.ACTUAL_ACTUAL_TREASURY:
        raise ValueError(f"unsupported day-count convention: {convention!r}")
    if reference_end_date <= reference_start_date:
        raise ValueError("reference_end_date must be after reference_start_date")
    if start_date < reference_start_date or end_date > reference_end_date:
        raise ValueError("measured dates must lie inside the reference coupon period")
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")

    segment_days = (end_date - start_date).days
    reference_days = (reference_end_date - reference_start_date).days
    return segment_days / reference_days / frequency


def accrued_interest(bond: FixedRateBond, settlement_date: date) -> float:
    """Calculate accrued coupon interest at settlement, in currency units.

    Args:
        bond: Bond terms; coupon rate is a decimal annual rate and face value is
            currency.
        settlement_date: Calendar settlement date, from accrual start through
            maturity. This is not a trade date and no settlement lag is added.

    Returns:
        Accrued interest in currency for the supplied face value. It is zero on
        every coupon date. No ex-coupon adjustment is supported.

    Notes:
        For a regular coupon ``C``, accrued interest is ``C * elapsed_days /
        coupon_period_days`` under the supported Actual/Actual convention.
    """
    _validate_settlement(bond, settlement_date, allow_maturity=True)
    if settlement_date == bond.maturity_date:
        return 0.0

    coupon_amount = bond.face_value * bond.coupon_rate / bond.frequency
    for period in generate_coupon_schedule(bond):
        if period.accrual_start_date <= settlement_date < period.accrual_end_date:
            elapsed_years = day_count_fraction(
                period.accrual_start_date,
                settlement_date,
                reference_start_date=period.accrual_start_date,
                reference_end_date=period.accrual_end_date,
                frequency=bond.frequency,
                convention=bond.day_count,
            )
            elapsed_coupon_periods = elapsed_years * bond.frequency
            return coupon_amount * elapsed_coupon_periods
    raise RuntimeError("settlement date was not located in the coupon schedule")


def dirty_price_from_ytm(
    bond: FixedRateBond, settlement_date: date, ytm: float
) -> float:
    """Price future cash flows from nominal annual YTM, in currency units.

    Args:
        bond: Bond terms; cash flows and returned price use the currency units
            of ``face_value``.
        settlement_date: Calendar settlement date. Cash flows on settlement
            are excluded, corresponding to settlement just after that coupon.
        ytm: Nominal annual yield as a decimal, compounded ``bond.frequency``
            times per year. For example, ``0.0425`` means 4.25%.

    Returns:
        Dirty (full) price in currency for the supplied face value.

    Notes:
        If settlement is between coupons, the first cash flow is discounted by
        the Actual/Actual fraction of a coupon period remaining; later cash
        flows add whole coupon periods. The admissible domain is
        ``ytm > -bond.frequency``, ensuring a positive periodic discount base.
        This is conventional YTM pricing and does not use a spot curve.
    """
    _validate_ytm(ytm, bond.frequency)
    cash_flows, period_counts = _future_cash_flows_and_period_counts(
        bond, settlement_date
    )
    periodic_base = 1.0 + ytm / bond.frequency
    return math.fsum(
        cash_flow.total_amount * periodic_base ** (-period_count)
        for cash_flow, period_count in zip(cash_flows, period_counts, strict=True)
    )


def clean_price_from_ytm(
    bond: FixedRateBond, settlement_date: date, ytm: float
) -> float:
    """Calculate clean price from nominal annual YTM, in currency units.

    Args:
        bond: Bond terms; face value and returned price are currency amounts.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.

    Returns:
        Clean price in currency, calculated exactly as dirty price minus accrued
        interest. A value of 100 therefore means 100 currency units, not an
        implicit price quote independent of face value.
    """
    dirty_price = dirty_price_from_ytm(bond, settlement_date, ytm)
    return dirty_price - accrued_interest(bond, settlement_date)


def price_from_ytm(
    bond: FixedRateBond,
    settlement_date: date,
    ytm: float,
    *,
    price_type: PriceType | str,
) -> float:
    """Price a bond from a single conventional YTM, in currency units.

    Args:
        bond: Bond terms; face value and returned price are currency amounts.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.
        price_type: Explicitly ``"clean"`` or ``"dirty"``. There is no
            default, preventing an implicit clean/dirty-price choice.

    Returns:
        Clean or dirty currency price, as explicitly selected by
        ``price_type``.

    Notes:
        This entry point uses a single YTM for every cash flow. Spot-curve
        discounting is deliberately exposed separately as
        :func:`fixed_income.risk.price_from_curve`.
    """
    try:
        normalized_price_type = PriceType(price_type)
    except ValueError as exc:
        raise ValueError("price_type must be explicitly 'clean' or 'dirty'") from exc
    if normalized_price_type is PriceType.DIRTY:
        return dirty_price_from_ytm(bond, settlement_date, ytm)
    return clean_price_from_ytm(bond, settlement_date, ytm)


def yield_to_maturity(
    bond: FixedRateBond,
    settlement_date: date,
    market_price: float,
    *,
    price_type: PriceType | str,
    bracket: tuple[float, float] | None = None,
    xtol: float = 1e-12,
    max_iterations: int = 100,
) -> float:
    """Solve the nominal annual YTM matching a supplied clean or dirty price.

    Args:
        bond: Bond terms; face value and market price share currency units.
        settlement_date: Calendar settlement date before maturity.
        market_price: Positive clean or dirty currency price, as identified by
            ``price_type``.
        price_type: Explicitly ``"clean"`` or ``"dirty"``; there is no default
            because silently mixing price bases is financially unsafe.
        bracket: Optional two-element bracket of decimal nominal annual yields.
            Both endpoints must exceed ``-bond.frequency``. If omitted, a
            bracket is expanded automatically over the valid yield domain.
        xtol: Absolute solver tolerance in decimal annual yield units.
        max_iterations: Positive maximum number of Brent iterations.

    Returns:
        Decimal nominal annual YTM, compounded at coupon frequency.

    Raises:
        ValueError: If inputs are invalid or a supplied bracket does not contain
            a root.
        RuntimeError: If a numerical bracket or converged root cannot be found.

    Notes:
        The dirty-price equation is solved with ``scipy.optimize.brentq``.
        Positive bullet-bond cash flows make price monotone for
        ``ytm > -frequency``. Negative yields are therefore supported whenever
        the periodic discount base remains positive.
    """
    _validate_settlement(bond, settlement_date, allow_maturity=False)
    _validate_finite_number(market_price, "market_price")
    if market_price <= 0.0:
        raise ValueError("market_price must be greater than zero")
    try:
        normalized_price_type = PriceType(price_type)
    except ValueError as exc:
        raise ValueError("price_type must be explicitly 'clean' or 'dirty'") from exc
    _validate_finite_number(xtol, "xtol")
    if xtol <= 0.0:
        raise ValueError("xtol must be greater than zero")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")

    target_dirty_price = market_price
    if normalized_price_type is PriceType.CLEAN:
        target_dirty_price += accrued_interest(bond, settlement_date)

    def price_residual(candidate_ytm: float) -> float:
        return dirty_price_from_ytm(bond, settlement_date, candidate_ytm) - (
            target_dirty_price
        )

    if bracket is None:
        lower, upper = _find_ytm_bracket(bond, settlement_date, target_dirty_price)
    else:
        lower, upper = _validate_ytm_bracket(bracket, bond.frequency)
        lower_residual = price_residual(lower)
        upper_residual = price_residual(upper)
        if not _opposite_signs_or_zero(lower_residual, upper_residual):
            raise ValueError(
                "supplied YTM bracket does not contain a price root: "
                f"residuals are {lower_residual:.12g} and {upper_residual:.12g}"
            )

    try:
        root, result = brentq(
            price_residual,
            lower,
            upper,
            xtol=xtol,
            rtol=4.0 * math.ulp(1.0),
            maxiter=max_iterations,
            full_output=True,
            disp=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"YTM root finding failed within bracket ({lower}, {upper}): {exc}"
        ) from exc
    if not result.converged:
        raise RuntimeError(
            f"YTM root finding did not converge after {result.iterations} iterations"
        )
    residual = price_residual(root)
    price_tolerance = max(1e-10, abs(target_dirty_price) * 1e-10)
    if not math.isfinite(residual) or abs(residual) > price_tolerance:
        raise RuntimeError(
            "YTM solver returned an unacceptable price residual: "
            f"{residual:.12g} currency units"
        )
    return root


def macaulay_duration(bond: FixedRateBond, settlement_date: date, ytm: float) -> float:
    """Calculate YTM-based Macaulay duration in years.

    Args:
        bond: Bond terms; coupon rate and YTM conventions are kept distinct.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.

    Returns:
        Present-value-weighted time to the future cash flows, in years. Time is
        fractional coupon-period count divided by frequency, matching the YTM
        discount exponents rather than an unrelated calendar-year convention.
    """
    _validate_ytm(ytm, bond.frequency)
    cash_flows, period_counts = _future_cash_flows_and_period_counts(
        bond, settlement_date
    )
    periodic_base = 1.0 + ytm / bond.frequency
    present_values = tuple(
        cash_flow.total_amount * periodic_base ** (-period_count)
        for cash_flow, period_count in zip(cash_flows, period_counts, strict=True)
    )
    dirty_price = math.fsum(present_values)
    if dirty_price <= 0.0 or not math.isfinite(dirty_price):
        raise ArithmeticError("dirty price must be finite and positive for duration")
    weighted_times = math.fsum(
        (period_count / bond.frequency) * present_value
        for period_count, present_value in zip(
            period_counts, present_values, strict=True
        )
    )
    return weighted_times / dirty_price


def modified_duration(bond: FixedRateBond, settlement_date: date, ytm: float) -> float:
    """Calculate modified duration with respect to decimal annual YTM.

    Args:
        bond: Bond terms; frequency is compounding periods per year.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.

    Returns:
        Modified duration in years per unit decimal annual yield. It satisfies
        ``D_modified = D_Macaulay / (1 + ytm / frequency)`` and approximates
        ``change_in_price / price = -D_modified * change_in_ytm``.
    """
    _validate_ytm(ytm, bond.frequency)
    return macaulay_duration(bond, settlement_date, ytm) / (1.0 + ytm / bond.frequency)


def convexity(bond: FixedRateBond, settlement_date: date, ytm: float) -> float:
    """Calculate analytical dirty-price convexity in years squared.

    Args:
        bond: Bond terms; cash flows are fixed currency amounts.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.

    Returns:
        ``(1 / dirty_price) * d2(dirty_price) / d(ytm)^2`` in years squared.

    Notes:
        For a cash flow with possibly fractional coupon-period exponent ``n``,
        its second derivative is ``CF * n * (n + 1) / frequency^2 *
        (1 + ytm / frequency)^(-n - 2)``. This generalizes the usual integer
        coupon-index formula consistently with this module's fractional-period
        price equation.
    """
    _validate_ytm(ytm, bond.frequency)
    cash_flows, period_counts = _future_cash_flows_and_period_counts(
        bond, settlement_date
    )
    periodic_base = 1.0 + ytm / bond.frequency
    dirty_price = math.fsum(
        cash_flow.total_amount * periodic_base ** (-period_count)
        for cash_flow, period_count in zip(cash_flows, period_counts, strict=True)
    )
    if dirty_price <= 0.0 or not math.isfinite(dirty_price):
        raise ArithmeticError("dirty price must be finite and positive for convexity")
    second_derivative = math.fsum(
        cash_flow.total_amount
        * period_count
        * (period_count + 1.0)
        / (bond.frequency**2)
        * periodic_base ** (-period_count - 2.0)
        for cash_flow, period_count in zip(cash_flows, period_counts, strict=True)
    )
    return second_derivative / dirty_price


def dv01(bond: FixedRateBond, settlement_date: date, ytm: float) -> float:
    """Calculate positive analytical dirty-price DV01, in currency per bp.

    Args:
        bond: Bond terms; returned DV01 uses the currency scale of face value.
        settlement_date: Calendar settlement date before maturity.
        ytm: Decimal nominal annual yield compounded at coupon frequency.

    Returns:
        Positive absolute first-order sensitivity in currency per basis point:
        ``-d(dirty_price)/d(ytm) * 0.0001``. It is the approximate amount
        gained by a long ordinary bond when YTM falls by 1 bp. One bp is
        explicitly ``0.0001`` in decimal yield units.
    """
    dirty_price = dirty_price_from_ytm(bond, settlement_date, ytm)
    return modified_duration(bond, settlement_date, ytm) * dirty_price * BASIS_POINT


def _regular_schedule_dates(
    accrual_start_date: date, maturity_date: date, frequency: int
) -> tuple[date, ...]:
    """Build regular dates backwards from maturity and reject stubs."""
    months_per_period = 12 // frequency
    maturity_is_month_end = (
        maturity_date.day
        == calendar.monthrange(maturity_date.year, maturity_date.month)[1]
    )
    roll_day = maturity_date.day
    descending_dates = [maturity_date]
    periods = 0
    while descending_dates[-1] > accrual_start_date:
        periods += 1
        candidate = _shift_months(
            maturity_date,
            -(periods * months_per_period),
            roll_day=roll_day,
            use_month_end=maturity_is_month_end,
        )
        descending_dates.append(candidate)
    if descending_dates[-1] != accrual_start_date:
        raise ValueError(
            "accrual_start_date does not lie on the regular schedule implied by "
            "maturity_date and frequency; stub coupon periods are not supported"
        )
    return tuple(reversed(descending_dates))


def _shift_months(
    anchor_date: date, months: int, *, roll_day: int, use_month_end: bool
) -> date:
    """Shift from an anchor without accumulating short-month date drift."""
    month_index = anchor_date.year * 12 + (anchor_date.month - 1) + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    last_day = calendar.monthrange(year, month)[1]
    day = last_day if use_month_end else min(roll_day, last_day)
    return date(year, month, day)


def _future_cash_flows_and_period_counts(
    bond: FixedRateBond, settlement_date: date
) -> tuple[tuple[BondCashFlow, ...], tuple[float, ...]]:
    """Return entitled cash flows and fractional YTM discount exponents."""
    _validate_settlement(bond, settlement_date, allow_maturity=False)
    all_cash_flows = generate_fixed_rate_cash_flows(bond)
    future_cash_flows = tuple(
        cash_flow
        for cash_flow in all_cash_flows
        if cash_flow.payment_date > settlement_date
    )
    if not future_cash_flows:
        raise ValueError("settlement_date must precede at least one payment date")

    first_cash_flow = future_cash_flows[0]
    days_in_period = (
        first_cash_flow.accrual_end_date - first_cash_flow.accrual_start_date
    ).days
    days_remaining = (first_cash_flow.payment_date - settlement_date).days
    first_period_count = days_remaining / days_in_period
    period_counts = tuple(
        first_period_count + index for index in range(len(future_cash_flows))
    )
    return future_cash_flows, period_counts


def _find_ytm_bracket(
    bond: FixedRateBond, settlement_date: date, target_dirty_price: float
) -> tuple[float, float]:
    """Expand a bracket within the positive-discount-base yield domain."""
    lower = -0.5 * bond.frequency
    upper = max(0.10, bond.coupon_rate * 2.0)

    def residual(candidate_ytm: float) -> float:
        return dirty_price_from_ytm(bond, settlement_date, candidate_ytm) - (
            target_dirty_price
        )

    lower_residual = residual(lower)
    for _ in range(16):
        if lower_residual >= 0.0:
            break
        distance_to_boundary = lower + bond.frequency
        lower = -bond.frequency + distance_to_boundary / 10.0
        lower_residual = residual(lower)
    else:
        raise RuntimeError(
            "could not bracket YTM near the lower admissible yield boundary"
        )

    upper_residual = residual(upper)
    for _ in range(64):
        if upper_residual <= 0.0:
            break
        upper = upper * 2.0 + 0.10
        upper_residual = residual(upper)
    else:
        raise RuntimeError("could not bracket YTM at a finite upper yield")

    if not _opposite_signs_or_zero(lower_residual, upper_residual):
        raise RuntimeError(
            "could not find a YTM bracket whose price residual changes sign"
        )
    return lower, upper


def _validate_ytm_bracket(
    bracket: tuple[float, float], frequency: int
) -> tuple[float, float]:
    """Validate a caller-supplied decimal-yield bracket."""
    if not isinstance(bracket, tuple) or len(bracket) != 2:
        raise TypeError("bracket must be a two-element tuple of decimal yields")
    lower, upper = bracket
    _validate_ytm(lower, frequency)
    _validate_ytm(upper, frequency)
    if lower >= upper:
        raise ValueError("YTM bracket lower endpoint must be below upper endpoint")
    return lower, upper


def _opposite_signs_or_zero(first: float, second: float) -> bool:
    """Check a bracket without multiplying potentially huge residuals."""
    return (
        first == 0.0
        or second == 0.0
        or (first < 0.0 < second)
        or (second < 0.0 < first)
    )


def _validate_settlement(
    bond: FixedRateBond, settlement_date: date, *, allow_maturity: bool
) -> None:
    """Validate settlement against the supported bond life."""
    if not isinstance(bond, FixedRateBond):
        raise TypeError("bond must be a FixedRateBond")
    _validate_date(settlement_date, "settlement_date")
    if settlement_date < bond.accrual_start_date:
        raise ValueError("settlement_date must not precede accrual_start_date")
    if allow_maturity:
        if settlement_date > bond.maturity_date:
            raise ValueError("settlement_date must not be after maturity_date")
    elif settlement_date >= bond.maturity_date:
        raise ValueError("settlement_date must be before maturity_date")


def _validate_frequency(frequency: int) -> None:
    """Validate supported payments per year."""
    if isinstance(frequency, bool) or not isinstance(frequency, int):
        raise TypeError("frequency must be an integer number of payments per year")
    if frequency not in (1, 2):
        raise ValueError("frequency must be 1 (annual) or 2 (semiannual)")


def _validate_ytm(ytm: float, frequency: int) -> None:
    """Validate decimal nominal annual YTM and its periodic base."""
    _validate_finite_number(ytm, "ytm")
    if ytm <= -frequency:
        raise ValueError(
            "ytm must exceed -frequency so that 1 + ytm / frequency is positive"
        )


def _validate_finite_number(value: float, name: str) -> None:
    """Reject booleans, non-real objects, NaN, and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_date(value: date, name: str) -> None:
    """Require an unambiguous date rather than a date-time."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date without a time component")


__all__ = [
    "BASIS_POINT",
    "BondCashFlow",
    "CouponPeriod",
    "DayCountConvention",
    "FixedRateBond",
    "PriceType",
    "accrued_interest",
    "clean_price_from_ytm",
    "convexity",
    "day_count_fraction",
    "dirty_price_from_ytm",
    "dv01",
    "generate_coupon_schedule",
    "generate_fixed_rate_cash_flows",
    "macaulay_duration",
    "modified_duration",
    "price_from_ytm",
    "yield_to_maturity",
]
