"""Spot-curve bond pricing and transparent interest-rate risk scenarios.

Rates and shocks are decimal annual values internally. Public scenario inputs
whose names end in ``_bp`` are basis points and are converted exactly once via
``1 bp = 0.0001``. All prices and P&L values are currency amounts for the
supplied bond face value.

Curve pricing is intentionally separate from conventional YTM pricing. This
module discounts every future cash flow at its interpolated zero rate; it never
solves for or applies one yield to all cash flows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd

from fixed_income.bonds import (
    BASIS_POINT,
    FixedRateBond,
    PriceType,
    accrued_interest,
    day_count_fraction,
    generate_fixed_rate_cash_flows,
)
from fixed_income.curves import (
    CompoundingConvention,
    CurveRepresentation,
    DiscountedCashFlows,
    ExtrapolationPolicy,
    YieldCurve,
    discount_cash_flows,
    interpolate_zero_rates,
)

DEFAULT_PARALLEL_SHOCKS_BP = (-100.0, -50.0, -25.0, 25.0, 50.0, 100.0)
"""Default parallel zero-curve shocks, in basis points."""

DEFAULT_KEY_MATURITIES_YEARS = (2.0, 5.0, 10.0, 30.0)
"""Default key-rate tenors, in years."""


class CurveShapeScenario(StrEnum):
    """Named direction of a continuous short-versus-long curve shock."""

    STEEPENER = "steepener"
    FLATTENER = "flattener"


@dataclass(frozen=True)
class ShapeScenarioParameters:
    """Parameters for the default piecewise-linear steepener shock.

    Args:
        short_maturity_years: Short-end anchor maturity in years.
        pivot_maturity_years: Zero-shock pivot maturity in years.
        long_maturity_years: Long-end anchor maturity in years.
        short_end_shock_bp: Signed steepener shock at the short anchor, in bp.
        long_end_shock_bp: Signed steepener shock at the long anchor, in bp.

    Notes:
        The default steepener is -25 bp at 2Y, zero at the 10Y pivot, and
        +25 bp at 30Y. Linear interpolation in maturity is used between the
        three anchors and endpoint shocks are held constant beyond them. A
        flattener is the exact sign reversal. Positive shocks raise zero rates;
        negative shocks lower them. This is a versioned analytical scenario,
        not a claim about a universal market convention.
    """

    short_maturity_years: float = 2.0
    pivot_maturity_years: float = 10.0
    long_maturity_years: float = 30.0
    short_end_shock_bp: float = -25.0
    long_end_shock_bp: float = 25.0

    def __post_init__(self) -> None:
        """Validate maturity anchors in years and shocks in basis points."""
        values = (
            self.short_maturity_years,
            self.pivot_maturity_years,
            self.long_maturity_years,
            self.short_end_shock_bp,
            self.long_end_shock_bp,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("shape-scenario parameters must be finite numbers")
        if not (
            0.0
            < self.short_maturity_years
            < self.pivot_maturity_years
            < self.long_maturity_years
        ):
            raise ValueError(
                "shape anchors must be positive and ordered short < pivot < long"
            )
        if not self.short_end_shock_bp < self.long_end_shock_bp:
            raise ValueError(
                "a steepener requires short_end_shock_bp < long_end_shock_bp"
            )


DEFAULT_SHAPE_PARAMETERS = ShapeScenarioParameters()
"""Default 2Y/10Y/30Y steepener anchors and signed shocks."""


@dataclass(frozen=True)
class CurvePriceResult:
    """Auditable spot-curve price components, all in currency units.

    ``dirty_price_currency`` is the discounted value of future contractual
    cash flows. ``clean_price_currency`` equals dirty price minus
    ``accrued_interest_currency``. The embedded cash-flow result contains times
    in years, decimal zero rates, dimensionless discount factors, and currency
    present values.
    """

    dirty_price_currency: float
    clean_price_currency: float
    accrued_interest_currency: float
    discounted_cash_flows: DiscountedCashFlows


@dataclass(frozen=True)
class CurveRiskMeasures:
    """Parallel spot-curve sensitivity of dirty price.

    Duration is years per unit decimal parallel zero-rate change, convexity is
    years squared, and DV01 is positive currency per bp for an ordinary long
    fixed-rate bond. ``up_price_currency`` and ``down_price_currency`` use the
    exact finite-difference bump recorded in ``bump_size_bp``.
    """

    base_price_currency: float
    effective_duration_years: float
    effective_convexity_years_squared: float
    dv01_currency_per_bp: float
    bump_size_bp: float
    up_price_currency: float
    down_price_currency: float


def curve_price_details(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> CurvePriceResult:
    """Price a fixed-rate bond from an explicit spot/zero curve.

    Args:
        bond: Fixed-rate bond; cash flows and prices are currency amounts.
        settlement_date: Curve-time origin and bond settlement calendar date.
        zero_curve: Decimal annual zero-rate curve with explicit compounding.
        extrapolation: Policy for cash-flow maturities outside curve nodes.

    Returns:
        Structured clean price, dirty price, accrued interest, and individual
        discounted cash flows in currency units.

    Notes:
        Payment times use the bond's Actual/Actual coupon-period convention.
        Each cash flow is discounted with the zero rate interpolated at its own
        maturity. If the curve carries a valuation date, it must equal the
        settlement date. This is not YTM pricing.
    """
    _validate_curve_date(zero_curve, settlement_date)
    times, amounts = _future_cash_flow_inputs(bond, settlement_date)
    discounted = discount_cash_flows(
        times, amounts, zero_curve, extrapolation=extrapolation
    )
    dirty_price = discounted.total_present_value
    accrued = accrued_interest(bond, settlement_date)
    return CurvePriceResult(
        dirty_price_currency=dirty_price,
        clean_price_currency=dirty_price - accrued,
        accrued_interest_currency=accrued,
        discounted_cash_flows=discounted,
    )


def price_from_curve(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    price_type: PriceType | str,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> float:
    """Return an explicitly selected clean or dirty spot-curve price.

    Args:
        bond: Fixed-rate bond; face value and result are currency amounts.
        settlement_date: Curve-time origin and bond settlement calendar date.
        zero_curve: Decimal annual zero-rate curve used cash flow by cash flow.
        price_type: Explicitly ``"clean"`` or ``"dirty"``; no default exists.
        extrapolation: Policy for maturities outside the curve-node range.

    Returns:
        Selected clean or dirty currency price for the bond face value.

    Notes:
        This function cannot accept a YTM. Use
        :func:`fixed_income.bonds.price_from_ytm` for single-yield pricing.
    """
    try:
        normalized_price_type = PriceType(price_type)
    except ValueError as exc:
        raise ValueError("price_type must be explicitly 'clean' or 'dirty'") from exc
    result = curve_price_details(
        bond, settlement_date, zero_curve, extrapolation=extrapolation
    )
    if normalized_price_type is PriceType.DIRTY:
        return result.dirty_price_currency
    return result.clean_price_currency


def dirty_price_from_curve(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> float:
    """Return the spot-curve dirty price in currency units.

    ``zero_curve`` contains decimal annual zero rates and each future currency
    cash flow is discounted at its own maturity-specific interpolated rate.
    Cash flows on settlement are excluded. This function does not use YTM.
    """
    return curve_price_details(
        bond, settlement_date, zero_curve, extrapolation=extrapolation
    ).dirty_price_currency


def clean_price_from_curve(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> float:
    """Return the spot-curve clean price in currency units.

    The result is the dirty currency value of individually curve-discounted
    cash flows minus accrued interest in currency under the bond convention.
    This function does not use YTM.
    """
    return curve_price_details(
        bond, settlement_date, zero_curve, extrapolation=extrapolation
    ).clean_price_currency


def parallel_curve_shock(zero_curve: YieldCurve, shock_bp: float) -> YieldCurve:
    """Add one parallel shock to every zero-rate node.

    Args:
        zero_curve: Curve explicitly tagged as decimal annual zero rates.
        shock_bp: Signed shock in basis points; +1 bp means +0.0001.

    Returns:
        A new zero curve with every decimal node rate increased by
        ``shock_bp * 0.0001``. Curve metadata and compounding are retained.
    """
    _require_zero_curve(zero_curve)
    shock = _bp_to_decimal(shock_bp, "shock_bp")
    return _curve_with_values(
        zero_curve,
        np.asarray(zero_curve.values, dtype=float) + shock,
        source_suffix=f"parallel shock {float(shock_bp):+g} bp",
    )


def shape_shock_function(
    maturities_years: Sequence[float] | np.ndarray,
    scenario: CurveShapeScenario | str,
    *,
    parameters: ShapeScenarioParameters = DEFAULT_SHAPE_PARAMETERS,
) -> np.ndarray:
    """Evaluate a continuous steepener or flattener shock in decimal rates.

    Args:
        maturities_years: Non-negative target maturities in years.
        scenario: ``"steepener"`` or ``"flattener"``.
        parameters: Anchor maturities in years and signed steepener shocks in
            bp. The pivot shock is exactly zero.

    Returns:
        Decimal annual zero-rate shocks. Positive values raise rates. Linear
        maturity interpolation joins short, pivot, and long anchors; the short
        and long endpoint values are constant outside their anchors. A
        flattener is the negative of the corresponding steepener.
    """
    maturities = _finite_1d(maturities_years, "maturities_years")
    if np.any(maturities < 0.0):
        raise ValueError("maturities_years must be non-negative")
    if not isinstance(parameters, ShapeScenarioParameters):
        raise TypeError("parameters must be ShapeScenarioParameters")
    try:
        normalized_scenario = CurveShapeScenario(scenario)
    except ValueError as exc:
        raise ValueError("scenario must be 'steepener' or 'flattener'") from exc
    anchor_shocks_bp = np.array(
        [parameters.short_end_shock_bp, 0.0, parameters.long_end_shock_bp]
    )
    shock_bp = np.interp(
        maturities,
        [
            parameters.short_maturity_years,
            parameters.pivot_maturity_years,
            parameters.long_maturity_years,
        ],
        anchor_shocks_bp,
    )
    direction = 1.0 if normalized_scenario is CurveShapeScenario.STEEPENER else -1.0
    return direction * shock_bp * BASIS_POINT


def curve_shape_shock(
    zero_curve: YieldCurve,
    scenario: CurveShapeScenario | str,
    *,
    parameters: ShapeScenarioParameters = DEFAULT_SHAPE_PARAMETERS,
) -> YieldCurve:
    """Apply a continuous shaped shock to zero-curve nodes.

    Node maturities are years and node shocks are decimal annual rates. The
    returned curve retains the base curve's compounding and valuation date.
    See :func:`shape_shock_function` for anchors and sign convention.
    """
    _require_zero_curve(zero_curve)
    normalized_scenario = CurveShapeScenario(scenario)
    shocks = shape_shock_function(
        zero_curve.maturities_years, normalized_scenario, parameters=parameters
    )
    return _curve_with_values(
        zero_curve,
        np.asarray(zero_curve.values) + shocks,
        source_suffix=f"{normalized_scenario.value} shaped shock",
    )


def curve_risk_measures(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    bump_size_bp: float = 1.0,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> CurveRiskMeasures:
    """Calculate parallel effective duration, convexity, and curve DV01.

    Args:
        bond: Fixed-rate bond; prices and DV01 use its currency face scale.
        settlement_date: Curve-time origin and settlement calendar date.
        zero_curve: Decimal annual zero curve.
        bump_size_bp: Positive central-difference bump in bp; default is 1 bp,
            exactly 0.0001 in decimal rate units.
        extrapolation: Curve-node extrapolation policy for bond cash flows.

    Returns:
        Dirty-price effective duration in years, effective convexity in years
        squared, and positive DV01 in currency per 1 bp. Up/down prices are
        full spot-curve revaluations.
    """
    bump = _positive_bp_to_decimal(bump_size_bp, "bump_size_bp")
    base = price_from_curve(
        bond,
        settlement_date,
        zero_curve,
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    up = price_from_curve(
        bond,
        settlement_date,
        parallel_curve_shock(zero_curve, bump_size_bp),
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    down = price_from_curve(
        bond,
        settlement_date,
        parallel_curve_shock(zero_curve, -bump_size_bp),
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    first_derivative = (up - down) / (2.0 * bump)
    second_derivative = (up - 2.0 * base + down) / (bump * bump)
    return CurveRiskMeasures(
        base_price_currency=base,
        effective_duration_years=-first_derivative / base,
        effective_convexity_years_squared=second_derivative / base,
        dv01_currency_per_bp=-first_derivative * BASIS_POINT,
        bump_size_bp=float(bump_size_bp),
        up_price_currency=up,
        down_price_currency=down,
    )


def bond_risk_report(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    parallel_shocks_bp: Sequence[float] = DEFAULT_PARALLEL_SHOCKS_BP,
    include_shape_scenarios: bool = True,
    shape_parameters: ShapeScenarioParameters = DEFAULT_SHAPE_PARAMETERS,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> pd.DataFrame:
    """Build a full-revaluation curve-scenario report for one bond.

    Args:
        bond: Fixed-rate bond; all prices and P&L are currency amounts.
        settlement_date: Curve-time origin and bond settlement date.
        zero_curve: Decimal annual zero curve.
        parallel_shocks_bp: Signed parallel shocks in bp. Defaults to -100,
            -50, -25, +25, +50, and +100 bp.
        include_shape_scenarios: Include parameterized steepener/flattener rows.
        shape_parameters: Shape anchor maturities in years and shocks in bp.
        extrapolation: Curve-node extrapolation policy for cash-flow pricing.

    Returns:
        DataFrame with dirty base/shocked prices, exact P&L, first-order
        duration/DV01 P&L, duration-plus-convexity P&L, and both approximation
        errors in currency. Error is defined as approximation minus exact P&L.

    Notes:
        Full revaluation shocks zero rates while holding contractual cash flows,
        accrued interest, and spreads fixed. For shaped scenarios, first- and
        second-order terms are cash-flow-level directional derivatives, not a
        parallel-duration proxy. Calculation functions do not write files;
        callers may save this table as ``outputs/bond_risk_report.csv``.
    """
    _require_zero_curve(zero_curve)
    shocks = _finite_1d(parallel_shocks_bp, "parallel_shocks_bp")
    base_details = curve_price_details(
        bond, settlement_date, zero_curve, extrapolation=extrapolation
    )
    measures = curve_risk_measures(
        bond, settlement_date, zero_curve, extrapolation=extrapolation
    )
    rows: list[dict[str, object]] = []
    for shock_bp in shocks:
        shocked_curve = parallel_curve_shock(zero_curve, float(shock_bp))
        rows.append(
            _scenario_row(
                scenario=f"parallel_{shock_bp:+g}bp",
                scenario_type="parallel",
                bond=bond,
                settlement_date=settlement_date,
                base_curve=zero_curve,
                shocked_curve=shocked_curve,
                base_details=base_details,
                measures=measures,
                extrapolation=extrapolation,
                parallel_shock_bp=float(shock_bp),
                shape_parameters=None,
            )
        )
    if include_shape_scenarios:
        for scenario in CurveShapeScenario:
            shocked_curve = curve_shape_shock(
                zero_curve, scenario, parameters=shape_parameters
            )
            rows.append(
                _scenario_row(
                    scenario=scenario.value,
                    scenario_type="shape",
                    bond=bond,
                    settlement_date=settlement_date,
                    base_curve=zero_curve,
                    shocked_curve=shocked_curve,
                    base_details=base_details,
                    measures=measures,
                    extrapolation=extrapolation,
                    parallel_shock_bp=math.nan,
                    shape_parameters=shape_parameters,
                )
            )
    return pd.DataFrame(rows)


def key_rate_bump_function(
    maturities_years: Sequence[float] | np.ndarray,
    key_maturity_years: float,
    *,
    key_maturities_years: Sequence[float] = DEFAULT_KEY_MATURITIES_YEARS,
    bump_size_bp: float = 1.0,
) -> np.ndarray:
    """Evaluate one localized piecewise-linear key-rate bump.

    Args:
        maturities_years: Target maturities in years.
        key_maturity_years: Centre key tenor in years.
        key_maturities_years: Increasing positive key tenors in years.
        bump_size_bp: Positive peak bump in bp; 1 bp is exactly 0.0001.

    Returns:
        Decimal annual rate bumps. Interior hats peak at the selected key and
        decline linearly to zero at adjacent keys, remaining zero outside that
        neighbouring region. The first and last keys use one-sided shoulders
        beyond the outer keys. Consequently all key weights sum exactly to one
        across the covered maturity domain, allowing first-order key-rate DV01s
        to reconcile with a parallel DV01.
    """
    maturities = _finite_1d(maturities_years, "maturities_years")
    if np.any(maturities < 0.0):
        raise ValueError("maturities_years must be non-negative")
    keys = _validate_key_maturities(key_maturities_years)
    key = _finite_number(key_maturity_years, "key_maturity_years")
    matches = np.flatnonzero(np.isclose(keys, key, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError("key_maturity_years must exactly identify one key tenor")
    bump = _positive_bp_to_decimal(bump_size_bp, "bump_size_bp")
    index = int(matches[0])
    weights = np.zeros_like(maturities)
    if index == 0:
        weights[maturities <= key] = 1.0
    else:
        left = keys[index - 1]
        mask = (maturities >= left) & (maturities <= key)
        weights[mask] = (maturities[mask] - left) / (key - left)
    if index == len(keys) - 1:
        weights[maturities >= key] = 1.0
    else:
        right = keys[index + 1]
        mask = (maturities >= key) & (maturities <= right)
        weights[mask] = (right - maturities[mask]) / (right - key)
    return weights * bump


def key_rate_dv01_report(
    bond: FixedRateBond,
    settlement_date: date,
    zero_curve: YieldCurve,
    *,
    key_maturities_years: Sequence[float] = DEFAULT_KEY_MATURITIES_YEARS,
    bump_size_bp: float = 1.0,
    extrapolation: ExtrapolationPolicy | str = ExtrapolationPolicy.RAISE,
) -> pd.DataFrame:
    """Calculate central finite-difference key-rate DV01s for one bond.

    Args:
        bond: Fixed-rate bond; prices and DV01 use its currency face scale.
        settlement_date: Curve-time origin and settlement calendar date.
        zero_curve: Decimal annual zero curve spanning every requested key.
        key_maturities_years: Increasing key tenors in years; defaults to 2Y,
            5Y, 10Y, and 30Y.
        bump_size_bp: Positive peak of each localized bump in bp. Each reported
            DV01 is normalized to currency per 1 bp.
        extrapolation: Curve-node extrapolation policy for cash-flow pricing.

    Returns:
        DataFrame containing base/up/down dirty prices, positive central key-rate
        DV01 in currency per bp, exact peak bump units, supports, and aggregate
        reconciliation against central parallel DV01. Positive DV01 means a
        long position gains when that localized rate falls.

    Notes:
        Interior bumps are triangular and zero outside adjacent keys. Edge
        bumps have one-sided shoulders to the curve boundary so the basis sums
        to a parallel bump. The curve is augmented at key tenors before
        bumping, ensuring the requested peak is applied exactly even when a key
        was not an original node. Nothing is written automatically; callers may
        save the result as ``outputs/key_rate_dv01.csv``.
    """
    _require_zero_curve(zero_curve)
    keys = _validate_key_maturities(key_maturities_years)
    bump_decimal = _positive_bp_to_decimal(bump_size_bp, "bump_size_bp")
    curve_min = zero_curve.maturities_years[0]
    curve_max = zero_curve.maturities_years[-1]
    if keys[0] < curve_min or keys[-1] > curve_max:
        raise ValueError("zero_curve must span every requested key maturity")
    working_curve = _augment_curve_at_maturities(zero_curve, keys)
    node_maturities = np.asarray(working_curve.maturities_years)
    base_values = np.asarray(working_curve.values)
    base_price = price_from_curve(
        bond,
        settlement_date,
        working_curve,
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    rows: list[dict[str, float | str]] = []
    for index, key in enumerate(keys):
        bump_vector = key_rate_bump_function(
            node_maturities,
            float(key),
            key_maturities_years=keys,
            bump_size_bp=bump_size_bp,
        )
        up_curve = _curve_with_values(
            working_curve,
            base_values + bump_vector,
            source_suffix=f"{key:g}Y key rate +{bump_size_bp:g} bp",
        )
        down_curve = _curve_with_values(
            working_curve,
            base_values - bump_vector,
            source_suffix=f"{key:g}Y key rate -{bump_size_bp:g} bp",
        )
        up_price = price_from_curve(
            bond,
            settlement_date,
            up_curve,
            price_type=PriceType.DIRTY,
            extrapolation=extrapolation,
        )
        down_price = price_from_curve(
            bond,
            settlement_date,
            down_curve,
            price_type=PriceType.DIRTY,
            extrapolation=extrapolation,
        )
        support_left = curve_min if index == 0 else float(keys[index - 1])
        support_right = curve_max if index == len(keys) - 1 else float(keys[index + 1])
        central_change = (down_price - up_price) / 2.0
        rows.append(
            {
                "key_tenor": f"{key:g}Y",
                "key_maturity_years": float(key),
                "bump_size_bp": float(bump_size_bp),
                "bump_size_decimal": bump_decimal,
                "support_left_years": support_left,
                "support_right_years": support_right,
                "base_dirty_price_currency": base_price,
                "up_dirty_price_currency": up_price,
                "down_dirty_price_currency": down_price,
                "central_price_change_currency": central_change,
                "key_rate_dv01_currency_per_bp": central_change / float(bump_size_bp),
                "price_basis": "dirty_currency",
                "dv01_sign_convention": (
                    "positive = long gains for a 1 bp localized rate fall"
                ),
            }
        )
    parallel_up = price_from_curve(
        bond,
        settlement_date,
        parallel_curve_shock(working_curve, bump_size_bp),
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    parallel_down = price_from_curve(
        bond,
        settlement_date,
        parallel_curve_shock(working_curve, -bump_size_bp),
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    parallel_dv01 = (parallel_down - parallel_up) / (2.0 * float(bump_size_bp))
    aggregate = math.fsum(float(row["key_rate_dv01_currency_per_bp"]) for row in rows)
    reconciliation_error = aggregate - parallel_dv01
    for row in rows:
        row["aggregate_key_rate_dv01_currency_per_bp"] = aggregate
        row["parallel_dv01_currency_per_bp"] = parallel_dv01
        row["reconciliation_error_currency_per_bp"] = reconciliation_error
    return pd.DataFrame(rows)


def _scenario_row(
    *,
    scenario: str,
    scenario_type: str,
    bond: FixedRateBond,
    settlement_date: date,
    base_curve: YieldCurve,
    shocked_curve: YieldCurve,
    base_details: CurvePriceResult,
    measures: CurveRiskMeasures,
    extrapolation: ExtrapolationPolicy | str,
    parallel_shock_bp: float,
    shape_parameters: ShapeScenarioParameters | None,
) -> dict[str, object]:
    shocked_price = price_from_curve(
        bond,
        settlement_date,
        shocked_curve,
        price_type=PriceType.DIRTY,
        extrapolation=extrapolation,
    )
    first_order, second_order = _curve_change_approximations(
        base_details.discounted_cash_flows,
        base_curve,
        shocked_curve,
        extrapolation=extrapolation,
    )
    exact_pnl = shocked_price - base_details.dirty_price_currency
    duration_convexity_pnl = first_order + second_order
    return {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "parallel_shock_bp": parallel_shock_bp,
        "short_end_shock_bp": (
            math.nan
            if shape_parameters is None
            else shape_parameters.short_end_shock_bp
            * (-1.0 if scenario == CurveShapeScenario.FLATTENER else 1.0)
        ),
        "long_end_shock_bp": (
            math.nan
            if shape_parameters is None
            else shape_parameters.long_end_shock_bp
            * (-1.0 if scenario == CurveShapeScenario.FLATTENER else 1.0)
        ),
        "pivot_maturity_years": (
            math.nan
            if shape_parameters is None
            else shape_parameters.pivot_maturity_years
        ),
        "base_dirty_price_currency": base_details.dirty_price_currency,
        "shocked_dirty_price_currency": shocked_price,
        "exact_pnl_currency": exact_pnl,
        "first_order_pnl_currency": first_order,
        "duration_convexity_pnl_currency": duration_convexity_pnl,
        "first_order_approximation_error_currency": first_order - exact_pnl,
        "approximation_error_currency": duration_convexity_pnl - exact_pnl,
        "base_parallel_dv01_currency_per_bp": measures.dv01_currency_per_bp,
        "base_effective_duration_years": measures.effective_duration_years,
        "base_effective_convexity_years_squared": (
            measures.effective_convexity_years_squared
        ),
        "price_basis": "dirty_currency",
        "shock_sign_convention": "positive bp raises zero rates",
        "pnl_sign_convention": "shocked price minus base price for a long bond",
    }


def _curve_change_approximations(
    discounted: DiscountedCashFlows,
    base_curve: YieldCurve,
    shocked_curve: YieldCurve,
    *,
    extrapolation: ExtrapolationPolicy | str,
) -> tuple[float, float]:
    times = discounted.times_years
    base_rates = discounted.interpolated_zero_rates
    shocked_rates = interpolate_zero_rates(
        shocked_curve.maturities_years,
        shocked_curve.values,
        times,
        extrapolation=extrapolation,
    )
    rate_changes = shocked_rates - base_rates
    present_values = discounted.present_values
    convention = base_curve.compounding
    if convention is CompoundingConvention.CONTINUOUS:
        first_derivatives = -times * present_values
        second_derivatives = times * times * present_values
    elif convention is CompoundingConvention.SIMPLE:
        bases = 1.0 + base_rates * times
        first_derivatives = -times * present_values / bases
        second_derivatives = 2.0 * times * times * present_values / (bases * bases)
    else:
        frequency = base_curve.periodic_frequency
        assert frequency is not None
        bases = 1.0 + base_rates / frequency
        first_derivatives = -times * present_values / bases
        second_derivatives = (
            times * (times + 1.0 / frequency) * present_values / (bases * bases)
        )
    first_order = float(np.sum(first_derivatives * rate_changes))
    second_order = float(0.5 * np.sum(second_derivatives * rate_changes * rate_changes))
    return first_order, second_order


def _future_cash_flow_inputs(
    bond: FixedRateBond, settlement_date: date
) -> tuple[np.ndarray, np.ndarray]:
    # accrued_interest performs the package's public bond/settlement validation.
    accrued_interest(bond, settlement_date)
    future = tuple(
        cash_flow
        for cash_flow in generate_fixed_rate_cash_flows(bond)
        if cash_flow.payment_date > settlement_date
    )
    if not future:
        raise ValueError("settlement_date must be before the final bond cash flow")
    first = future[0]
    first_time = day_count_fraction(
        settlement_date,
        first.accrual_end_date,
        reference_start_date=first.accrual_start_date,
        reference_end_date=first.accrual_end_date,
        frequency=bond.frequency,
        convention=bond.day_count,
    )
    times: list[float] = []
    elapsed = first_time
    for index, cash_flow in enumerate(future):
        if index > 0:
            elapsed += cash_flow.accrual_fraction
        times.append(elapsed)
    amounts = [cash_flow.total_amount for cash_flow in future]
    return np.asarray(times), np.asarray(amounts)


def _augment_curve_at_maturities(
    zero_curve: YieldCurve, extra_maturities_years: np.ndarray
) -> YieldCurve:
    base_maturities = np.asarray(zero_curve.maturities_years)
    combined = np.unique(np.concatenate((base_maturities, extra_maturities_years)))
    values = interpolate_zero_rates(
        zero_curve.maturities_years,
        zero_curve.values,
        combined,
        extrapolation=ExtrapolationPolicy.RAISE,
    )
    return YieldCurve(
        maturities_years=tuple(map(float, combined)),
        values=tuple(map(float, values)),
        representation=zero_curve.representation,
        valuation_date=zero_curve.valuation_date,
        compounding=zero_curve.compounding,
        periodic_frequency=zero_curve.periodic_frequency,
        source=_source_with_suffix(zero_curve.source, "key-rate basis nodes"),
    )


def _curve_with_values(
    zero_curve: YieldCurve,
    values: Sequence[float] | np.ndarray,
    *,
    source_suffix: str,
) -> YieldCurve:
    shifted = _finite_1d(values, "shifted curve values")
    if shifted.size != len(zero_curve.values):
        raise ValueError("shifted curve values must match the base curve nodes")
    if np.any(np.abs(shifted) > 1.0):
        raise ValueError("shocked zero rates exceed decimal annual rate bounds")
    return YieldCurve(
        maturities_years=zero_curve.maturities_years,
        values=tuple(map(float, shifted)),
        representation=zero_curve.representation,
        valuation_date=zero_curve.valuation_date,
        compounding=zero_curve.compounding,
        periodic_frequency=zero_curve.periodic_frequency,
        source=_source_with_suffix(zero_curve.source, source_suffix),
    )


def _source_with_suffix(source: str, suffix: str) -> str:
    return f"{source}; {suffix}" if source else suffix


def _require_zero_curve(zero_curve: YieldCurve) -> None:
    if not isinstance(zero_curve, YieldCurve):
        raise TypeError("zero_curve must be a YieldCurve")
    if zero_curve.representation not in {
        CurveRepresentation.ZERO_RATE,
        CurveRepresentation.FITTED_ZERO_RATE,
    }:
        raise ValueError(
            "curve risk requires an explicitly tagged zero-rate curve; "
            f"received {zero_curve.representation.value!r}"
        )


def _validate_curve_date(zero_curve: YieldCurve, settlement_date: date) -> None:
    _require_zero_curve(zero_curve)
    if (
        zero_curve.valuation_date is not None
        and zero_curve.valuation_date != settlement_date
    ):
        raise ValueError("zero_curve valuation_date must equal settlement_date")


def _validate_key_maturities(
    key_maturities_years: Sequence[float],
) -> np.ndarray:
    keys = _finite_1d(key_maturities_years, "key_maturities_years")
    if keys.size < 2:
        raise ValueError("at least two key maturities are required")
    if np.any(keys <= 0.0) or np.any(np.diff(keys) <= 0.0):
        raise ValueError("key maturities must be positive, unique, and increasing")
    return keys


def _bp_to_decimal(value: float, name: str) -> float:
    return _finite_number(value, name) * BASIS_POINT


def _positive_bp_to_decimal(value: float, name: str) -> float:
    normalized = _bp_to_decimal(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive basis points")
    return normalized


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _finite_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


__all__ = [
    "DEFAULT_KEY_MATURITIES_YEARS",
    "DEFAULT_PARALLEL_SHOCKS_BP",
    "DEFAULT_SHAPE_PARAMETERS",
    "CurvePriceResult",
    "CurveRiskMeasures",
    "CurveShapeScenario",
    "ShapeScenarioParameters",
    "bond_risk_report",
    "clean_price_from_curve",
    "curve_price_details",
    "curve_risk_measures",
    "curve_shape_shock",
    "dirty_price_from_curve",
    "key_rate_bump_function",
    "key_rate_dv01_report",
    "parallel_curve_shock",
    "price_from_curve",
    "shape_shock_function",
]
