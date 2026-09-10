"""Look-ahead-safe curve residual, ranking, and butterfly analytics.

The residual convention in this module is always ``observed yield - fitted
yield``. Yields and residuals are decimal annual rates internally. A positive
residual therefore means that the observed yield is above the fitted yield and
is labelled ``cheap``; a negative residual is labelled ``rich``. ``Cheap`` and
``rich`` are yield-relative descriptions, not claims of arbitrage or executable
bond mispricing.

One basis point is exactly ``0.0001`` in decimal yield terms. Treasury
constant-maturity observations remain statistical yield series when used here;
fitting or ranking them does not turn them into zero-coupon rates or tradable
bonds.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from fixed_income.pca import maturity_in_years

BASIS_POINT_DECIMAL = 0.0001
"""One basis point expressed as a decimal annual yield."""

_BUTTERFLY_MATURITIES = (2.0, 5.0, 10.0)
_BUTTERFLY_LEGS = ("2Y wing", "5Y belly", "10Y wing")


class RelativeValueLabel(StrEnum):
    """Yield-relative classification under the observed-minus-fitted convention."""

    RICH = "rich"
    ON_CURVE = "on-curve"
    CHEAP = "cheap"


class ZeroStandardDeviationPolicy(StrEnum):
    """Handling for a historical rolling standard deviation of zero."""

    NAN = "nan"
    ZERO = "zero"
    RAISE = "raise"


@dataclass(frozen=True)
class RollingZScoreResult:
    """Look-ahead-safe rolling normalization result.

    All inputs, means, and standard deviations are decimal yield residuals;
    z-scores are dimensionless. At date ``t``, ``historical_mean_decimal`` and
    ``historical_std_decimal`` use at most the preceding ``lookback`` rows and
    exclude row ``t``. Missing observations are never filled.

    ``window_start_dates`` and ``window_end_dates`` identify the first and last
    non-missing observations used for each estimate. They are ``NaT`` until
    ``min_observations`` are available.
    """

    residuals_decimal: pd.DataFrame
    historical_mean_decimal: pd.DataFrame
    historical_std_decimal: pd.DataFrame
    z_scores: pd.DataFrame
    valid_observation: pd.DataFrame
    window_start_dates: pd.DataFrame
    window_end_dates: pd.DataFrame
    lookback: int
    min_observations: int
    ddof: int
    zero_std_policy: ZeroStandardDeviationPolicy

    @property
    def rolling_mean(self) -> pd.DataFrame:
        """Return trailing means through ``t-1`` in decimal yield units."""
        return self.historical_mean_decimal

    @property
    def rolling_std(self) -> pd.DataFrame:
        """Return trailing standard deviations through ``t-1`` in decimals."""
        return self.historical_std_decimal

    @property
    def zscore(self) -> pd.DataFrame:
        """Return dimensionless z-scores as a compatibility alias."""
        return self.z_scores


@dataclass(frozen=True)
class ButterflyAnalytics:
    """A 2Y/5Y/10Y yield butterfly and its DV01-neutral leg report.

    ``yield_butterfly_decimal`` is ``2*y_5Y - y_2Y - y_10Y`` in decimal annual
    yield units. ``yield_butterfly_bp`` divides that value by exactly
    ``0.0001``. The report's notionals are signed currency face amounts and
    its DV01 values are currency per bp.
    """

    yield_butterfly_decimal: float
    yield_butterfly_bp: float
    legs: pd.DataFrame
    aggregate_dv01_currency_per_bp: float


def calculate_curve_residuals(
    observed_yields: pd.DataFrame, fitted_yields: pd.DataFrame
) -> pd.DataFrame:
    """Return ``observed - fitted`` residuals in decimal annual yield units.

    Both arguments must be date-by-tenor DataFrames with identical dates and
    maturity labels. Column order may differ and is aligned to the observed
    panel. A missing observed or fitted value produces an explicit ``NaN``
    residual; no interpolation or filling is performed.
    """
    observed = _validate_yield_panel(observed_yields, "observed_yields")
    fitted = _validate_yield_panel(fitted_yields, "fitted_yields")
    if not observed.index.equals(fitted.index):
        raise ValueError("observed_yields and fitted_yields dates must match exactly")
    if set(observed.columns) != set(fitted.columns):
        raise ValueError(
            "observed_yields and fitted_yields tenor labels must match exactly"
        )
    fitted = fitted.loc[:, observed.columns]
    residuals = observed - fitted
    residuals.index.name = observed.index.name
    return residuals


def calculate_nss_residuals(
    observed_yields: pd.DataFrame, fitted_yields: pd.DataFrame
) -> pd.DataFrame:
    """Return NSS residuals as observed minus fitted decimal annual yields.

    This named entry point does not itself calibrate NSS; it aligns dated
    observed and already-fitted panels and delegates to
    :func:`calculate_curve_residuals`.
    """
    return calculate_curve_residuals(observed_yields, fitted_yields)


def residuals_to_basis_points(
    residuals_decimal: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """Convert decimal yield residuals to basis points by dividing by 0.0001."""
    if isinstance(residuals_decimal, (pd.Series, pd.DataFrame)):
        numeric = _numeric_pandas(residuals_decimal, "residuals_decimal")
        return numeric / BASIS_POINT_DECIMAL
    value = _finite_number(residuals_decimal, "residuals_decimal")
    return value / BASIS_POINT_DECIMAL


def classify_residual(
    residual_decimal: float, *, on_curve_tolerance_decimal: float = 0.0
) -> str:
    """Classify one decimal yield residual as rich, on-curve, or cheap.

    A residual below minus the non-negative decimal tolerance is ``rich``; one
    above plus the tolerance is ``cheap``; a value within it is ``on-curve``.
    Positive/cheap means observed yield exceeds fitted yield. Negative/rich
    means observed yield is below fitted yield.
    """
    residual = _finite_number(residual_decimal, "residual_decimal")
    tolerance = _non_negative_number(
        on_curve_tolerance_decimal, "on_curve_tolerance_decimal"
    )
    if residual > tolerance:
        return RelativeValueLabel.CHEAP.value
    if residual < -tolerance:
        return RelativeValueLabel.RICH.value
    return RelativeValueLabel.ON_CURVE.value


def classify_rich_cheap(
    residuals_decimal: float | pd.Series | pd.DataFrame,
    *,
    on_curve_tolerance_decimal: float = 0.0,
) -> str | pd.Series | pd.DataFrame:
    """Classify decimal residuals while preserving scalar or pandas shape.

    Missing pandas values remain missing. Labels use observed-minus-fitted:
    positive is ``cheap``, negative is ``rich``, and a value inside the stated
    decimal-yield tolerance is ``on-curve``.
    """
    tolerance = _non_negative_number(
        on_curve_tolerance_decimal, "on_curve_tolerance_decimal"
    )
    if not isinstance(residuals_decimal, (pd.Series, pd.DataFrame)):
        return classify_residual(
            residuals_decimal, on_curve_tolerance_decimal=tolerance
        )
    numeric = _numeric_pandas(residuals_decimal, "residuals_decimal")
    classified = numeric.astype("object")
    classified[numeric > tolerance] = RelativeValueLabel.CHEAP.value
    classified[numeric < -tolerance] = RelativeValueLabel.RICH.value
    classified[numeric.abs() <= tolerance] = RelativeValueLabel.ON_CURVE.value
    classified[numeric.isna()] = pd.NA
    return classified


def residual_report(
    observed_yields: pd.DataFrame,
    fitted_yields: pd.DataFrame,
    *,
    on_curve_tolerance_decimal: float = 0.0,
) -> pd.DataFrame:
    """Return a long dated residual table with decimal, bp, and classification.

    Observed yield, fitted yield, and residual columns are decimal annual rates;
    ``residual_bp`` is basis points. Classification follows the documented
    observed-minus-fitted convention.
    """
    residuals = calculate_curve_residuals(observed_yields, fitted_yields)
    observed = _validate_yield_panel(observed_yields, "observed_yields")
    fitted = _validate_yield_panel(fitted_yields, "fitted_yields").loc[
        :, observed.columns
    ]
    rows: list[dict[str, object]] = []
    stacked = residuals.stack(future_stack=True)
    for date, tenor in stacked.index:
        residual = residuals.loc[date, tenor]
        rows.append(
            {
                "date": date,
                "tenor": tenor,
                "maturity_years": maturity_in_years(tenor),
                "observed_yield_decimal": observed.loc[date, tenor],
                "fitted_yield_decimal": fitted.loc[date, tenor],
                "residual_decimal": residual,
                "residual_bp": (
                    residual / BASIS_POINT_DECIMAL if pd.notna(residual) else np.nan
                ),
                "classification": (
                    classify_residual(
                        float(residual),
                        on_curve_tolerance_decimal=on_curve_tolerance_decimal,
                    )
                    if pd.notna(residual)
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def rolling_z_scores(
    residuals_decimal: pd.Series | pd.DataFrame,
    *,
    lookback: int = 60,
    min_observations: int | None = None,
    ddof: int = 1,
    zero_std: ZeroStandardDeviationPolicy | str = ZeroStandardDeviationPolicy.NAN,
) -> RollingZScoreResult:
    """Calculate look-ahead-safe rolling z-scores for decimal residuals.

    At signal date ``t``, historical estimates are calculated as
    ``residuals.shift(1).rolling(lookback)``. Thus the current value and all
    future values are excluded. ``lookback`` is a number of prior rows, not
    calendar days. ``min_observations`` defaults to ``lookback`` and counts
    non-missing observations independently by tenor. ``ddof`` defaults to one
    (sample standard deviation) and must be smaller than ``min_observations``.

    If historical standard deviation is exactly zero, ``zero_std='nan'``
    leaves the signal unavailable, ``'zero'`` emits zero for a finite current
    residual, and ``'raise'`` rejects the calculation. Missing values are not
    filled or backfilled. Input residuals and estimates use decimal annual
    yield units; returned z-scores are dimensionless.
    """
    frame = _as_residual_frame(residuals_decimal)
    lookback = _positive_integer(lookback, "lookback")
    if min_observations is None:
        minimum = lookback
    else:
        minimum = _positive_integer(min_observations, "min_observations")
    if minimum > lookback:
        raise ValueError("min_observations cannot exceed lookback")
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be a non-negative integer")
    if ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if ddof >= minimum:
        raise ValueError("ddof must be smaller than min_observations")
    try:
        policy = ZeroStandardDeviationPolicy(zero_std)
    except ValueError as exc:
        raise ValueError("zero_std must be 'nan', 'zero', or 'raise'") from exc

    history = frame.shift(1)
    rolling = history.rolling(window=lookback, min_periods=minimum)
    historical_mean = rolling.mean()
    historical_std = rolling.std(ddof=ddof)
    zero_mask = historical_std.eq(0.0)
    if policy is ZeroStandardDeviationPolicy.RAISE and bool(zero_mask.any().any()):
        raise ValueError("rolling historical standard deviation is zero")
    z_scores = (frame - historical_mean) / historical_std
    if policy is ZeroStandardDeviationPolicy.ZERO:
        z_scores = z_scores.mask(zero_mask & frame.notna(), 0.0)
    else:
        z_scores = z_scores.mask(zero_mask)
    valid = z_scores.notna()
    starts, ends = _rolling_window_dates(frame, lookback, minimum)
    return RollingZScoreResult(
        residuals_decimal=frame,
        historical_mean_decimal=historical_mean,
        historical_std_decimal=historical_std,
        z_scores=z_scores,
        valid_observation=valid,
        window_start_dates=starts,
        window_end_dates=ends,
        lookback=lookback,
        min_observations=minimum,
        ddof=ddof,
        zero_std_policy=policy,
    )


def calculate_rolling_z_scores(
    residuals_decimal: pd.Series | pd.DataFrame,
    *,
    lookback: int = 60,
    min_observations: int | None = None,
    ddof: int = 1,
    zero_std: ZeroStandardDeviationPolicy | str = ZeroStandardDeviationPolicy.NAN,
) -> RollingZScoreResult:
    """Alias for :func:`rolling_z_scores`; inputs are decimal yield residuals."""
    return rolling_z_scores(
        residuals_decimal,
        lookback=lookback,
        min_observations=min_observations,
        ddof=ddof,
        zero_std=zero_std,
    )


def rolling_z_score(
    residuals_decimal: pd.Series,
    *,
    lookback: int = 60,
    min_observations: int | None = None,
    ddof: int = 1,
    zero_std: ZeroStandardDeviationPolicy | str = ZeroStandardDeviationPolicy.NAN,
) -> pd.DataFrame:
    """Return one series' decimal rolling inputs and dimensionless z-score table.

    Historical mean and standard deviation at ``t`` use only observations
    through ``t-1``. The result columns are ``residual_decimal``,
    ``historical_mean_decimal``, ``historical_std_decimal``, ``z_score``, and
    ``valid_observation``.
    """
    if not isinstance(residuals_decimal, pd.Series):
        raise TypeError("residuals_decimal must be a pandas Series")
    result = rolling_z_scores(
        residuals_decimal,
        lookback=lookback,
        min_observations=min_observations,
        ddof=ddof,
        zero_std=zero_std,
    )
    column = result.residuals_decimal.columns[0]
    return pd.DataFrame(
        {
            "residual_decimal": result.residuals_decimal[column],
            "historical_mean_decimal": result.historical_mean_decimal[column],
            "historical_std_decimal": result.historical_std_decimal[column],
            "z_score": result.z_scores[column],
            "valid_observation": result.valid_observation[column],
            "window_start_date": result.window_start_dates[column],
            "window_end_date": result.window_end_dates[column],
        }
    )


def rank_curve_points(
    residuals_decimal: pd.Series | pd.DataFrame,
    *,
    date: Hashable | None = None,
    z_scores: pd.Series | pd.DataFrame | None = None,
    rank_by: str = "residual",
) -> pd.DataFrame:
    """Rank one curve cross-section from relatively rich to relatively cheap.

    Residuals are decimal annual yields and are reported in both decimals and
    bp. ``rank_by='residual'`` sorts ascending observed-minus-fitted residual;
    ``rank_by='z_score'`` sorts ascending dimensionless z-score and requires
    ``z_scores``. For residual ranking, rank one is richest and the last rank
    is cheapest. Z-score ranking instead orders deviations from each tenor's
    historical residual mean; it need not agree with current rich/cheap labels.
    Missing ranking values are retained at the bottom without a rank.

    A DataFrame input requires ``date`` and is interpreted as dates by tenors.
    A Series is already one cross-section and does not accept ``date``.
    """
    residual_series = _cross_section(residuals_decimal, date, "residuals_decimal")
    z_series: pd.Series | None = None
    if z_scores is not None:
        z_series = _cross_section(z_scores, date, "z_scores")
        if set(z_series.index) != set(residual_series.index):
            raise ValueError("z_scores tenors must match residual tenors exactly")
        z_series = z_series.reindex(residual_series.index)
    if rank_by not in {"residual", "z_score"}:
        raise ValueError("rank_by must be 'residual' or 'z_score'")
    if rank_by == "z_score" and z_series is None:
        raise ValueError("rank_by='z_score' requires z_scores")

    table = pd.DataFrame(
        {
            "tenor": residual_series.index,
            "maturity_years": [maturity_in_years(x) for x in residual_series.index],
            "residual_decimal": residual_series.to_numpy(dtype=float),
        }
    )
    table["residual_bp"] = table["residual_decimal"] / BASIS_POINT_DECIMAL
    table["classification"] = classify_rich_cheap(table["residual_decimal"])
    if z_series is not None:
        table["z_score"] = z_series.to_numpy(dtype=float)
    ranking_column = "residual_decimal" if rank_by == "residual" else "z_score"
    table = table.sort_values(
        [ranking_column, "maturity_years"], na_position="last", kind="stable"
    ).reset_index(drop=True)
    table.insert(0, "rank_rich_to_cheap", table[ranking_column].rank(method="first"))
    return table


def butterfly_yield(
    yields_decimal: Mapping[Hashable, float] | pd.Series,
) -> float:
    """Return ``2*y_5Y - y_2Y - y_10Y`` in decimal annual yield units.

    A positive value means the 5Y belly yield is high relative to the average
    of the 2Y and 10Y wing yields (belly cheap on this raw yield measure); a
    negative value means the belly is rich. This quoted yield measure is
    separate from the DV01-weighted trade notionals.
    """
    values = _required_tenor_values(yields_decimal, "yields_decimal")
    _validate_decimal_yields(values, "yields_decimal")
    return float(2.0 * values[5.0] - values[2.0] - values[10.0])


def construct_dv01_neutral_butterfly(
    unit_dv01_currency_per_bp_per_notional: Mapping[Hashable, float] | pd.Series,
    *,
    belly_notional: float = 1_000_000.0,
    belly_side: str = "long",
) -> pd.DataFrame:
    """Construct signed 2Y/5Y/10Y notionals with zero first-order DV01.

    ``unit_dv01_currency_per_bp_per_notional`` contains positive DV01 magnitude
    per one currency unit of face notional for each tenor. ``belly_notional``
    is a strictly positive absolute currency face amount. For a long belly,
    its signed notional is positive and both wings are short; for a short belly
    the signs reverse. Each wing offsets exactly half of the belly's signed
    position DV01:

    ``N_2*DV01_2 = N_10*DV01_10 = -0.5*N_5*DV01_5``.

    The returned ``notional`` is signed currency face, unit DV01 is currency
    per bp per currency face unit, and signed/aggregate DV01 is currency per bp.
    Local neutrality does not remove convexity, carry, roll, liquidity, or
    curve-shape risk.
    """
    unit_dv01 = _required_tenor_values(
        unit_dv01_currency_per_bp_per_notional,
        "unit_dv01_currency_per_bp_per_notional",
    )
    for maturity, value in unit_dv01.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "unit DV01 values must be finite positive currency-per-bp "
                f"magnitudes; invalid {maturity:g}Y value"
            )
    absolute_belly = _finite_number(belly_notional, "belly_notional")
    if absolute_belly <= 0.0:
        raise ValueError("belly_notional must be a strictly positive absolute amount")
    if belly_side not in {"long", "short"}:
        raise ValueError("belly_side must be 'long' or 'short'")
    belly_sign = 1.0 if belly_side == "long" else -1.0
    notionals = {
        5.0: belly_sign * absolute_belly,
        2.0: -0.5 * belly_sign * absolute_belly * unit_dv01[5.0] / unit_dv01[2.0],
        10.0: -0.5 * belly_sign * absolute_belly * unit_dv01[5.0] / unit_dv01[10.0],
    }
    signed_dv01 = {
        maturity: notionals[maturity] * unit_dv01[maturity]
        for maturity in _BUTTERFLY_MATURITIES
    }
    aggregate = float(math.fsum(signed_dv01.values()))
    return pd.DataFrame(
        {
            "leg": _BUTTERFLY_LEGS,
            "maturity_years": _BUTTERFLY_MATURITIES,
            "notional": [notionals[x] for x in _BUTTERFLY_MATURITIES],
            "side": [
                "long" if notionals[x] > 0.0 else "short" for x in _BUTTERFLY_MATURITIES
            ],
            "unit_dv01_currency_per_bp_per_notional": [
                unit_dv01[x] for x in _BUTTERFLY_MATURITIES
            ],
            "signed_position_dv01_currency_per_bp": [
                signed_dv01[x] for x in _BUTTERFLY_MATURITIES
            ],
            "aggregate_dv01_currency_per_bp": aggregate,
        }
    )


def butterfly_report(
    yields_decimal: Mapping[Hashable, float] | pd.Series,
    unit_dv01_currency_per_bp_per_notional: Mapping[Hashable, float] | pd.Series,
    residuals_decimal: Mapping[Hashable, float] | pd.Series,
    rolling_z_score_values: Mapping[Hashable, float] | pd.Series,
    *,
    belly_notional: float = 1_000_000.0,
    belly_side: str = "long",
) -> pd.DataFrame:
    """Build the requested 2Y/5Y/10Y relative-value leg report.

    Yields and curve residuals are decimal annual rates, z-scores are
    dimensionless, notionals are signed currency face amounts, and DV01 values
    are currency per bp. The repeated butterfly columns explicitly report
    ``2*y_5Y-y_2Y-y_10Y`` in decimal yield and bp units. ``aggregate_dv01`` is
    repeated on each leg to make the portfolio reconciliation visible.
    """
    yields = _required_tenor_values(yields_decimal, "yields_decimal")
    residuals = _required_tenor_values(
        residuals_decimal, "residuals_decimal", allow_nan=True
    )
    z_scores = _required_tenor_values(
        rolling_z_score_values, "rolling_z_score_values", allow_nan=True
    )
    _validate_decimal_yields(yields, "yields_decimal")
    table = construct_dv01_neutral_butterfly(
        unit_dv01_currency_per_bp_per_notional,
        belly_notional=belly_notional,
        belly_side=belly_side,
    )
    table.insert(2, "yield_decimal", [yields[x] for x in _BUTTERFLY_MATURITIES])
    residual_values = np.asarray(
        [residuals[x] for x in _BUTTERFLY_MATURITIES], dtype=float
    )
    table["curve_residual_decimal"] = residual_values
    table["curve_residual_bp"] = residual_values / BASIS_POINT_DECIMAL
    table["rolling_z_score"] = [z_scores[x] for x in _BUTTERFLY_MATURITIES]
    measure = butterfly_yield(yields)
    table["butterfly_yield_decimal"] = measure
    table["butterfly_yield_bp"] = measure / BASIS_POINT_DECIMAL
    return table


def analyse_2s5s10s_butterfly(
    yields_decimal: Mapping[Hashable, float] | pd.Series,
    unit_dv01_currency_per_bp_per_notional: Mapping[Hashable, float] | pd.Series,
    residuals_decimal: Mapping[Hashable, float] | pd.Series,
    rolling_z_score_values: Mapping[Hashable, float] | pd.Series,
    *,
    belly_notional: float = 1_000_000.0,
    belly_side: str = "long",
) -> ButterflyAnalytics:
    """Return structured 2Y/5Y/10Y analytics with explicit units.

    Inputs use decimal annual yields/residuals, dimensionless z-scores, positive
    unit DV01 magnitudes in currency per bp per unit notional, and an absolute
    belly face notional in currency.
    """
    legs = butterfly_report(
        yields_decimal,
        unit_dv01_currency_per_bp_per_notional,
        residuals_decimal,
        rolling_z_score_values,
        belly_notional=belly_notional,
        belly_side=belly_side,
    )
    measure = float(legs["butterfly_yield_decimal"].iloc[0])
    aggregate = float(legs["aggregate_dv01_currency_per_bp"].iloc[0])
    return ButterflyAnalytics(
        yield_butterfly_decimal=measure,
        yield_butterfly_bp=measure / BASIS_POINT_DECIMAL,
        legs=legs,
        aggregate_dv01_currency_per_bp=aggregate,
    )


def _validate_yield_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if panel.empty or len(panel.columns) == 0:
        raise ValueError(f"{name} must contain dates and tenor columns")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if panel.index.hasnans or panel.index.has_duplicates:
        raise ValueError(f"{name} dates must be valid and unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be strictly increasing")
    if panel.columns.has_duplicates:
        raise ValueError(f"{name} tenor labels must be unique")
    maturities = [maturity_in_years(label) for label in panel.columns]
    if len(set(maturities)) != len(maturities):
        raise ValueError(f"{name} contains duplicate numerical tenors")
    numeric = _numeric_pandas(panel, name)
    present = numeric.to_numpy(dtype=float)
    present = present[~np.isnan(present)]
    if np.any((present < -0.2) | (present > 1.0)):
        raise ValueError(
            f"{name} must contain decimal annual yields between -0.2 and 1.0"
        )
    return numeric.copy()


def _numeric_pandas(
    values: pd.Series | pd.DataFrame, name: str
) -> pd.Series | pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    malformed = numeric.isna() & ~values.isna()
    if bool(np.asarray(malformed).any()):
        raise ValueError(f"{name} contains non-numeric values")
    array = numeric.to_numpy(dtype=float)
    if not bool((np.isfinite(array) | np.isnan(array)).all()):
        raise ValueError(f"{name} contains infinite values")
    return numeric.astype(float)


def _as_residual_frame(residuals: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if not isinstance(residuals, (pd.Series, pd.DataFrame)):
        raise TypeError("residuals_decimal must be a pandas Series or DataFrame")
    if residuals.empty:
        raise ValueError("residuals_decimal must contain observations")
    if not isinstance(residuals.index, pd.DatetimeIndex):
        raise TypeError("residuals_decimal must have a DatetimeIndex")
    if residuals.index.hasnans or residuals.index.has_duplicates:
        raise ValueError("residual dates must be valid and unique")
    if not residuals.index.is_monotonic_increasing:
        raise ValueError("residual dates must be strictly increasing")
    if isinstance(residuals, pd.Series):
        name = residuals.name if residuals.name is not None else "residual"
        frame = residuals.to_frame(name=name)
    else:
        if len(residuals.columns) == 0 or residuals.columns.has_duplicates:
            raise ValueError("residual tenor columns must be non-empty and unique")
        frame = residuals
    return _numeric_pandas(frame, "residuals_decimal").copy()


def _rolling_window_dates(
    frame: pd.DataFrame, lookback: int, minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = pd.DataFrame(pd.NaT, index=frame.index, columns=frame.columns)
    ends = starts.copy()
    for row_number in range(len(frame)):
        lower = max(0, row_number - lookback)
        history = frame.iloc[lower:row_number]
        for column_number, column in enumerate(frame.columns):
            available = history[column].dropna()
            if len(available) >= minimum:
                starts.iat[row_number, column_number] = available.index[0]
                ends.iat[row_number, column_number] = available.index[-1]
    return starts, ends


def _cross_section(
    values: pd.Series | pd.DataFrame, date: Hashable | None, name: str
) -> pd.Series:
    if isinstance(values, pd.Series):
        if date is not None:
            raise ValueError(f"date must be omitted when {name} is a Series")
        series = values
    elif isinstance(values, pd.DataFrame):
        if date is None:
            raise ValueError(f"date is required when {name} is a DataFrame")
        if date not in values.index:
            raise ValueError(f"date {date!r} is not present in {name}")
        series = values.loc[date]
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame")
    if series.index.has_duplicates or len(series) == 0:
        raise ValueError(f"{name} tenors must be non-empty and unique")
    numeric = _numeric_pandas(series, name)
    maturities = [maturity_in_years(label) for label in numeric.index]
    if len(set(maturities)) != len(maturities):
        raise ValueError(f"{name} contains duplicate numerical tenors")
    return numeric


def _required_tenor_values(
    values: Mapping[Hashable, float] | pd.Series,
    name: str,
    *,
    allow_nan: bool = False,
) -> dict[float, float]:
    if isinstance(values, pd.Series):
        items = values.items()
    elif isinstance(values, Mapping):
        items = values.items()
    else:
        raise TypeError(f"{name} must be a mapping or pandas Series")
    normalized: dict[float, float] = {}
    for label, raw_value in items:
        maturity = maturity_in_years(label)
        if maturity in normalized:
            raise ValueError(f"{name} contains duplicate numerical tenor {maturity:g}Y")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} values must be numeric") from exc
        if not math.isfinite(value) and not (allow_nan and math.isnan(value)):
            raise ValueError(f"{name} values must be finite")
        normalized[maturity] = value
    missing = [
        maturity for maturity in _BUTTERFLY_MATURITIES if maturity not in normalized
    ]
    if missing:
        labels = ", ".join(f"{maturity:g}Y" for maturity in missing)
        raise ValueError(f"{name} is missing required tenor(s): {labels}")
    return {maturity: normalized[maturity] for maturity in _BUTTERFLY_MATURITIES}


def _validate_decimal_yields(values: Mapping[float, float], name: str) -> None:
    if any(value < -0.2 or value > 1.0 for value in values.values()):
        raise ValueError(
            f"{name} must contain decimal annual yields between -0.2 and 1.0"
        )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _non_negative_number(value: object, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "BASIS_POINT_DECIMAL",
    "ButterflyAnalytics",
    "RelativeValueLabel",
    "RollingZScoreResult",
    "ZeroStandardDeviationPolicy",
    "analyse_2s5s10s_butterfly",
    "butterfly_report",
    "butterfly_yield",
    "calculate_curve_residuals",
    "calculate_nss_residuals",
    "calculate_rolling_z_scores",
    "classify_residual",
    "classify_rich_cheap",
    "construct_dv01_neutral_butterfly",
    "rank_curve_points",
    "residual_report",
    "residuals_to_basis_points",
    "rolling_z_score",
    "rolling_z_scores",
]
