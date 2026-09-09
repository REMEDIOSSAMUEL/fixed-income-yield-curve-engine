"""Historical 2Y/5Y/10Y relative-value research backtesting.

This module is an educational sensitivity study, not a deployable trading
strategy or evidence of live alpha. The signal is the 5Y Nelson-Siegel-
Svensson (NSS) fitted-curve residual, defined as observed minus fitted decimal
annual yield. That choice reuses :mod:`fixed_income.relative_value`'s explicit
rich/cheap convention while keeping the signal distinct from the DV01-neutral
trade used to express it.

Timing is deliberately visible. On observation date ``t`` the residual is
standardised against rows ending at ``t-1``. The resulting signal determines
the target entered after observing ``t`` and held over ``t`` to ``t+1``. In
the dated result, ``gross_pnl_currency_approx`` at ``t`` is consequently earned
by ``held_*`` positions decided at ``t-1`` from yield changes between ``t-1``
and ``t``. A signal can affect same-date transaction cost, but never
same-observation gross P&L.

The data are Treasury constant-maturity yields, which are statistical,
par-yield-like observations rather than directly tradable securities. P&L is
the first-order approximation ``-signed_DV01 * yield_change_bp``. It excludes
true cash-bond or futures prices, carry, roll, convexity, financing, funding,
and executable bid/ask data. The simple cost model is basis points of traded
face notional and does not purport to reproduce institutional execution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fixed_income.bonds import FixedRateBond, dv01
from fixed_income.curves import CurveRepresentation, calibrate_nss
from fixed_income.pca import maturity_in_years
from fixed_income.relative_value import BASIS_POINT_DECIMAL, rolling_z_score

TRADING_DAYS_PER_YEAR = 252
"""Conventional trading-day annualisation factor, in observations per year."""

_MATURITIES = (2.0, 5.0, 10.0)
_LABELS = {2.0: "2Y", 5.0: "5Y", 10.0: "10Y"}


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration with explicit dimensionless, currency, and cost units.

    Args:
        lookback: Number of preceding observations used for rolling estimates.
        min_observations: Minimum preceding non-missing residual observations.
            ``None`` means ``lookback``.
        ddof: Degrees of freedom used for the historical standard deviation.
        entry_z: Positive dimensionless absolute entry threshold. Equality
            enters: ``z >= entry_z`` is cheap and ``z <= -entry_z`` is rich.
        exit_z: Non-negative dimensionless exit threshold strictly below
            ``entry_z``. Long positions exit at ``z <= exit_z`` and shorts at
            ``z >= -exit_z``, including jumps across the exit band. An opposite
            entry threshold reverses the position directly.
        belly_notional_currency: Absolute 5Y face-notional normalisation.
        transaction_cost_bps_per_face: Non-negative one-way cost in basis
            points of absolute face notional traded. One bp of face is
            explicitly ``0.0001`` currency per currency of face.
        annualisation_factor: Observations per year used only for P&L mean,
            P&L volatility, and Sharpe annualisation; it is not a return.
        liquidate_at_end: If true, force the final target to zero and charge the
            corresponding cost because no later observation can earn P&L.
    """

    lookback: int = 60
    min_observations: int | None = None
    ddof: int = 1
    entry_z: float = 2.0
    exit_z: float = 0.5
    belly_notional_currency: float = 1_000_000.0
    transaction_cost_bps_per_face: float = 0.01
    annualisation_factor: int = TRADING_DAYS_PER_YEAR
    liquidate_at_end: bool = True

    def __post_init__(self) -> None:
        """Validate all fields without changing their documented units."""
        _positive_integer(self.lookback, "lookback")
        minimum = (
            self.lookback if self.min_observations is None else self.min_observations
        )
        _positive_integer(minimum, "min_observations")
        if minimum > self.lookback:
            raise ValueError("min_observations cannot exceed lookback")
        if isinstance(self.ddof, bool) or not isinstance(self.ddof, int):
            raise TypeError("ddof must be a non-negative integer")
        if self.ddof < 0 or self.ddof >= minimum:
            raise ValueError("ddof must be non-negative and below min_observations")
        entry = _finite_number(self.entry_z, "entry_z")
        exit_value = _finite_number(self.exit_z, "exit_z")
        if entry <= 0.0:
            raise ValueError("entry_z must be strictly positive")
        if exit_value < 0.0 or exit_value >= entry:
            raise ValueError("exit_z must be non-negative and strictly below entry_z")
        if (
            _finite_number(self.belly_notional_currency, "belly_notional_currency")
            <= 0.0
        ):
            raise ValueError("belly_notional_currency must be strictly positive")
        if (
            _finite_number(
                self.transaction_cost_bps_per_face,
                "transaction_cost_bps_per_face",
            )
            < 0.0
        ):
            raise ValueError("transaction_cost_bps_per_face must be non-negative")
        _positive_integer(self.annualisation_factor, "annualisation_factor")
        if not isinstance(self.liquidate_at_end, bool):
            raise TypeError("liquidate_at_end must be a bool")


@dataclass(frozen=True)
class BacktestResult:
    """Dated backtest observations and summary metrics with explicit units.

    ``observations`` contains decimal yields/residuals, yield changes in bp,
    signed face notionals in currency, DV01 in currency per bp, approximate P&L
    and drawdowns in currency, and dimensionless z-scores/directions. Metrics
    are a two-column table (gross and net); no percentage return is reported
    because invested capital is not defined by a DV01-normalised study.
    """

    observations: pd.DataFrame
    metrics: pd.DataFrame
    config: BacktestConfig
    methodology: str
    limitations: tuple[str, ...]


def generate_threshold_positions(
    z_scores: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    """Generate stateful target directions from dimensionless z-scores.

    Returns ``+1`` for a long 5Y belly/short wings trade, ``-1`` for a short
    belly/long wings trade, and zero when flat. Positive 5Y NSS residual z means
    above its trailing mean and maps to ``+1``; the raw residual can still be
    negative (rich to the current cross-sectional fit). Equality at entry and exit
    thresholds triggers the action. Missing signals close an existing target
    because no observation is backfilled. A jump through the opposite entry
    threshold reverses the target directly.
    """
    signal = _validate_dated_series(z_scores, "z_scores", allow_missing=True)
    entry = _finite_number(entry_z, "entry_z")
    exit_value = _finite_number(exit_z, "exit_z")
    if entry <= 0.0:
        raise ValueError("entry_z must be strictly positive")
    if exit_value < 0.0 or exit_value >= entry:
        raise ValueError("exit_z must be non-negative and strictly below entry_z")

    positions: list[int] = []
    current = 0
    for value in signal:
        if pd.isna(value):
            current = 0
        elif current == 0:
            if value >= entry:
                current = 1
            elif value <= -entry:
                current = -1
        elif current == 1 and value <= -entry:
            current = -1
        elif current == -1 and value >= entry:
            current = 1
        elif (current == 1 and value <= exit_value) or (
            current == -1 and value >= -exit_value
        ):
            current = 0
        positions.append(current)
    return pd.Series(
        positions, index=signal.index, name="decision_direction", dtype=int
    )


def fit_historical_nss_residuals(
    observed_yields_decimal: pd.DataFrame,
    *,
    failure_policy: str = "raise",
) -> pd.DataFrame:
    """Fit each dated cross-section and return NSS residuals in decimal yield.

    Each row is calibrated using only information available on or before that
    date; no later date enters the fit. Every date uses the same deterministic
    multistart search, avoiding dependence on a prior local optimum. Input
    Treasury CMT observations retain their representation.
    ``failure_policy`` is ``'raise'`` or ``'nan'``; the latter leaves the
    entire failed row missing. ``result.attrs['calibration_diagnostics']`` holds
    dated parameters, decimal/bp RMSE, conditioning, active bounds and failures.
    """
    panel = _validate_yield_panel(observed_yields_decimal, require_fly=False)
    if failure_policy not in {"raise", "nan"}:
        raise ValueError("failure_policy must be 'raise' or 'nan'")
    maturities = np.asarray([maturity_in_years(label) for label in panel.columns])
    if len(maturities) < 6:
        raise ValueError("historical NSS requires at least six distinct maturities")
    order = np.argsort(maturities)
    inverse_order = np.argsort(order)
    residual_rows: list[np.ndarray] = []
    diagnostic_rows: list[dict[str, object]] = []
    for timestamp, row in panel.iterrows():
        try:
            fit = calibrate_nss(
                maturities,
                row.to_numpy(dtype=float),
                input_representation=CurveRepresentation.TREASURY_CMT,
            )
            if not fit.diagnostics.success:
                raise ArithmeticError(f"NSS fit did not converge on {timestamp.date()}")
            residual_rows.append(fit.residuals[inverse_order])
            diagnostic_rows.append(
                {
                    "date": timestamp.date().isoformat(),
                    "success": True,
                    "rmse_decimal": fit.rmse,
                    "rmse_bp": fit.rmse / BASIS_POINT_DECIMAL,
                    **asdict(fit.parameters),
                    "loading_condition_number": (
                        fit.diagnostics.loading_condition_number
                    ),
                    "jacobian_condition_number": (
                        fit.diagnostics.jacobian_condition_number
                    ),
                    "active_bounds": ",".join(fit.diagnostics.active_bounds),
                    "starts_attempted": fit.diagnostics.starts_attempted,
                    "successful_starts": fit.diagnostics.successful_starts,
                    "message": fit.diagnostics.message,
                }
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            if failure_policy == "raise":
                raise
            residual_rows.append(np.full(len(panel.columns), np.nan))
            diagnostic_rows.append(
                {
                    "date": timestamp.date().isoformat(),
                    "success": False,
                    "message": str(exc),
                }
            )
    result = pd.DataFrame(residual_rows, index=panel.index, columns=panel.columns)
    result.attrs["calibration_diagnostics"] = diagnostic_rows
    return result


def calculate_par_proxy_unit_dv01(
    yields_decimal: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate dated proxy unit DV01s in currency per bp per face unit.

    For each 2Y, 5Y, and 10Y CMT observation, this constructs a hypothetical
    semiannual bullet with coupon equal to its decimal CMT yield when
    non-negative (and zero for a negative yield), one currency unit of face,
    settlement on the observation date, and maturity at the stated year tenor.
    The contract starts at the calendar month end one year before maturity's
    backward tenor offset so leap-year February remains on a regular schedule.
    Only future cash flows enter DV01. If the observation is not on that roll,
    the proxy has fractional accrual and need not be exactly par. It is a transparent
    duration proxy, not an assertion that the CMT series identifies a specific
    tradable bond. DV01 is conventional YTM-based dirty-price sensitivity.
    """
    panel = _validate_yield_panel(yields_decimal, require_fly=True)
    fly = _select_fly_columns(panel)
    result = pd.DataFrame(index=fly.index, columns=fly.columns, dtype=float)
    for timestamp, row in fly.iterrows():
        settlement = timestamp.date()
        for label in fly.columns:
            years = int(maturity_in_years(label))
            ytm = float(row[label])
            maturity = _add_years(settlement, years)
            accrual_start = _add_years(maturity, -(years + 1))
            if maturity == (pd.Timestamp(maturity) + pd.offsets.MonthEnd(0)).date():
                accrual_start = (
                    pd.Timestamp(accrual_start) + pd.offsets.MonthEnd(0)
                ).date()
            proxy = FixedRateBond(
                accrual_start_date=accrual_start,
                maturity_date=maturity,
                coupon_rate=max(ytm, 0.0),
                face_value=1.0,
                frequency=2,
            )
            result.loc[timestamp, label] = dv01(proxy, settlement, ytm)
    return result


def run_relative_value_backtest(
    yields_decimal: pd.DataFrame,
    five_year_nss_residual_decimal: pd.Series,
    unit_dv01_currency_per_bp_per_notional: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the lagged DV01-neutral research backtest.

    Args:
        yields_decimal: Dated 2Y/5Y/10Y Treasury CMT yields as decimal annual
            rates. Other tenor columns are allowed and ignored for P&L.
        five_year_nss_residual_decimal: Dated observed-minus-NSS-fitted 5Y
            residual in decimal annual yield units, aligned exactly to yields.
        unit_dv01_currency_per_bp_per_notional: Dated positive 2Y/5Y/10Y unit
            DV01 magnitudes, currency per bp per currency face unit.
        config: Threshold, normalisation, face-notional, cost, annualisation,
            and terminal-liquidation settings.

    Returns:
        Dated gross/net approximate currency P&L and accurately labelled
        summary metrics. It does not calculate a percentage return.
    """
    settings = BacktestConfig() if config is None else config
    if not isinstance(settings, BacktestConfig):
        raise TypeError("config must be a BacktestConfig")
    panel = _select_fly_columns(_validate_yield_panel(yields_decimal, require_fly=True))
    residual = _validate_dated_series(
        five_year_nss_residual_decimal,
        "five_year_nss_residual_decimal",
        allow_missing=True,
    )
    if not residual.index.equals(panel.index):
        raise ValueError("residual and yield dates must match exactly")
    unit_dv01 = _select_fly_columns(
        _validate_dv01_panel(unit_dv01_currency_per_bp_per_notional)
    )
    if not unit_dv01.index.equals(panel.index):
        raise ValueError("unit DV01 and yield dates must match exactly")
    if len(panel) < 2:
        raise ValueError("backtest requires at least two dated observations")

    rolling = rolling_z_score(
        residual,
        lookback=settings.lookback,
        min_observations=settings.min_observations,
        ddof=settings.ddof,
        zero_std="nan",
    )
    decisions = generate_threshold_positions(
        rolling["z_score"], entry_z=settings.entry_z, exit_z=settings.exit_z
    )
    if settings.liquidate_at_end:
        decisions.iloc[-1] = 0

    target_notionals = _target_notionals(
        decisions, unit_dv01, settings.belly_notional_currency
    )
    if not np.isfinite(target_notionals.to_numpy()).all():
        raise ArithmeticError("hedge sizing produced non-finite face notionals")
    prior_targets = target_notionals.shift(1).fillna(0.0)
    prior_unit_dv01 = unit_dv01.shift(1)
    yield_changes_bp = panel.diff() / BASIS_POINT_DECIMAL
    signed_held_dv01 = prior_targets * prior_unit_dv01
    leg_pnl = -(signed_held_dv01 * yield_changes_bp)
    leg_pnl.iloc[0] = 0.0  # Only the initial row has no prior holding interval.
    if not np.isfinite(leg_pnl.to_numpy()).all():
        raise ArithmeticError("held DV01 or yield-change P&L is non-finite")
    gross_pnl = leg_pnl.sum(axis=1)

    trades = target_notionals - prior_targets
    turnover = trades.abs().sum(axis=1)
    transaction_cost = (
        turnover * settings.transaction_cost_bps_per_face * BASIS_POINT_DECIMAL
    )
    net_pnl = gross_pnl - transaction_cost
    gross_cumulative = gross_pnl.cumsum()
    net_cumulative = net_pnl.cumsum()
    held_direction = decisions.shift(1).fillna(0).astype(int)

    observations = pd.DataFrame(index=panel.index)
    observations.index.name = "date"
    observations["residual_5y_decimal"] = residual
    observations["elapsed_calendar_days"] = panel.index.to_series().diff().dt.days
    observations["historical_mean_decimal"] = rolling["historical_mean_decimal"]
    observations["historical_std_decimal"] = rolling["historical_std_decimal"]
    observations["z_score"] = rolling["z_score"]
    observations["decision_direction"] = decisions
    observations["held_direction_from_prior_date"] = held_direction
    observations["signal_window_end_date"] = rolling["window_end_date"]
    for label in ("2Y", "5Y", "10Y"):
        key = label.lower()
        observations[f"yield_{key}_decimal"] = panel[label]
        observations[f"yield_change_{key}_bp"] = yield_changes_bp[label]
        observations[f"unit_dv01_{key}_currency_per_bp_per_notional"] = unit_dv01[label]
        observations[f"target_notional_{key}_currency"] = target_notionals[label]
        observations[f"held_notional_{key}_currency"] = prior_targets[label]
        observations[f"held_dv01_{key}_currency_per_bp"] = signed_held_dv01[label]
        observations[f"pnl_{key}_currency_approx"] = leg_pnl[label]
        observations[f"trade_notional_{key}_currency"] = trades[label]
    observations["aggregate_target_dv01_currency_per_bp"] = (
        target_notionals * unit_dv01
    ).sum(axis=1)
    observations["aggregate_held_dv01_currency_per_bp"] = signed_held_dv01.sum(
        axis=1, min_count=1
    ).fillna(0.0)
    observations["turnover_abs_notional_currency"] = turnover
    observations["gross_pnl_currency_approx"] = gross_pnl
    observations["transaction_cost_currency"] = transaction_cost
    observations["net_pnl_currency_approx"] = net_pnl
    observations["cumulative_gross_pnl_currency_approx"] = gross_cumulative
    observations["cumulative_net_pnl_currency_approx"] = net_cumulative
    observations["gross_drawdown_currency"] = calculate_drawdown(gross_cumulative)
    observations["net_drawdown_currency"] = calculate_drawdown(net_cumulative)
    if not np.isfinite(net_pnl.to_numpy()).all():
        raise ArithmeticError("transaction costs or net P&L are non-finite")

    metrics = calculate_performance_metrics(
        observations, annualisation_factor=settings.annualisation_factor
    )
    return BacktestResult(
        observations=observations,
        metrics=metrics,
        config=settings,
        methodology=(
            "5Y observed-minus-NSS-fitted residual z-score; history through t-1; "
            "decision at t held over t to t+1; DV01-neutral long-belly target for "
            "above-trailing-mean signals; directional exits include band crossings; "
            "daily hedge rebalancing; costs booked on decision date; "
            "first-order yield-change P&L at prior-date DV01"
        ),
        limitations=(
            "Treasury constant-maturity yields are not directly tradable securities.",
            "Observation-date execution is hypothetical: publication timestamps and "
            "an executable subsequent price are unavailable; "
            "no trading lag is inferred.",
            "Data are a later historical snapshot, not a point-in-time vintage; "
            "revisions and selected complete-case dates can affect results.",
            "Annualisation assumes daily observations, includes signal warm-up zeros, "
            "and does not adjust for serial correlation or irregular calendar gaps.",
            "A positive residual z-score means above its historical mean, not "
            "necessarily a positive observed-minus-fitted raw residual.",
            "Execution costs are a simple bp-of-face approximation, not "
            "historical bid/ask or market impact.",
            "Carry, roll, convexity, financing, funding, futures basis, taxes, "
            "and security selection are omitted.",
            "NSS fit choice, proxy DV01s, thresholds, and sample period create "
            "model risk.",
            "This is educational research evidence, not evidence of live "
            "alpha or profitability.",
        ),
    )


def calculate_drawdown(cumulative_pnl_currency: pd.Series) -> pd.Series:
    """Return currency drawdown from the prior high-water mark including zero.

    The input is cumulative currency P&L, not portfolio value or percentage
    return. The initial capital-free high-water mark is zero, so an immediately
    losing path has a negative drawdown rather than an artificial value of zero.
    """
    cumulative = _validate_dated_series(
        cumulative_pnl_currency, "cumulative_pnl_currency", allow_missing=False
    )
    running_peak = cumulative.cummax().clip(lower=0.0)
    return (cumulative - running_peak).rename("drawdown_currency")


def calculate_performance_metrics(
    observations: pd.DataFrame,
    *,
    annualisation_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate gross and net P&L metrics without inventing a return denominator.

    P&L, annualised mean P&L, annualised P&L volatility, and maximum drawdown
    are currency. Sharpe and hit rate are dimensionless; time in market is a
    fraction; turnover is total absolute currency face traded. Annualisation
    assumes equally weighted observations and the supplied observations/year.
    The first no-prior-period row is excluded from mean, volatility, Sharpe,
    hit rate, and time-in-market calculations.
    """
    if not isinstance(observations, pd.DataFrame) or observations.empty:
        raise ValueError("observations must be a non-empty DataFrame")
    factor = _positive_integer(annualisation_factor, "annualisation_factor")
    required = {
        "gross_pnl_currency_approx",
        "net_pnl_currency_approx",
        "held_direction_from_prior_date",
        "decision_direction",
        "turnover_abs_notional_currency",
        "transaction_cost_currency",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations are missing metric columns: {sorted(missing)}")
    _validate_index(observations.index, "performance")
    numeric = _numeric_frame(observations.loc[:, sorted(required)], "observations")
    if numeric.isna().any().any():
        raise ValueError("performance observations must not contain missing values")
    if (
        (numeric[["transaction_cost_currency", "turnover_abs_notional_currency"]] < 0)
        .any()
        .any()
    ):
        raise ValueError("transaction costs and turnover must be non-negative currency")
    for column in ("decision_direction", "held_direction_from_prior_date"):
        if not numeric[column].isin([-1, 0, 1]).all():
            raise ValueError("directions must be -1, 0, or 1")
    observations = numeric
    if not np.allclose(
        observations["net_pnl_currency_approx"],
        observations["gross_pnl_currency_approx"]
        - observations["transaction_cost_currency"],
        rtol=1e-12,
        atol=1e-10,
    ):
        raise ValueError("net P&L must equal gross P&L minus transaction costs")
    interval_rows = observations.iloc[1:]
    active = interval_rows["held_direction_from_prior_date"].ne(0)
    denominator = len(interval_rows)
    previous_decision = observations["decision_direction"].shift(1).fillna(0)
    decisions = observations["decision_direction"]
    entries = ((decisions.ne(0)) & (decisions.ne(previous_decision))).sum()
    trade_events = observations["turnover_abs_notional_currency"].gt(1e-9).sum()

    columns: dict[str, dict[str, float]] = {}
    for variant in ("gross", "net"):
        pnl = interval_rows[f"{variant}_pnl_currency_approx"].astype(float)
        mean = float(pnl.mean()) if len(pnl) else math.nan
        volatility = float(pnl.std(ddof=1)) if len(pnl) > 1 else math.nan
        annual_mean = mean * factor
        annual_vol = volatility * math.sqrt(factor)
        sharpe = (
            mean / volatility * math.sqrt(factor)
            if volatility > 0.0 and math.isfinite(volatility)
            else math.nan
        )
        active_pnl = pnl.loc[active]
        hit_rate = float(active_pnl.gt(0.0).mean()) if len(active_pnl) else math.nan
        columns[variant] = {
            "cumulative_pnl_currency_approx": float(
                observations[f"{variant}_pnl_currency_approx"].sum()
            ),
            "annualised_mean_pnl_currency_approx": annual_mean,
            "annualised_volatility_currency_approx": annual_vol,
            "sharpe_ratio_on_approx_pnl": sharpe,
            "maximum_drawdown_currency": -float(
                calculate_drawdown(
                    observations[f"{variant}_pnl_currency_approx"].cumsum()
                ).min()
            ),
            "turnover_abs_notional_currency": float(
                observations["turnover_abs_notional_currency"].sum()
            ),
            "hit_rate_active_intervals": hit_rate,
            "number_of_entries": float(entries),
            "number_of_trade_events": float(trade_events),
            "time_in_market_fraction": float(active.sum() / denominator)
            if denominator
            else math.nan,
            "total_transaction_cost_currency": float(
                observations["transaction_cost_currency"].sum()
            )
            if variant == "net"
            else 0.0,
        }
    table = pd.DataFrame(columns)
    table.index.name = "metric"
    return table


def transaction_cost_sensitivity(
    yields_decimal: pd.DataFrame,
    five_year_nss_residual_decimal: pd.Series,
    unit_dv01_currency_per_bp_per_notional: pd.DataFrame,
    *,
    cost_bps_per_face: Sequence[float] = (0.0, 0.005, 0.01, 0.02),
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Evaluate net approximate P&L across bp-of-face cost assumptions.

    Every cost input is one-way basis points of absolute face notional traded;
    one bp equals ``0.0001``. Signal, positions, and gross P&L are held fixed
    across scenarios. Returned P&L, drawdown, turnover, and total cost fields
    are currency; Sharpe/hit rate/time in market are dimensionless.
    """
    base = BacktestConfig() if config is None else config
    if not isinstance(base, BacktestConfig):
        raise TypeError("config must be a BacktestConfig")
    if isinstance(cost_bps_per_face, (str, bytes)):
        raise TypeError("cost_bps_per_face must be a sequence of non-negative values")
    scenarios = tuple(
        _finite_number(value, "cost_bps_per_face") for value in cost_bps_per_face
    )
    if not scenarios or any(value < 0.0 for value in scenarios):
        raise ValueError("cost_bps_per_face must contain non-negative values")
    rows: list[dict[str, float]] = []
    for cost in scenarios:
        result = run_relative_value_backtest(
            yields_decimal,
            five_year_nss_residual_decimal,
            unit_dv01_currency_per_bp_per_notional,
            config=replace(base, transaction_cost_bps_per_face=cost),
        )
        net = result.metrics["net"]
        rows.append(
            {
                "transaction_cost_bps_per_face": cost,
                **{name: float(value) for name, value in net.items()},
            }
        )
    return pd.DataFrame(rows)


def plot_backtest(result: BacktestResult) -> Figure:
    """Plot dimensionless signal and cumulative approximate currency P&L.

    No file is written. Use :func:`save_backtest_outputs` to create the required
    PNG together with dated and cost-sensitivity CSV artifacts.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    frame = result.observations
    figure, (signal_axis, pnl_axis) = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True
    )
    signal_axis.plot(frame.index, frame["z_score"], color="tab:blue", linewidth=1.2)
    signal_axis.axhline(result.config.entry_z, color="tab:red", linestyle="--")
    signal_axis.axhline(-result.config.entry_z, color="tab:red", linestyle="--")
    signal_axis.axhline(result.config.exit_z, color="grey", linestyle=":")
    signal_axis.axhline(-result.config.exit_z, color="grey", linestyle=":")
    signal_axis.set_ylabel("5Y residual z-score")
    signal_axis.set_title("Research signal (history through prior observation)")
    signal_axis.grid(visible=True, alpha=0.25)
    pnl_axis.plot(
        frame.index,
        frame["cumulative_gross_pnl_currency_approx"],
        label="Gross",
    )
    pnl_axis.plot(
        frame.index,
        frame["cumulative_net_pnl_currency_approx"],
        label="Net of simple costs",
    )
    pnl_axis.set_ylabel("Cumulative approximate P&L (currency)")
    pnl_axis.set_xlabel("Observation date")
    pnl_axis.grid(visible=True, alpha=0.25)
    pnl_axis.legend()
    return figure


def save_backtest_outputs(
    result: BacktestResult,
    cost_sensitivity: pd.DataFrame,
    *,
    outputs_directory: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write dated P&L, figure, sensitivity CSV and metadata beneath ``outputs/``.

    The dated CSV contains all stated units in column names, the PNG labels P&L
    as an approximation, and the sensitivity CSV reports bp-of-face costs.
    Paths outside the repository's ``outputs`` tree are rejected.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult")
    if not isinstance(cost_sensitivity, pd.DataFrame) or cost_sensitivity.empty:
        raise ValueError("cost_sensitivity must be a non-empty DataFrame")
    outputs_root = (Path(__file__).resolve().parents[1] / "outputs").resolve()
    requested = outputs_root if outputs_directory is None else Path(outputs_directory)
    if not requested.is_absolute():
        requested = Path(__file__).resolve().parents[1] / requested
    target = requested.resolve()
    if target != outputs_root and outputs_root not in target.parents:
        raise ValueError("generated backtest output must be saved below outputs/")
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "rv_backtest.csv"
    png_path = target / "rv_backtest.png"
    sensitivity_path = target / "transaction_cost_sensitivity.csv"
    result.observations.to_csv(csv_path, index=True)
    cost_sensitivity.to_csv(sensitivity_path, index=False)
    metadata = {
        "config": asdict(result.config),
        "methodology": result.methodology,
        "limitations": result.limitations,
        "start_date": result.observations.index[0].date().isoformat(),
        "end_date": result.observations.index[-1].date().isoformat(),
        "observation_count": len(result.observations),
        "provenance": result.observations.attrs.get("provenance", {}),
    }
    (target / "rv_backtest_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure = plot_backtest(result)
    try:
        figure.savefig(png_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return csv_path, png_path, sensitivity_path


def _target_notionals(
    directions: pd.Series,
    unit_dv01: pd.DataFrame,
    belly_notional: float,
) -> pd.DataFrame:
    targets = pd.DataFrame(0.0, index=unit_dv01.index, columns=unit_dv01.columns)
    belly_signed = directions.astype(float) * belly_notional
    targets["5Y"] = belly_signed
    targets["2Y"] = -0.5 * belly_signed * unit_dv01["5Y"] / unit_dv01["2Y"]
    targets["10Y"] = -0.5 * belly_signed * unit_dv01["5Y"] / unit_dv01["10Y"]
    return targets


def _validate_yield_panel(panel: pd.DataFrame, *, require_fly: bool) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty or panel.shape[1] == 0:
        raise ValueError("yields_decimal must be a non-empty DataFrame")
    _validate_index(panel.index, "yield")
    if panel.columns.has_duplicates:
        raise ValueError("yield tenor columns must be unique")
    maturities = [maturity_in_years(label) for label in panel.columns]
    if len(maturities) != len(set(maturities)):
        raise ValueError("yield panel contains duplicate numerical tenors")
    numeric = _numeric_frame(panel, "yields_decimal")
    if bool(numeric.isna().any().any()):
        raise ValueError("yields_decimal must not contain missing values")
    values = numeric.to_numpy(dtype=float)
    if np.any((values < -0.2) | (values > 1.0)):
        raise ValueError("yields_decimal must contain decimal annual rates")
    if require_fly:
        _select_fly_columns(numeric)
    return numeric


def _validate_dv01_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("unit DV01 must be a non-empty DataFrame")
    _validate_index(panel.index, "unit DV01")
    if panel.columns.has_duplicates:
        raise ValueError("unit DV01 tenor columns must be unique")
    maturities = [maturity_in_years(label) for label in panel.columns]
    if len(maturities) != len(set(maturities)):
        raise ValueError("unit DV01 panel contains duplicate numerical tenors")
    numeric = _numeric_frame(panel, "unit DV01")
    if bool(numeric.isna().any().any()) or bool((numeric <= 0.0).any().any()):
        raise ValueError("unit DV01 values must be finite and strictly positive")
    _select_fly_columns(numeric)
    return numeric


def _select_fly_columns(panel: pd.DataFrame) -> pd.DataFrame:
    by_maturity = {maturity_in_years(column): column for column in panel.columns}
    missing = [value for value in _MATURITIES if value not in by_maturity]
    if missing:
        raise ValueError("panel must contain 2Y, 5Y, and 10Y tenors")
    selected = panel.loc[:, [by_maturity[value] for value in _MATURITIES]].copy()
    selected.columns = [_LABELS[value] for value in _MATURITIES]
    return selected


def _validate_dated_series(
    values: pd.Series, name: str, *, allow_missing: bool
) -> pd.Series:
    if not isinstance(values, pd.Series) or values.empty:
        raise ValueError(f"{name} must be a non-empty pandas Series")
    _validate_index(values.index, name)
    numeric = pd.to_numeric(values, errors="coerce")
    malformed = numeric.isna() & ~values.isna()
    if bool(malformed.any()):
        raise ValueError(f"{name} contains non-numeric values")
    array = numeric.to_numpy(dtype=float)
    valid = (
        np.isfinite(array) | np.isnan(array) if allow_missing else np.isfinite(array)
    )
    if not bool(valid.all()):
        raise ValueError(f"{name} contains invalid values")
    if not allow_missing and bool(numeric.isna().any()):
        raise ValueError(f"{name} must not contain missing values")
    return numeric.astype(float).copy()


def _numeric_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    malformed = numeric.isna() & ~frame.isna()
    if bool(malformed.any().any()):
        raise ValueError(f"{name} contains non-numeric values")
    values = numeric.to_numpy(dtype=float)
    if not bool((np.isfinite(values) | np.isnan(values)).all()):
        raise ValueError(f"{name} contains infinite values")
    return numeric.astype(float).copy()


def _validate_index(index: pd.Index, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} dates must use a DatetimeIndex")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be valid, unique, and increasing")


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "TRADING_DAYS_PER_YEAR",
    "calculate_drawdown",
    "calculate_par_proxy_unit_dv01",
    "calculate_performance_metrics",
    "fit_historical_nss_residuals",
    "generate_threshold_positions",
    "plot_backtest",
    "run_relative_value_backtest",
    "save_backtest_outputs",
    "transaction_cost_sensitivity",
]
