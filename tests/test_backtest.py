"""Offline tests for the lagged historical relative-value backtest."""

import math

import numpy as np
import pandas as pd
import pytest

from fixed_income.backtest import (
    BacktestConfig,
    calculate_drawdown,
    calculate_par_proxy_unit_dv01,
    calculate_performance_metrics,
    generate_threshold_positions,
    run_relative_value_backtest,
    transaction_cost_sensitivity,
)


def _inputs(
    residual_values: list[float],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=len(residual_values), freq="B")
    yields = pd.DataFrame(
        {
            "2Y": np.full(len(dates), 0.04),
            "5Y": np.full(len(dates), 0.04),
            "10Y": np.full(len(dates), 0.04),
        },
        index=dates,
    )
    residual = pd.Series(residual_values, index=dates, name="5Y")
    unit_dv01 = pd.DataFrame(
        {
            "2Y": np.full(len(dates), 0.0002),
            "5Y": np.full(len(dates), 0.0004),
            "10Y": np.full(len(dates), 0.0008),
        },
        index=dates,
    )
    return yields, residual, unit_dv01


def _config(*, cost: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        lookback=2,
        min_observations=2,
        ddof=0,
        entry_z=2.0,
        exit_z=0.5,
        belly_notional_currency=1_000_000.0,
        transaction_cost_bps_per_face=cost,
        liquidate_at_end=False,
    )


def test_future_mutation_cannot_change_past_signals_positions_or_pnl() -> None:
    """Future residuals and yields cannot leak into any preceding result row."""
    yields, residual, unit_dv01 = _inputs([0.0, 0.0001, 0.0003, 0.0002, 0.0])
    first = run_relative_value_backtest(yields, residual, unit_dv01, config=_config())
    changed_yields = yields.copy()
    changed_residual = residual.copy()
    changed_yields.iloc[-1, :] = [0.08, 0.01, 0.09]
    changed_residual.iloc[-1] = -0.05
    second = run_relative_value_backtest(
        changed_yields, changed_residual, unit_dv01, config=_config()
    )

    pd.testing.assert_frame_equal(
        first.observations.iloc[:-1], second.observations.iloc[:-1]
    )


def test_signal_position_is_shifted_one_observation_for_pnl() -> None:
    """An entry signal changes target now but can only earn next-row P&L."""
    yields, residual, unit_dv01 = _inputs([0.0, 0.0001, 0.0003, 0.0004])
    yields.iloc[2, 1] = 0.0410
    yields.iloc[3, 1] = 0.0409
    result = run_relative_value_backtest(yields, residual, unit_dv01, config=_config())
    rows = result.observations

    assert rows.iloc[2]["decision_direction"] == 1
    assert rows.iloc[2]["held_direction_from_prior_date"] == 0
    assert rows.iloc[2]["gross_pnl_currency_approx"] == pytest.approx(0.0)
    assert rows.iloc[3]["held_direction_from_prior_date"] == 1
    assert rows.iloc[3]["gross_pnl_currency_approx"] == pytest.approx(400.0)


def test_cost_is_charged_on_position_changes_not_held_positions() -> None:
    """Entry and exit trade dates incur cost; an unchanged target does not."""
    yields, residual, unit_dv01 = _inputs(
        [0.0, 0.0001, 0.0003, 0.0004, 0.00035, 0.00035]
    )
    result = run_relative_value_backtest(
        yields, residual, unit_dv01, config=_config(cost=0.01)
    )
    rows = result.observations

    entry_row = rows.iloc[2]
    hold_row = rows.iloc[3]
    assert entry_row["transaction_cost_currency"] > 0.0
    assert entry_row["gross_pnl_currency_approx"] == pytest.approx(0.0)
    assert hold_row["transaction_cost_currency"] == pytest.approx(0.0)
    assert np.allclose(
        rows["net_pnl_currency_approx"],
        rows["gross_pnl_currency_approx"] - rows["transaction_cost_currency"],
    )


@pytest.mark.parametrize(
    ("residuals", "expected_direction", "expected_sides"),
    [
        ([0.0, 0.0001, 0.0003, 0.0004], 1, ("short", "long", "short")),
        ([0.0, -0.0001, -0.0003, -0.0004], -1, ("long", "short", "long")),
    ],
)
def test_signal_direction_and_position_signs(
    residuals: list[float],
    expected_direction: int,
    expected_sides: tuple[str, str, str],
) -> None:
    """Cheap means long belly; rich means short belly, with opposite wings."""
    yields, residual, unit_dv01 = _inputs(residuals)
    rows = run_relative_value_backtest(
        yields, residual, unit_dv01, config=_config()
    ).observations
    row = rows.iloc[2]
    notionals = [
        row[f"target_notional_{label}_currency"]
        for label in ("2y", "5y", "10y")
    ]
    sides = tuple("long" if value > 0 else "short" for value in notionals)

    assert row["decision_direction"] == expected_direction
    assert sides == expected_sides


def test_every_dated_target_and_held_portfolio_is_dv01_neutral() -> None:
    """Changing unit DV01s trigger new hedge weights that remain neutral."""
    yields, residual, unit_dv01 = _inputs([0.0, 0.0001, 0.0003, 0.0004, 0.0005])
    unit_dv01["2Y"] *= np.linspace(1.0, 1.2, len(unit_dv01))
    unit_dv01["10Y"] *= np.linspace(1.0, 0.8, len(unit_dv01))
    rows = run_relative_value_backtest(
        yields, residual, unit_dv01, config=_config()
    ).observations

    assert np.allclose(rows["aggregate_target_dv01_currency_per_bp"], 0.0, atol=1e-12)
    assert np.allclose(rows["aggregate_held_dv01_currency_per_bp"], 0.0, atol=1e-12)
    assert rows.iloc[3]["turnover_abs_notional_currency"] > 0.0


def test_first_order_pnl_sign_for_long_and_short_belly() -> None:
    """A long belly gains on a yield fall and a short belly gains on a rise."""
    cheap_yields, cheap_residual, unit_dv01 = _inputs([0.0, 0.0001, 0.0003, 0.0004])
    cheap_yields.iloc[3, 1] -= 0.0001
    cheap = run_relative_value_backtest(
        cheap_yields, cheap_residual, unit_dv01, config=_config()
    )
    rich_yields, rich_residual, rich_dv01 = _inputs(
        [0.0, -0.0001, -0.0003, -0.0004]
    )
    rich_yields.iloc[3, 1] += 0.0001
    rich = run_relative_value_backtest(
        rich_yields, rich_residual, rich_dv01, config=_config()
    )

    assert cheap.observations.iloc[3]["gross_pnl_currency_approx"] == pytest.approx(
        400.0
    )
    assert rich.observations.iloc[3]["gross_pnl_currency_approx"] == pytest.approx(
        400.0
    )


def test_drawdown_includes_zero_initial_high_water_mark() -> None:
    """Drawdown handles an initial loss and later peak-to-trough loss correctly."""
    dates = pd.date_range("2025-01-01", periods=5)
    cumulative = pd.Series([-2.0, 3.0, 1.0, 5.0, 0.0], index=dates)

    drawdown = calculate_drawdown(cumulative)

    assert list(drawdown) == pytest.approx([-2.0, 0.0, -2.0, 0.0, -5.0])


def test_sharpe_uses_sqrt_annualisation_and_sample_volatility() -> None:
    """P&L Sharpe is mean/sample-std times sqrt(observations per year)."""
    dates = pd.date_range("2025-01-01", periods=4)
    frame = pd.DataFrame(index=dates)
    frame["gross_pnl_currency_approx"] = [0.0, 1.0, 2.0, 3.0]
    frame["net_pnl_currency_approx"] = frame["gross_pnl_currency_approx"]
    frame["gross_drawdown_currency"] = 0.0
    frame["net_drawdown_currency"] = 0.0
    frame["held_direction_from_prior_date"] = [0, 1, 1, 1]
    frame["decision_direction"] = [1, 1, 1, 0]
    frame["turnover_abs_notional_currency"] = [1.0, 0.0, 0.0, 1.0]
    frame["transaction_cost_currency"] = 0.0

    metrics = calculate_performance_metrics(frame, annualisation_factor=252)
    pnl = np.array([1.0, 2.0, 3.0])
    expected = pnl.mean() / pnl.std(ddof=1) * math.sqrt(252)

    assert metrics.loc["sharpe_ratio_on_approx_pnl", "gross"] == pytest.approx(expected)
    assert metrics.loc[
        "annualised_mean_pnl_currency_approx", "gross"
    ] == pytest.approx(2.0 * 252)


def test_empty_position_period_has_zero_pnl_cost_and_time_in_market() -> None:
    """Unavailable or sub-threshold signals preserve genuinely empty periods."""
    yields, residual, unit_dv01 = _inputs([0.0, 0.00001, 0.00002, 0.000015])
    config = BacktestConfig(
        lookback=2,
        min_observations=2,
        ddof=0,
        entry_z=100.0,
        exit_z=0.5,
        transaction_cost_bps_per_face=1.0,
        liquidate_at_end=False,
    )
    result = run_relative_value_backtest(yields, residual, unit_dv01, config=config)

    assert not result.observations["decision_direction"].any()
    assert result.observations["gross_pnl_currency_approx"].eq(0.0).all()
    assert result.observations["transaction_cost_currency"].eq(0.0).all()
    assert result.metrics.loc["time_in_market_fraction", "net"] == 0.0
    assert math.isnan(result.metrics.loc["hit_rate_active_intervals", "net"])


def test_threshold_equalities_enter_exit_and_opposite_jump_reverses() -> None:
    """Threshold edge behavior is inclusive and deterministic."""
    dates = pd.date_range("2025-01-01", periods=6)
    z_scores = pd.Series([2.0, 1.0, 0.5, -2.0, -3.0, 2.0], index=dates)

    positions = generate_threshold_positions(z_scores, entry_z=2.0, exit_z=0.5)

    assert list(positions) == [1, 1, 0, -1, -1, 1]
    with pytest.raises(ValueError, match="strictly below"):
        generate_threshold_positions(z_scores, entry_z=1.0, exit_z=1.0)


def test_linear_transaction_cost_sensitivity() -> None:
    """Zero costs preserve gross P&L and doubling cost doubles total costs."""
    yields, residual, unit_dv01 = _inputs([0.0, 0.0001, 0.0003, 0.0002])
    table = transaction_cost_sensitivity(
        yields,
        residual,
        unit_dv01,
        cost_bps_per_face=(0.0, 0.01, 0.02),
        config=_config(),
    )

    assert table.iloc[0]["total_transaction_cost_currency"] == 0.0
    assert table.iloc[2]["total_transaction_cost_currency"] == pytest.approx(
        2.0 * table.iloc[1]["total_transaction_cost_currency"]
    )


def test_proxy_dv01_handles_february_schedule_boundaries() -> None:
    """Month-end shifts retain valid, positive time-varying proxy DV01s."""
    dates = pd.DatetimeIndex(["2024-02-28", "2024-02-29"])
    yields = pd.DataFrame(
        {"2Y": [0.04, 0.041], "5Y": [0.04, 0.041], "10Y": [0.04, 0.041]},
        index=dates,
    )

    result = calculate_par_proxy_unit_dv01(yields)

    assert result.shape == (2, 3)
    assert (result > 0.0).all().all()
    assert not np.allclose(result.iloc[0], result.iloc[1])
