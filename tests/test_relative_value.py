"""Offline tests for curve relative-value and butterfly analytics."""

import numpy as np
import pandas as pd
import pytest

from fixed_income.relative_value import (
    BASIS_POINT_DECIMAL,
    butterfly_report,
    butterfly_yield,
    calculate_curve_residuals,
    classify_residual,
    construct_dv01_neutral_butterfly,
    rank_curve_points,
    residual_report,
    residuals_to_basis_points,
    rolling_z_scores,
)


def test_residual_sign_convention_and_bp_conversion() -> None:
    """Residual is observed minus fitted and converts explicitly to bp."""
    dates = pd.date_range("2025-01-02", periods=2)
    observed = pd.DataFrame(
        {"2Y": [0.0410, 0.0420], "5Y": [0.0390, 0.0400]}, index=dates
    )
    fitted = pd.DataFrame({"5Y": [0.0400, 0.0400], "2Y": [0.0400, 0.0415]}, index=dates)

    residuals = calculate_curve_residuals(observed, fitted)

    assert residuals.loc[dates[0], "2Y"] == pytest.approx(0.0010)
    assert residuals.loc[dates[0], "5Y"] == pytest.approx(-0.0010)
    assert residuals_to_basis_points(residuals).loc[dates[0], "2Y"] == pytest.approx(
        10.0
    )
    assert BASIS_POINT_DECIMAL == 0.0001


def test_residual_report_retains_explicit_missing_fit() -> None:
    """A failed fitted point remains a labelled missing residual in long output."""
    dates = pd.date_range("2025-01-02", periods=1)
    observed = pd.DataFrame({"2Y": [0.0410], "5Y": [0.0390]}, index=dates)
    fitted = pd.DataFrame({"2Y": [0.0400], "5Y": [np.nan]}, index=dates)

    report = residual_report(observed, fitted)

    assert len(report) == 2
    missing = report.loc[report["tenor"] == "5Y"].iloc[0]
    assert np.isnan(missing["residual_decimal"])
    assert pd.isna(missing["classification"])


def test_rich_cheap_classification_and_ranking() -> None:
    """Negative is rich, positive is cheap, and rankings run in that order."""
    assert classify_residual(-0.0002) == "rich"
    assert classify_residual(0.0) == "on-curve"
    assert classify_residual(0.0002) == "cheap"
    residuals = pd.Series({"2Y": 0.0003, "5Y": -0.0002, "10Y": 0.0})

    ranking = rank_curve_points(residuals)

    assert list(ranking["tenor"]) == ["5Y", "10Y", "2Y"]
    assert list(ranking["classification"]) == ["rich", "on-curve", "cheap"]


def test_rolling_mean_and_std_use_only_history_through_t_minus_one() -> None:
    """Hand calculations establish both lagged rolling estimates."""
    dates = pd.date_range("2025-01-01", periods=5)
    residuals = pd.Series([0.0001, 0.0002, 0.0005, 0.0009, -0.0010], index=dates)

    result = rolling_z_scores(residuals, lookback=3, min_observations=3, ddof=1)
    column = result.z_scores.columns[0]
    history = np.array([0.0001, 0.0002, 0.0005])

    assert result.historical_mean_decimal.loc[dates[3], column] == pytest.approx(
        history.mean()
    )
    assert result.historical_std_decimal.loc[dates[3], column] == pytest.approx(
        history.std(ddof=1)
    )
    expected_z = (0.0009 - history.mean()) / history.std(ddof=1)
    assert result.z_scores.loc[dates[3], column] == pytest.approx(expected_z)


def test_rolling_signal_has_no_look_ahead_under_future_mutation() -> None:
    """Changing a future residual cannot alter prior estimates or signals."""
    dates = pd.date_range("2025-01-01", periods=8)
    original = pd.Series(np.arange(8, dtype=float) * 0.0001, index=dates, name="5Y")
    mutated = original.copy()
    mutated.iloc[-1] = 0.5

    first = rolling_z_scores(original, lookback=3, min_observations=3, ddof=1)
    second = rolling_z_scores(mutated, lookback=3, min_observations=3, ddof=1)

    pd.testing.assert_frame_equal(
        first.historical_mean_decimal.iloc[:-1],
        second.historical_mean_decimal.iloc[:-1],
    )
    pd.testing.assert_frame_equal(
        first.historical_std_decimal.iloc[:-1],
        second.historical_std_decimal.iloc[:-1],
    )
    pd.testing.assert_frame_equal(first.z_scores.iloc[:-1], second.z_scores.iloc[:-1])
    original_last_mean = first.historical_mean_decimal.iloc[-1, 0]
    mutated_last_mean = second.historical_mean_decimal.iloc[-1, 0]
    assert original_last_mean == mutated_last_mean


def test_zero_standard_deviation_policy_is_explicit() -> None:
    """A constant history gives unavailable z-score unless zero is requested."""
    dates = pd.date_range("2025-01-01", periods=4)
    constant = pd.Series([0.0001] * 4, index=dates)

    unavailable = rolling_z_scores(constant, lookback=3, min_observations=3, ddof=1)
    zeroed = rolling_z_scores(
        constant, lookback=3, min_observations=3, ddof=1, zero_std="zero"
    )

    assert np.isnan(unavailable.z_scores.iloc[-1, 0])
    assert zeroed.z_scores.iloc[-1, 0] == 0.0
    with pytest.raises(ValueError, match="standard deviation is zero"):
        rolling_z_scores(
            constant, lookback=3, min_observations=3, ddof=1, zero_std="raise"
        )


@pytest.mark.parametrize(
    ("belly_side", "expected_sides"),
    [("long", ["short", "long", "short"]), ("short", ["long", "short", "long"])],
)
def test_dv01_neutral_weights_total_and_signs(
    belly_side: str, expected_sides: list[str]
) -> None:
    """Each wing offsets half the belly DV01 and the signed total is zero."""
    unit_dv01 = {"2Y": 0.00018, "5Y": 0.00043, "10Y": 0.00078}

    hedge = construct_dv01_neutral_butterfly(
        unit_dv01, belly_notional=1_000_000.0, belly_side=belly_side
    )
    signed = hedge["signed_position_dv01_currency_per_bp"]

    assert list(hedge["side"]) == expected_sides
    assert abs(signed.iloc[0]) == pytest.approx(abs(signed.iloc[1]) / 2.0)
    assert abs(signed.iloc[2]) == pytest.approx(abs(signed.iloc[1]) / 2.0)
    assert signed.sum() == pytest.approx(0.0, abs=1e-12)
    assert hedge["aggregate_dv01_currency_per_bp"].iloc[0] == pytest.approx(
        0.0, abs=1e-12
    )


def test_butterfly_measure_and_complete_report() -> None:
    """The raw yield measure is separate from the complete risk-weighted table."""
    yields = {"2Y": 0.0400, "5Y": 0.0450, "10Y": 0.0470}
    unit_dv01 = {"2Y": 0.00018, "5Y": 0.00043, "10Y": 0.00078}
    residuals = {"2Y": -0.0001, "5Y": 0.0002, "10Y": 0.0}
    z_scores = {"2Y": -1.0, "5Y": 1.5, "10Y": 0.0}

    report = butterfly_report(yields, unit_dv01, residuals, z_scores)

    assert butterfly_yield(yields) == pytest.approx(0.0030)
    assert report["butterfly_yield_bp"].iloc[0] == pytest.approx(30.0)
    expected_columns = {
        "leg",
        "maturity_years",
        "yield_decimal",
        "notional",
        "side",
        "unit_dv01_currency_per_bp_per_notional",
        "signed_position_dv01_currency_per_bp",
        "curve_residual_decimal",
        "curve_residual_bp",
        "rolling_z_score",
        "aggregate_dv01_currency_per_bp",
    }
    assert expected_columns.issubset(report.columns)


@pytest.mark.parametrize(
    "values",
    [
        {"2Y": 0.04, "5Y": 0.05},
        {"2Y": 0.04, "5Y": 0.05, "7Y": 0.06},
        {"2Y": 0.04, 2.0: 0.04, "5Y": 0.05, "10Y": 0.06},
    ],
)
def test_invalid_or_missing_butterfly_tenors(values: dict[object, float]) -> None:
    """Missing 10Y and duplicate numerical tenor inputs are rejected."""
    with pytest.raises(ValueError, match="tenor"):
        butterfly_yield(values)


def test_residual_panels_require_matching_tenors() -> None:
    """Residual subtraction never silently aligns missing curve points."""
    dates = pd.date_range("2025-01-01", periods=2)
    observed = pd.DataFrame({"2Y": [0.04, 0.04]}, index=dates)
    fitted = pd.DataFrame({"5Y": [0.04, 0.04]}, index=dates)

    with pytest.raises(ValueError, match="tenor labels must match"):
        calculate_curve_residuals(observed, fitted)


@pytest.mark.parametrize("ddof, expected_std", [(0, np.sqrt(2 / 3)), (1, 1.0)])
def test_rolling_ddof_and_window_endpoints(ddof: int, expected_std: float) -> None:
    """Three historical values 1,2,3 bp have known variance and endpoints."""
    dates = pd.bdate_range("2025-01-01", periods=4)
    residuals = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=dates)
    result = rolling_z_scores(residuals, lookback=3, ddof=ddof)
    assert result.historical_std_decimal.iloc[-1, 0] == pytest.approx(
        expected_std * 0.0001
    )
    assert result.z_scores.iloc[-1, 0] == pytest.approx(2 / expected_std)
    assert result.window_start_dates.iloc[-1, 0] == dates[0]
    assert result.window_end_dates.iloc[-1, 0] == dates[2]


def test_yield_butterfly_cancels_parallel_move() -> None:
    """The raw fly is invariant under equal decimal yield shifts in all three legs."""
    curve = {"2Y": 0.04, "5Y": 0.042, "10Y": 0.048}
    shifted = {tenor: value + 0.005 for tenor, value in curve.items()}
    assert butterfly_yield(shifted) == pytest.approx(butterfly_yield(curve), abs=1e-16)
