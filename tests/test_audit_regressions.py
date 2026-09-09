"""Independent numerical and boundary regressions from the repository audit."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fixed_income.bonds import FixedRateBond, yield_to_maturity
from fixed_income.curves import (
    CompoundingConvention,
    CurveRepresentation,
    YieldCurve,
    discount_cash_flows,
    interpolate_zero_rates,
)
from fixed_income.data import (
    TreasuryDataError,
    load_treasury_yield_csv,
    validate_treasury_yield_panel,
)
from fixed_income.pca import fit_yield_change_pca
from fixed_income.risk import curve_shape_shock


def test_ytm_bracket_does_not_overflow_for_long_bond() -> None:
    """The par identity determines YTM without evaluating huge negative-yield PVs."""
    bond = FixedRateBond(date(2025, 1, 15), date(3025, 1, 15), 0.04)
    assert yield_to_maturity(
        bond, bond.accrual_start_date, 100.0, price_type="dirty"
    ) == pytest.approx(0.04, abs=1e-12)


def test_time_zero_cash_flow_needs_no_extrapolation() -> None:
    """Immediate currency payments have discount factor one regardless of nodes."""
    curve = YieldCurve(
        (1.0, 2.0),
        (0.03, 0.05),
        CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.CONTINUOUS,
    )
    result = discount_cash_flows([0.0, 1.0], [7.0, 100.0], curve)
    assert result.discount_factors[0] == 1.0
    assert result.total_present_value == pytest.approx(7.0 + 100.0 / np.exp(0.03))
    with pytest.raises(ValueError, match="outside"):
        curve.discount_factors([0.5])


def test_shape_shock_preserves_unrepresented_anchors() -> None:
    """A zero-shock 10Y pivot must survive when base nodes are only 1Y and 30Y."""
    curve = YieldCurve(
        (1.0, 30.0),
        (0.04, 0.04),
        CurveRepresentation.ZERO_RATE,
        compounding=CompoundingConvention.CONTINUOUS,
    )
    shocked = curve_shape_shock(curve, "steepener")
    rates = interpolate_zero_rates(
        shocked.maturities_years, shocked.values, [1.0, 2.0, 6.0, 10.0, 20.0, 30.0]
    )
    np.testing.assert_allclose(
        rates, [0.0375, 0.0375, 0.03875, 0.04, 0.04125, 0.0425], atol=1e-15
    )


def test_intraday_rows_cannot_collapse_to_duplicate_dates() -> None:
    """Normalization must not turn two timestamps into a valid-looking daily panel."""
    panel = pd.DataFrame(
        {"1Y": [0.04, 0.05]},
        index=pd.to_datetime(["2025-01-01 09:00", "2025-01-01 16:00"]),
    )
    with pytest.raises(TreasuryDataError, match="date|daily"):
        validate_treasury_yield_panel(panel)


@pytest.mark.parametrize("missing", ["drop", "keep", "raise"])
def test_decimal_csv_cannot_hide_corruption_as_missing(tmp_path, missing: str) -> None:
    """A typo is a schema error under every missing-observation policy."""
    path = tmp_path / "corrupt.csv"
    path.write_text("Date,1Y\n2024-01-02,0.04\n2024-01-03,typo\n")
    with pytest.raises(TreasuryDataError, match="non-numeric|malformed"):
        load_treasury_yield_csv(
            path, maturities=["1Y"], raw_units="decimal", missing=missing
        )


def test_covariance_pca_is_scale_equivariant() -> None:
    """Changing decimal-change scale changes eigenvalues quadratically, not rank."""
    changes = np.array([[1, 2, 3], [-1, -2, -3], [2, 4, 6], [-2, -4, -6]])
    dates = pd.date_range("2025-01-01", periods=5)

    def fit(scale: float):
        levels = np.vstack([np.zeros(3), np.cumsum(changes * scale, axis=0)])
        return fit_yield_change_pca(
            pd.DataFrame(levels, index=dates, columns=["1Y", "2Y", "5Y"])
        )

    large, small = fit(1e-4), fit(1e-8)
    np.testing.assert_allclose(small.eigenvalues, large.eigenvalues * 1e-8, atol=1e-29)
    np.testing.assert_allclose(small.loadings[:, 0], large.loadings[:, 0], atol=1e-14)
    assert small.explained_variance_ratios[0] == pytest.approx(1.0)


def test_risk_workflow_handles_coupon_before_first_cmt_tenor(monkeypatch) -> None:
    """A next-day coupon needs an explicit sub-3M discounting assumption."""
    import fixed_income.workflows as workflows

    market = replace(
        workflows.load_market_state(offline=True), valuation_date=date(2024, 9, 29)
    )
    monkeypatch.setattr(
        workflows, "_save_csv", lambda _frame, name: workflows.OUTPUTS_DIRECTORY / name
    )
    risk, keys = workflows.run_risk_workflow(offline=True, state=market)
    assert len(risk) == 8
    assert np.isfinite(risk["exact_pnl_currency"]).all()
    assert np.isfinite(keys["key_rate_dv01_currency_per_bp"]).all()
    assert risk["curve_extrapolation"].eq("flat").all()
