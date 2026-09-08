"""Offline-only tests for Treasury CMT data loading and validation."""

from datetime import UTC, datetime
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import fixed_income.data as treasury_data
from fixed_income.curves import CurveRepresentation
from fixed_income.data import (
    MATURITY_YEARS,
    MissingValuePolicy,
    TreasuryDataDownloadError,
    TreasuryDataError,
    fetch_fred_treasury_yields,
    load_offline_treasury_yields,
    load_treasury_yield_csv,
    load_treasury_yields,
    maturity_label_to_years,
    percentages_to_decimals,
    validate_treasury_yield_panel,
)


def test_offline_loader_returns_complete_decimal_cmt_panel() -> None:
    """Bundled observations load without network and retain their meaning."""
    result = load_offline_treasury_yields()

    assert result.is_offline
    assert result.fallback_reason is None
    assert result.representation is CurveRepresentation.TREASURY_CMT
    assert result.raw_units == "percent per annum"
    assert result.internal_units == "decimal annual rate"
    assert result.yields.shape[0] >= 100
    assert list(result.yields.columns) == list(MATURITY_YEARS)
    assert result.yields.index.is_monotonic_increasing
    assert result.yields.index.is_unique
    assert not result.yields.isna().any().any()
    assert result.yields.to_numpy().max() < 0.1


def test_percentage_to_decimal_conversion_is_exactly_scaled() -> None:
    """Published percentage points are divided by 100 once."""
    raw = pd.DataFrame({"3M": [4.25, 5.0]})

    converted = percentages_to_decimals(raw)

    np.testing.assert_allclose(converted["3M"], [0.0425, 0.05])


@pytest.mark.parametrize(
    "label, expected",
    [("3m", 0.25), ("6M", 0.5), ("1Y", 1.0), ("30y", 30.0)],
)
def test_maturity_label_conversion(label: str, expected: float) -> None:
    """Month and year tenors map to documented numerical years."""
    assert maturity_label_to_years(label) == expected


def test_missing_data_policy_is_explicit(tmp_path) -> None:
    """Missing cells can be rejected, dropped, or retained but never filled."""
    path = tmp_path / "missing.csv"
    path.write_text("Date,3M,1Y\n2024-01-02,5.40,4.80\n2024-01-03,.,4.75\n")

    with pytest.raises(TreasuryDataError, match="missing"):
        load_treasury_yield_csv(path, maturities=["3M", "1Y"], missing="raise")
    dropped = load_treasury_yield_csv(path, maturities=["3M", "1Y"], missing="drop")
    retained = load_treasury_yield_csv(path, maturities=["3M", "1Y"], missing="keep")

    assert len(dropped) == 1
    assert len(retained) == 2
    assert np.isnan(retained.iloc[1, 0])


def test_duplicate_and_unsorted_dates_are_rejected() -> None:
    """Historical ordering defects fail rather than being silently sorted."""
    duplicate = pd.DataFrame(
        {"1Y": [0.04, 0.041]},
        index=pd.to_datetime(["2024-01-02", "2024-01-02"]),
    )
    unsorted = pd.DataFrame(
        {"1Y": [0.04, 0.041]},
        index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
    )

    with pytest.raises(TreasuryDataError, match="duplicated"):
        validate_treasury_yield_panel(duplicate)
    with pytest.raises(TreasuryDataError, match="increasing"):
        validate_treasury_yield_panel(unsorted)


def test_malformed_columns_and_invalid_yields_are_rejected() -> None:
    """Unknown tenors, percentage-scale values, and infinities fail validation."""
    dates = pd.to_datetime(["2024-01-02"])
    with pytest.raises(TreasuryDataError, match="columns"):
        validate_treasury_yield_panel(pd.DataFrame({"4Y": [0.04]}, index=dates))
    with pytest.raises(TreasuryDataError, match="decimal"):
        validate_treasury_yield_panel(pd.DataFrame({"1Y": [4.0]}, index=dates))
    with pytest.raises(TreasuryDataError, match="infinite"):
        validate_treasury_yield_panel(pd.DataFrame({"1Y": [float("inf")]}, index=dates))


def test_mocked_fred_adapter_has_same_schema_without_network() -> None:
    """A mocked official response follows the offline schema and unit path."""
    csv = b"observation_date,DGS3MO,DGS1\n2024-01-02,5.46,4.80\n2024-01-03,5.48,4.81\n"

    def opener(*_args, **_kwargs):
        return BytesIO(csv)

    result = fetch_fred_treasury_yields(
        maturities=["3M", "1Y"], missing="raise", opener=opener
    )

    assert not result.is_offline
    assert list(result.yields.columns) == ["3M", "1Y"]
    assert result.yields.iloc[0, 0] == pytest.approx(0.0546)


def test_network_failure_uses_labelled_offline_fallback(monkeypatch) -> None:
    """Online failure is recorded when the bundled data are substituted."""

    def fail(**_kwargs):
        raise TreasuryDataDownloadError("network unavailable in test")

    monkeypatch.setattr(treasury_data, "fetch_fred_treasury_yields", fail)
    result = load_treasury_yields(
        maturities=["2Y", "10Y"],
        start_date="2024-01-02",
        end_date="2024-01-10",
    )

    assert result.is_offline
    assert result.fallback_reason == "network unavailable in test"
    assert list(result.yields.columns) == ["2Y", "10Y"]


def test_dataset_retrieval_metadata_is_timezone_aware() -> None:
    """Offline snapshot provenance carries an unambiguous UTC timestamp."""
    result = load_offline_treasury_yields(maturities=["10Y"])

    assert isinstance(result.retrieved_at, datetime)
    assert result.retrieved_at.tzinfo is UTC


def test_invalid_missing_policy_fails_clearly() -> None:
    """Loaders do not infer an undocumented missing-data convention."""
    with pytest.raises(ValueError, match="missing-value"):
        load_offline_treasury_yields(missing="fill")
    assert MissingValuePolicy.KEEP.value == "keep"
