"""Treasury constant-maturity yield data with an offline official-data sample.

The supported series are daily H.15 Treasury constant-maturity yields made
available by the Federal Reserve Bank of St. Louis (FRED) and sourced from the
Board of Governors of the Federal Reserve System. Published values are percent
per annum and are converted exactly once to decimal annual rates on loading.

These observations are retained as Treasury CMT yields. They are par-yield-like
statistical series, not automatically bootstrapped zero rates or discount
factors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fixed_income.curves import CurveRepresentation

PERCENT_TO_DECIMAL = 0.01
"""Exact multiplier converting a published percentage rate to decimal form."""

MATURITY_YEARS: dict[str, float] = {
    "3M": 0.25,
    "6M": 0.5,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}
"""Supported maturity labels mapped to numerical years."""

FRED_SERIES_BY_MATURITY: dict[str, str] = {
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}
"""FRED identifiers for the supported daily H.15 CMT series."""

FRED_SOURCE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OFFLINE_SAMPLE_PATH = (
    Path(__file__).resolve().parent / "sample_data" / "sample_treasury_yields.csv"
)


class MissingValuePolicy(StrEnum):
    """Explicit handling policy for missing Treasury yield observations."""

    RAISE = "raise"
    DROP = "drop"
    KEEP = "keep"


class TreasuryDataError(ValueError):
    """A Treasury dataset failed schema, unit, or observation validation."""


class TreasuryDataDownloadError(RuntimeError):
    """The optional official online Treasury data download failed."""


@dataclass(frozen=True)
class TreasuryYieldDataset:
    """Validated historical Treasury CMT panel plus provenance.

    Attributes:
        yields: Date-indexed frame of decimal annual CMT yields. ``0.04`` means
            4%. Columns are standard maturity labels.
        maturity_years: Column labels mapped to numerical years.
        representation: Always the explicit Treasury CMT representation.
        source: Human-readable source or offline fallback description.
        source_url: Official source URL, when applicable.
        retrieved_at: UTC retrieval time for online data, or the bundled
            snapshot creation date.
        raw_units: Units before loading, normally published percent per annum.
        internal_units: Units in ``yields``, always decimal annual rates.
        missing_value_policy: Explicit policy applied during loading.
        is_offline: Whether the returned rows came from the bundled CSV.
        fallback_reason: Online error text when an offline fallback was used.
    """

    yields: pd.DataFrame
    maturity_years: dict[str, float]
    representation: CurveRepresentation
    source: str
    source_url: str | None
    retrieved_at: datetime
    raw_units: str
    internal_units: str
    missing_value_policy: MissingValuePolicy
    is_offline: bool
    fallback_reason: str | None = None


def maturity_label_to_years(label: str) -> float:
    """Convert a supported maturity label to numerical years.

    Args:
        label: Case-insensitive tenor such as ``"3M"`` or ``"10Y"``.

    Returns:
        Maturity in years, where 3 months is represented as ``0.25`` years.
    """
    if not isinstance(label, str):
        raise TypeError("maturity label must be a string")
    normalized = label.strip().upper()
    try:
        return MATURITY_YEARS[normalized]
    except KeyError as exc:
        supported = ", ".join(MATURITY_YEARS)
        raise ValueError(
            f"unsupported Treasury maturity label {label!r}; supported: {supported}"
        ) from exc


def percentages_to_decimals(
    values: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Convert numeric annual percentage yields to decimal annual rates.

    A published value of ``4.25`` percent becomes ``0.0425``. Missing values
    remain missing for subsequent explicit policy handling.
    """
    if not isinstance(values, (pd.DataFrame, pd.Series)):
        raise TypeError("values must be a pandas DataFrame or Series")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    newly_missing = numeric.isna() & ~values.isna()
    if bool(newly_missing.to_numpy().any()):
        raise TreasuryDataError("yield values contain malformed non-numeric entries")
    return numeric * PERCENT_TO_DECIMAL


def validate_treasury_yield_panel(
    panel: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Validate a date-by-maturity panel of decimal annual Treasury CMT yields.

    Args:
        panel: DataFrame with a strictly increasing, unique ``DatetimeIndex``
            and supported maturity-label columns. Values are decimal annual
            yields; percentages are rejected when they exceed plausible
            decimal bounds.
        allow_missing: If true, NaN cells are retained. Infinite values and
            malformed entries are rejected under either setting.

    Returns:
        A defensive copy with float columns and a normalized date-only index.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise TreasuryDataError("Treasury yield panel must not be empty")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TreasuryDataError("Treasury yield panel index must be a DatetimeIndex")
    if panel.index.hasnans:
        raise TreasuryDataError("Treasury yield panel contains invalid dates")
    if panel.index.has_duplicates:
        raise TreasuryDataError("Treasury yield panel contains duplicated dates")
    if panel.index.tz is not None or not panel.index.equals(panel.index.normalize()):
        raise TreasuryDataError(
            "daily Treasury dates must be timezone-naive midnight dates"
        )
    if not panel.index.is_monotonic_increasing:
        raise TreasuryDataError("Treasury yield panel dates must be increasing")
    if panel.columns.has_duplicates:
        raise TreasuryDataError("Treasury yield panel contains duplicated columns")
    if len(panel.columns) == 0:
        raise TreasuryDataError("Treasury yield panel must contain maturity columns")
    invalid_columns = [
        column for column in panel.columns if column not in MATURITY_YEARS
    ]
    if invalid_columns:
        raise TreasuryDataError(
            f"malformed or unsupported maturity columns: {invalid_columns}"
        )

    numeric = panel.apply(pd.to_numeric, errors="coerce")
    malformed = numeric.isna() & ~panel.isna()
    if bool(malformed.to_numpy().any()):
        raise TreasuryDataError("Treasury yield panel contains non-numeric yields")
    finite_or_missing = np.isfinite(numeric.to_numpy()) | numeric.isna().to_numpy()
    if not bool(finite_or_missing.all()):
        raise TreasuryDataError("Treasury yield panel contains infinite yields")
    if not allow_missing and bool(numeric.isna().to_numpy().any()):
        raise TreasuryDataError("Treasury yield panel contains missing values")
    present = numeric.to_numpy()[~numeric.isna().to_numpy()]
    if np.any((present < -0.2) | (present > 1.0)):
        raise TreasuryDataError(
            "invalid decimal yields; expected annual rates between -0.2 and 1.0"
        )

    validated = numeric.astype(float).copy()
    validated.index = validated.index.normalize()
    validated.index.name = "date"
    ordered_columns = sorted(validated.columns, key=MATURITY_YEARS.__getitem__)
    return validated.loc[:, ordered_columns]


def load_treasury_yield_csv(
    path: str | Path,
    *,
    maturities: Sequence[str] | None = None,
    raw_units: str = "percent",
    missing: MissingValuePolicy | str = MissingValuePolicy.RAISE,
) -> pd.DataFrame:
    """Load and validate a Treasury CMT CSV into decimal annual rates.

    The CSV must have one date column named ``date``, ``DATE``,
    ``observation_date``, or ``Date`` and maturity columns named either by
    labels (for example ``10Y``) or supported FRED series IDs. ``raw_units`` is
    explicitly ``"percent"`` or ``"decimal"``. Missing rows are raised,
    dropped, or retained according to ``missing``; no interpolation or
    forward/backward fill is performed.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Treasury yield CSV not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, na_values=[".", "NA", "N/A", "null"])
    except (OSError, pd.errors.ParserError) as exc:
        raise TreasuryDataError(f"could not parse Treasury yield CSV: {exc}") from exc
    return _normalise_raw_frame(
        frame,
        maturities=maturities,
        raw_units=raw_units,
        missing=missing,
    )


def load_offline_treasury_yields(
    *,
    maturities: Sequence[str] | None = None,
    missing: MissingValuePolicy | str = MissingValuePolicy.RAISE,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> TreasuryYieldDataset:
    """Load the bundled official H.15/FRED sample without internet access.

    Returned yields are decimal annual Treasury CMT rates indexed by date.
    The bundled raw CSV contains percent observations from 2024 and is loaded
    through the same schema and unit-conversion path as online data.
    """
    policy = _normalise_missing_policy(missing)
    panel = load_treasury_yield_csv(
        OFFLINE_SAMPLE_PATH,
        maturities=maturities,
        raw_units="percent",
        missing=policy,
    )
    panel = _filter_dates(panel, start_date, end_date)
    return _build_dataset(
        panel,
        source=(
            "Bundled snapshot of Federal Reserve H.15 Treasury constant-maturity "
            "series retrieved through FRED"
        ),
        source_url=FRED_SOURCE_URL,
        retrieved_at=datetime(2026, 9, 8, tzinfo=UTC),
        policy=policy,
        is_offline=True,
    )


def fetch_fred_treasury_yields(
    *,
    maturities: Sequence[str] | None = None,
    missing: MissingValuePolicy | str = MissingValuePolicy.DROP,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    timeout_seconds: float = 15.0,
    opener: Callable[..., object] = urlopen,
) -> TreasuryYieldDataset:
    """Download official daily H.15 Treasury CMT yields through FRED.

    Args:
        maturities: Supported maturity labels. Defaults to all ten labels.
        missing: Explicit missing-observation handling; no values are filled.
        start_date: Optional inclusive ISO date bound.
        end_date: Optional inclusive ISO date bound.
        timeout_seconds: Positive network timeout in seconds.
        opener: Injectable URL opener for offline unit tests.

    Returns:
        A validated date-indexed panel of decimal annual CMT yields plus source
        metadata. FRED publishes these series in percent and they are divided
        by 100 exactly once.
    """
    labels = _normalise_maturities(maturities)
    policy = _normalise_missing_policy(missing)
    if isinstance(timeout_seconds, bool) or not np.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite positive number")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive number")
    requested_start = _parse_optional_date(start_date, "start_date")
    requested_end = _parse_optional_date(end_date, "end_date")
    if requested_start and requested_end and requested_start > requested_end:
        raise ValueError("start_date must not be after end_date")
    params = {"id": ",".join(FRED_SERIES_BY_MATURITY[label] for label in labels)}
    if requested_start:
        params["cosd"] = requested_start.isoformat()
    if requested_end:
        params["coed"] = requested_end.isoformat()
    request = Request(
        f"{FRED_SOURCE_URL}?{urlencode(params, safe=',')}",
        headers={"User-Agent": "fixed-income-yield-curve-engine/0.1"},
    )
    try:
        response = opener(request, timeout=float(timeout_seconds))
        with response:  # type: ignore[attr-defined]
            payload = response.read()  # type: ignore[attr-defined]
        frame = pd.read_csv(BytesIO(payload), na_values=[".", "NA", "N/A"])
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise TreasuryDataDownloadError(
            f"official FRED Treasury yield download failed: {exc}"
        ) from exc
    panel = _normalise_raw_frame(
        frame,
        maturities=labels,
        raw_units="percent",
        missing=policy,
        start_date=requested_start,
        end_date=requested_end,
    )
    panel = _filter_dates(panel, requested_start, requested_end)
    return _build_dataset(
        panel,
        source=(
            "Federal Reserve H.15 Treasury constant-maturity series via FRED; "
            "source agency: Board of Governors of the Federal Reserve System"
        ),
        source_url=request.full_url,
        retrieved_at=datetime.now(UTC),
        policy=policy,
        is_offline=False,
    )


def load_treasury_yields(
    *,
    offline: bool = False,
    fallback_to_offline: bool = True,
    maturities: Sequence[str] | None = None,
    missing: MissingValuePolicy | str = MissingValuePolicy.DROP,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    timeout_seconds: float = 15.0,
) -> TreasuryYieldDataset:
    """Load decimal annual CMT yields online or from the offline fallback.

    ``offline=True`` performs no network call. Otherwise the official FRED
    adapter is tried first. If it fails and ``fallback_to_offline=True``, the
    bundled official-data snapshot is returned with ``fallback_reason`` set.
    Date bounds that exclude the snapshot still raise rather than silently
    returning data outside the requested range.
    """
    if offline:
        return load_offline_treasury_yields(
            maturities=maturities,
            missing=missing,
            start_date=start_date,
            end_date=end_date,
        )
    try:
        return fetch_fred_treasury_yields(
            maturities=maturities,
            missing=missing,
            start_date=start_date,
            end_date=end_date,
            timeout_seconds=timeout_seconds,
        )
    except (TreasuryDataDownloadError, TreasuryDataError) as exc:
        if not fallback_to_offline:
            raise
        fallback = load_offline_treasury_yields(
            maturities=maturities,
            missing=missing,
            start_date=start_date,
            end_date=end_date,
        )
        return TreasuryYieldDataset(
            yields=fallback.yields,
            maturity_years=fallback.maturity_years,
            representation=fallback.representation,
            source=fallback.source,
            source_url=fallback.source_url,
            retrieved_at=fallback.retrieved_at,
            raw_units=fallback.raw_units,
            internal_units=fallback.internal_units,
            missing_value_policy=fallback.missing_value_policy,
            is_offline=True,
            fallback_reason=str(exc),
        )


def _normalise_raw_frame(
    frame: pd.DataFrame,
    *,
    maturities: Sequence[str] | None,
    raw_units: str,
    missing: MissingValuePolicy | str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    labels = _normalise_maturities(maturities)
    policy = _normalise_missing_policy(missing)
    date_candidates = [
        column
        for column in frame.columns
        if str(column) in {"date", "DATE", "Date", "observation_date"}
    ]
    if len(date_candidates) != 1:
        raise TreasuryDataError(
            "CSV must contain exactly one date/DATE/Date/observation_date column"
        )
    rename = {series_id: label for label, series_id in FRED_SERIES_BY_MATURITY.items()}
    normalized = frame.rename(columns=rename)
    missing_columns = [label for label in labels if label not in normalized.columns]
    if missing_columns:
        raise TreasuryDataError(f"CSV is missing maturity columns: {missing_columns}")
    unexpected = [
        column
        for column in normalized.columns
        if column != date_candidates[0] and column not in MATURITY_YEARS
    ]
    if unexpected:
        raise TreasuryDataError(
            f"CSV has malformed or unsupported columns: {unexpected}"
        )
    date_strings = normalized[date_candidates[0]]
    parsed_dates = pd.to_datetime(date_strings, errors="coerce")
    if bool(parsed_dates.isna().any()):
        raise TreasuryDataError("CSV contains malformed dates")
    panel = normalized.loc[:, labels].copy()
    if raw_units == "percent":
        panel = percentages_to_decimals(panel)
    elif raw_units == "decimal":
        numeric = panel.apply(pd.to_numeric, errors="coerce")
        if bool((numeric.isna() & ~panel.isna()).to_numpy().any()):
            raise TreasuryDataError(
                "yield values contain malformed non-numeric entries"
            )
        panel = numeric
    else:
        raise ValueError("raw_units must be explicitly 'percent' or 'decimal'")
    panel.index = pd.DatetimeIndex(parsed_dates, name="date")
    # Validate dates, units and malformed cells before a missing policy can drop
    # rows and conceal a schema error.
    panel = validate_treasury_yield_panel(panel, allow_missing=True)
    panel = _filter_dates(panel, start_date, end_date)
    panel.attrs["missing_dates"] = [
        timestamp.date().isoformat()
        for timestamp in panel.index[panel.isna().any(axis=1)]
    ]
    panel = _apply_missing_policy(panel, policy)
    return validate_treasury_yield_panel(
        panel, allow_missing=policy is MissingValuePolicy.KEEP
    )


def _apply_missing_policy(
    panel: pd.DataFrame, policy: MissingValuePolicy
) -> pd.DataFrame:
    has_missing = bool(panel.isna().to_numpy().any())
    if has_missing and policy is MissingValuePolicy.RAISE:
        counts = panel.isna().sum()
        detail = {column: int(count) for column, count in counts.items() if count}
        raise TreasuryDataError(f"Treasury yield data contain missing values: {detail}")
    if policy is MissingValuePolicy.DROP:
        panel = panel.dropna(axis="index", how="any")
        if panel.empty:
            raise TreasuryDataError(
                "no complete Treasury yield rows remain after dropping"
            )
    return panel


def _normalise_missing_policy(
    policy: MissingValuePolicy | str,
) -> MissingValuePolicy:
    try:
        return MissingValuePolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unsupported missing-value policy: {policy!r}") from exc


def _normalise_maturities(maturities: Sequence[str] | None) -> tuple[str, ...]:
    if maturities is None:
        return tuple(MATURITY_YEARS)
    if isinstance(maturities, str):
        raise TypeError("maturities must be a sequence of labels, not one string")
    labels = tuple(str(label).strip().upper() for label in maturities)
    if not labels:
        raise ValueError("maturities must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("maturities must be unique")
    for label in labels:
        maturity_label_to_years(label)
    return tuple(sorted(labels, key=MATURITY_YEARS.__getitem__))


def _parse_optional_date(value: date | str | None, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        raise TypeError(f"{name} must be a date without a time component")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must use ISO YYYY-MM-DD format") from exc
    raise TypeError(f"{name} must be a datetime.date or ISO date string")


def _filter_dates(
    panel: pd.DataFrame,
    start_date: date | str | None,
    end_date: date | str | None,
) -> pd.DataFrame:
    start = _parse_optional_date(start_date, "start_date")
    end = _parse_optional_date(end_date, "end_date")
    if start and end and start > end:
        raise ValueError("start_date must not be after end_date")
    filtered = panel
    if start:
        filtered = filtered.loc[filtered.index.date >= start]
    if end:
        filtered = filtered.loc[filtered.index.date <= end]
    if filtered.empty:
        raise TreasuryDataError(
            "no Treasury yield observations in requested date range"
        )
    return filtered.copy()


def _build_dataset(
    panel: pd.DataFrame,
    *,
    source: str,
    source_url: str | None,
    retrieved_at: datetime,
    policy: MissingValuePolicy,
    is_offline: bool,
) -> TreasuryYieldDataset:
    return TreasuryYieldDataset(
        yields=panel,
        maturity_years={column: MATURITY_YEARS[column] for column in panel.columns},
        representation=CurveRepresentation.TREASURY_CMT,
        source=source,
        source_url=source_url,
        retrieved_at=retrieved_at,
        raw_units="percent per annum",
        internal_units="decimal annual rate",
        missing_value_policy=policy,
        is_offline=is_offline,
    )
