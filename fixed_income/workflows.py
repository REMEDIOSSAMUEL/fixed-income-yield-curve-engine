"""Reusable command-line workflows for the fixed-income research engine.

This module composes the package's calculation APIs and handles terminal and
artifact presentation. It does not introduce alternative pricing or risk
formulas. Treasury constant-maturity yields retain their published meaning;
where the risk demonstration needs zero rates, it uses a separately tagged and
prominently labelled illustrative proxy assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fixed_income.bonds import (
    FixedRateBond,
    PriceType,
    accrued_interest,
    clean_price_from_ytm,
    convexity,
    dirty_price_from_ytm,
    dv01,
    macaulay_duration,
    modified_duration,
    yield_to_maturity,
)
from fixed_income.curves import (
    CompoundingConvention,
    CurveRepresentation,
    NSSCalibrationResult,
    YieldCurve,
    calibrate_nss,
)
from fixed_income.data import (
    MissingValuePolicy,
    TreasuryYieldDataset,
    load_treasury_yields,
)
from fixed_income.pca import PCAResult, fit_yield_change_pca, interpret_pca_factors
from fixed_income.plotting import plot_nss_curve, plot_pca_loadings, save_figure
from fixed_income.relative_value import (
    butterfly_yield,
    construct_dv01_neutral_butterfly,
    rank_curve_points,
)
from fixed_income.risk import bond_risk_report, key_rate_dv01_report

BASIS_POINT_DECIMAL = 0.0001
"""One basis point expressed as a decimal annual yield."""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIRECTORY = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class MarketState:
    """Latest CMT cross-section and NSS fit, with rates in decimal annual units."""

    dataset: TreasuryYieldDataset
    valuation_date: date
    maturity_labels: tuple[str, ...]
    maturities_years: np.ndarray
    observed_yields_decimal: pd.Series
    nss_fit: NSSCalibrationResult


@dataclass(frozen=True)
class BondAnalytics:
    """YTM-based bond analytics with prices in currency and risk in stated units.

    Durations are years, convexity is years squared, YTM is a decimal nominal
    annual rate, and DV01 is currency per basis point for the supplied face.
    """

    bond: FixedRateBond
    settlement_date: date
    clean_price_currency: float
    dirty_price_currency: float
    accrued_interest_currency: float
    ytm_decimal: float
    macaulay_duration_years: float
    modified_duration_years: float
    convexity_years_squared: float
    dv01_currency_per_bp: float


def load_market_state(*, offline: bool) -> MarketState:
    """Load CMT history and fit the latest curve using decimal annual yields.

    Args:
        offline: When true, load only the bundled sample and make no network call.

    Returns:
        Historical source metadata, the latest observation date, maturities in
        years, decimal CMT yields, and an NSS calibration retaining CMT meaning.
    """
    dataset = load_treasury_yields(
        offline=offline,
        fallback_to_offline=True,
        missing=MissingValuePolicy.DROP,
    )
    latest = dataset.yields.iloc[-1].copy()
    labels = tuple(str(column) for column in latest.index)
    maturities = np.asarray(
        [dataset.maturity_years[label] for label in labels], dtype=float
    )
    fit = calibrate_nss(
        maturities,
        latest.to_numpy(dtype=float),
        input_representation=CurveRepresentation.TREASURY_CMT,
    )
    return MarketState(
        dataset=dataset,
        valuation_date=dataset.yields.index[-1].date(),
        maturity_labels=labels,
        maturities_years=maturities,
        observed_yields_decimal=latest,
        nss_fit=fit,
    )


def representative_bond(
    *,
    face_value: float = 100.0,
    coupon_rate: float = 0.0425,
) -> FixedRateBond:
    """Return the demo bond; face is currency and coupon is a decimal annual rate.

    The regular semiannual bullet runs from 31 March 2020 to 31 March 2034 and
    uses the package's Actual/Actual Treasury-style coupon-period convention.
    """
    return FixedRateBond(
        accrual_start_date=date(2020, 3, 31),
        maturity_date=date(2034, 3, 31),
        coupon_rate=coupon_rate,
        face_value=face_value,
        frequency=2,
    )


def calculate_bond_analytics(
    bond: FixedRateBond, settlement_date: date, pricing_ytm_decimal: float
) -> BondAnalytics:
    """Calculate explicit YTM-based analytics for one fixed-rate bond.

    Args:
        bond: Contract terms; face and returned prices share currency units.
        settlement_date: Calendar settlement date before maturity.
        pricing_ytm_decimal: Nominal annual decimal YTM, compounded at the
            bond's coupon frequency.

    Returns:
        Clean, dirty and accrued currency amounts; solved decimal YTM;
        durations in years; convexity in years squared; and currency DV01 per bp.
    """
    clean_price = clean_price_from_ytm(bond, settlement_date, pricing_ytm_decimal)
    dirty_price = dirty_price_from_ytm(bond, settlement_date, pricing_ytm_decimal)
    solved_ytm = yield_to_maturity(
        bond,
        settlement_date,
        clean_price,
        price_type=PriceType.CLEAN,
    )
    return BondAnalytics(
        bond=bond,
        settlement_date=settlement_date,
        clean_price_currency=clean_price,
        dirty_price_currency=dirty_price,
        accrued_interest_currency=accrued_interest(bond, settlement_date),
        ytm_decimal=solved_ytm,
        macaulay_duration_years=macaulay_duration(bond, settlement_date, solved_ytm),
        modified_duration_years=modified_duration(bond, settlement_date, solved_ytm),
        convexity_years_squared=convexity(bond, settlement_date, solved_ytm),
        dv01_currency_per_bp=dv01(bond, settlement_date, solved_ytm),
    )


def illustrative_zero_rate_proxy(state: MarketState) -> YieldCurve:
    """Copy decimal CMT nodes into an explicitly labelled zero-rate proxy.

    This is a scenario-analysis approximation, not a bootstrap or an automatic
    representation conversion. The returned rates are assumed continuously
    compounded solely for illustrative discounting and curve shocks.
    """
    return YieldCurve(
        maturities_years=tuple(float(value) for value in state.maturities_years),
        values=tuple(float(value) for value in state.observed_yields_decimal),
        representation=CurveRepresentation.ZERO_RATE,
        valuation_date=state.valuation_date,
        compounding=CompoundingConvention.CONTINUOUS,
        source=(
            "Illustrative continuously compounded zero-rate proxy copied from "
            "Treasury CMT observations; not bootstrapped"
        ),
    )


def run_curve_workflow(*, offline: bool) -> MarketState:
    """Display and plot the latest CMT/NSS curve; all rates are decimal internally."""
    state = load_market_state(offline=offline)
    _print_curve_summary(state)
    figure, _ = plot_nss_curve(state.nss_fit)
    try:
        path = save_figure(figure, "outputs/yield_curve.png")
    finally:
        plt.close(figure)
    print(f"Saved: {_relative_path(path)}")
    return state


def run_bond_workflow(
    *,
    bond: FixedRateBond | None = None,
    settlement_date: date = date(2024, 6, 28),
    ytm_decimal: float = 0.0436,
) -> BondAnalytics:
    """Display conventional bond analytics using decimal YTM and currency prices."""
    terms = representative_bond() if bond is None else bond
    analytics = calculate_bond_analytics(terms, settlement_date, ytm_decimal)
    _print_bond_summary(analytics)
    return analytics


def run_risk_workflow(
    *, offline: bool, state: MarketState | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run curve shocks and key-rate DV01, returning currency-valued reports.

    Shock sizes are basis points and internally convert using 1 bp = 0.0001.
    Prices and P&L are currency amounts per 100 face for the demo bond; key-rate
    DV01 is currency per bp.
    """
    market = load_market_state(offline=offline) if state is None else state
    zero_proxy = illustrative_zero_rate_proxy(market)
    bond = representative_bond()
    print("\nCurve risk (illustrative zero-rate proxy)")
    print("-----------------------------------------")
    print(
        "Assumption: latest CMT yields are copied into a continuously compounded\n"
        "zero-rate proxy for this scenario only; this is not a bootstrapped curve."
    )
    risk = bond_risk_report(
        bond,
        market.valuation_date,
        zero_proxy,
        include_shape_scenarios=False,
    )
    key_rates = key_rate_dv01_report(bond, market.valuation_date, zero_proxy)
    for report in (risk, key_rates):
        report.insert(0, "valuation_date", market.valuation_date.isoformat())
        report["curve_representation"] = zero_proxy.representation.value
        report["curve_compounding"] = zero_proxy.compounding.value
        report["curve_source_and_assumption"] = zero_proxy.source
    risk_path = _save_csv(risk, "bond_risk_report.csv")
    key_path = _save_csv(key_rates, "key_rate_dv01.csv")
    risk_columns = [
        "scenario",
        "parallel_shock_bp",
        "exact_pnl_currency",
        "first_order_pnl_currency",
        "duration_convexity_pnl_currency",
    ]
    print("\nParallel shocks: full revaluation versus approximations")
    print(
        risk.loc[:, risk_columns].to_string(
            index=False, float_format="{:,.6f}".format
        )
    )
    key_columns = [
        "key_tenor",
        "key_rate_dv01_currency_per_bp",
        "aggregate_key_rate_dv01_currency_per_bp",
        "parallel_dv01_currency_per_bp",
    ]
    print("\nKey-rate DV01")
    print(
        key_rates.loc[:, key_columns].to_string(
            index=False, float_format="{:,.6f}".format
        )
    )
    print(f"Saved: {_relative_path(risk_path)}, {_relative_path(key_path)}")
    return risk, key_rates


def run_pca_workflow(
    *, offline: bool, state: MarketState | None = None
) -> PCAResult:
    """Fit and plot covariance PCA of daily decimal CMT yield changes."""
    market = load_market_state(offline=offline) if state is None else state
    result = fit_yield_change_pca(market.dataset.yields)
    diagnostics = interpret_pca_factors(result, max_components=3)
    rows = []
    for index, diagnostic in enumerate(diagnostics):
        label = diagnostic.suggested_label or "unlabelled (ambiguous shape)"
        rows.append(
            {
                "component": f"PC{index + 1}",
                "explained_variance_percent": (
                    result.explained_variance_ratios[index] * 100.0
                ),
                "qualitative_interpretation": label,
            }
        )
    print("\nHistorical yield-change PCA")
    print("---------------------------")
    print(
        pd.DataFrame(rows).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print("Interpretations are sample diagnostics, not structural identities.")
    figure, _ = plot_pca_loadings(result)
    try:
        path = save_figure(figure, "outputs/pca_loadings.png")
    finally:
        plt.close(figure)
    print(f"Saved: {_relative_path(path)}")
    return result


def run_relative_value_workflow(
    *, offline: bool, state: MarketState | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank current CMT residuals and construct a currency DV01-neutral fly.

    Residuals are observed minus fitted decimal annual yields and are displayed
    in bp. Hedge notionals are currency face and position DV01 is currency per bp.
    """
    market = load_market_state(offline=offline) if state is None else state
    residuals = pd.Series(
        market.nss_fit.residuals,
        index=market.maturity_labels,
        name=market.valuation_date.isoformat(),
    )
    rankings = rank_curve_points(residuals)
    observed = market.observed_yields_decimal
    fitted = pd.Series(market.nss_fit.fitted_yields, index=market.maturity_labels)
    rankings.insert(2, "valuation_date", market.valuation_date.isoformat())
    rankings.insert(
        4,
        "observed_yield_decimal",
        rankings["tenor"].map(observed).astype(float),
    )
    rankings.insert(
        5,
        "fitted_yield_decimal",
        rankings["tenor"].map(fitted).astype(float),
    )
    rankings["input_representation"] = market.dataset.representation.value
    rankings["residual_sign_convention"] = (
        "observed minus fitted; negative=rich, positive=cheap"
    )
    report_path = _save_csv(rankings, "relative_value_report.csv")

    unit_dv01 = _butterfly_unit_dv01(market)
    butterfly = construct_dv01_neutral_butterfly(
        unit_dv01,
        belly_notional=1_000_000.0,
        belly_side="long",
    )
    fly_yield_bp = butterfly_yield(observed) / BASIS_POINT_DECIMAL
    print("\nCurrent relative value")
    print("----------------------")
    print("Residual convention: observed minus fitted; negative=rich, positive=cheap.")
    display_columns = [
        "rank_rich_to_cheap",
        "tenor",
        "residual_bp",
        "classification",
    ]
    print(
        rankings.loc[:, display_columns].to_string(
            index=False, float_format="{:,.3f}".format
        )
    )
    print(f"\n2Y/5Y/10Y raw yield butterfly (2*5Y - 2Y - 10Y): {fly_yield_bp:.3f} bp")
    print("DV01-neutral butterfly (long 1,000,000 currency face in the 5Y belly)")
    butterfly_columns = [
        "leg",
        "side",
        "notional",
        "unit_dv01_currency_per_bp_per_notional",
        "signed_position_dv01_currency_per_bp",
    ]
    print(
        butterfly.loc[:, butterfly_columns].to_string(
            index=False, float_format="{:,.6f}".format
        )
    )
    aggregate = float(butterfly["aggregate_dv01_currency_per_bp"].iloc[0])
    print(f"Aggregate first-order DV01: {aggregate:,.10f} currency/bp")
    print(f"Saved: {_relative_path(report_path)}")
    return rankings, butterfly


def run_demo_workflow(*, offline: bool) -> None:
    """Run the end-to-end demo using decimal rates and currency prices.

    ``offline=True`` guarantees that bundled data are used without a network
    call. Generated figures and CSV reports are written only while this workflow
    is executed and always below ``outputs/``.
    """
    print("Fixed-Income Yield Curve & Relative-Value Demo")
    print("================================================")
    state = run_curve_workflow(offline=offline)
    pricing_ytm = float(state.observed_yields_decimal["10Y"])
    run_bond_workflow(
        bond=representative_bond(),
        settlement_date=state.valuation_date,
        ytm_decimal=pricing_ytm,
    )
    run_risk_workflow(offline=offline, state=state)
    run_pca_workflow(offline=offline, state=state)
    run_relative_value_workflow(offline=offline, state=state)


def _print_curve_summary(state: MarketState) -> None:
    print("\nLatest observed Treasury CMT curve")
    print("----------------------------------")
    print(f"Observation date: {state.valuation_date.isoformat()}")
    data_mode = "offline bundled sample" if state.dataset.is_offline else "online"
    print(f"Data mode: {data_mode}")
    print("Representation: Treasury constant-maturity yields (not zero rates)")
    observed = pd.DataFrame(
        {
            "tenor": state.maturity_labels,
            "maturity_years": state.maturities_years,
            "yield_decimal": state.observed_yields_decimal.to_numpy(dtype=float),
            "yield_percent": state.observed_yields_decimal.to_numpy(dtype=float)
            * 100.0,
        }
    )
    print(observed.to_string(index=False, float_format="{:,.5f}".format))
    parameters = state.nss_fit.parameters
    parameter_table = pd.DataFrame(
        {
            "parameter": ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"],
            "value": parameters.as_array(),
            "unit": ["decimal", "decimal", "decimal", "decimal", "years", "years"],
        }
    )
    print("\nNelson-Siegel-Svensson parameters")
    print(parameter_table.to_string(index=False, float_format="{:.8f}".format))
    print(f"Fit RMSE: {state.nss_fit.rmse / BASIS_POINT_DECIMAL:.4f} bp")


def _print_bond_summary(analytics: BondAnalytics) -> None:
    bond = analytics.bond
    print("\nRepresentative fixed-rate bond (conventional YTM valuation)")
    print("-----------------------------------------------------------")
    print(
        f"Settlement {analytics.settlement_date.isoformat()} | maturity "
        f"{bond.maturity_date.isoformat()} | coupon {bond.coupon_rate * 100:.3f}% | "
        f"face {bond.face_value:,.2f}"
    )
    rows = [
        ("Clean price", analytics.clean_price_currency, "currency"),
        ("Dirty price", analytics.dirty_price_currency, "currency"),
        ("Accrued interest", analytics.accrued_interest_currency, "currency"),
        ("YTM", analytics.ytm_decimal * 100.0, "% nominal annual"),
        ("Macaulay duration", analytics.macaulay_duration_years, "years"),
        ("Modified duration", analytics.modified_duration_years, "years"),
        ("Convexity", analytics.convexity_years_squared, "years^2"),
        ("DV01", analytics.dv01_currency_per_bp, "currency/bp"),
    ]
    table = pd.DataFrame(rows, columns=["measure", "value", "unit"])
    print(table.to_string(index=False, float_format="{:,.6f}".format))


def _butterfly_unit_dv01(state: MarketState) -> pd.Series:
    values: dict[str, float] = {}
    for tenor, years in (("2Y", 2), ("5Y", 5), ("10Y", 10)):
        ytm = float(state.observed_yields_decimal[tenor])
        bond = FixedRateBond(
            accrual_start_date=state.valuation_date,
            maturity_date=_add_years(state.valuation_date, years),
            coupon_rate=ytm,
            face_value=1.0,
            frequency=2,
        )
        values[tenor] = dv01(bond, state.valuation_date, ytm)
    return pd.Series(values, name="unit_dv01_currency_per_bp_per_notional")


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def _save_csv(frame: pd.DataFrame, filename: str) -> Path:
    OUTPUTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = (OUTPUTS_DIRECTORY / filename).resolve()
    if OUTPUTS_DIRECTORY.resolve() not in path.parents:
        raise ValueError("generated reports must be saved below outputs/")
    frame.to_csv(path, index=False)
    return path


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


__all__ = [
    "BASIS_POINT_DECIMAL",
    "BondAnalytics",
    "MarketState",
    "calculate_bond_analytics",
    "illustrative_zero_rate_proxy",
    "load_market_state",
    "representative_bond",
    "run_bond_workflow",
    "run_curve_workflow",
    "run_demo_workflow",
    "run_pca_workflow",
    "run_relative_value_workflow",
    "run_risk_workflow",
]
