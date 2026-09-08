"""Command-line entry point for the fixed-income analytics project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from fixed_income.bonds import FixedRateBond
from fixed_income.workflows import (
    run_backtest_workflow,
    run_bond_workflow,
    run_curve_workflow,
    run_demo_workflow,
    run_pca_workflow,
    run_relative_value_workflow,
    run_risk_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; this function has no financial units."""
    parser = argparse.ArgumentParser(
        description="Fixed-Income Yield Curve & Relative-Value Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the end-to-end demonstration")
    _add_offline_argument(demo_parser)

    curve_parser = subparsers.add_parser("curve", help="Fit and plot the latest curve")
    _add_offline_argument(curve_parser)

    bond_parser = subparsers.add_parser("bond", help="Run fixed-rate bond analytics")
    bond_parser.add_argument(
        "--settlement",
        type=_iso_date,
        default=date(2024, 6, 28),
        help="settlement date in YYYY-MM-DD form (default: 2024-06-28)",
    )
    bond_parser.add_argument(
        "--accrual-start",
        type=_iso_date,
        default=date(2020, 3, 31),
        help="regular schedule start in YYYY-MM-DD form (default: 2020-03-31)",
    )
    bond_parser.add_argument(
        "--maturity",
        type=_iso_date,
        default=date(2034, 3, 31),
        help="maturity in YYYY-MM-DD form (default: 2034-03-31)",
    )
    bond_parser.add_argument(
        "--coupon-rate",
        type=float,
        default=0.0425,
        help="decimal annual coupon rate; use 0.0425 for 4.25%%",
    )
    bond_parser.add_argument(
        "--ytm",
        type=float,
        default=0.0436,
        help="decimal nominal annual YTM; use 0.0436 for 4.36%%",
    )
    bond_parser.add_argument(
        "--face-value",
        type=float,
        default=100.0,
        help="positive face value in currency units (default: 100)",
    )
    bond_parser.add_argument(
        "--frequency",
        type=int,
        choices=(1, 2),
        default=2,
        help="coupon payments and YTM compounding periods per year",
    )

    risk_parser = subparsers.add_parser("risk", help="Run curve shocks and key rates")
    _add_offline_argument(risk_parser)

    pca_parser = subparsers.add_parser("pca", help="Run historical yield-change PCA")
    _add_offline_argument(pca_parser)

    relative_value_parser = subparsers.add_parser(
        "relative-value", help="Rank current residuals and build the 2s5s10s fly"
    )
    _add_offline_argument(relative_value_parser)
    backtest_parser = subparsers.add_parser(
        "backtest", help="Run the historical 5Y NSS-residual research backtest"
    )
    _add_offline_argument(backtest_parser)
    backtest_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="preceding observations used for signal estimates (default: 60)",
    )
    backtest_parser.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help="minimum preceding observations (default: full lookback)",
    )
    backtest_parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="positive absolute z-score entry threshold (default: 2.0)",
    )
    backtest_parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="non-negative absolute z-score exit threshold (default: 0.5)",
    )
    backtest_parser.add_argument(
        "--transaction-cost-bps-per-face",
        type=float,
        default=0.01,
        help="one-way cost in bp of traded face notional (default: 0.01)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and return a process status code; no financial units."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            run_demo_workflow(offline=args.offline)
        elif args.command == "curve":
            run_curve_workflow(offline=args.offline)
        elif args.command == "bond":
            bond = FixedRateBond(
                accrual_start_date=args.accrual_start,
                maturity_date=args.maturity,
                coupon_rate=args.coupon_rate,
                face_value=args.face_value,
                frequency=args.frequency,
            )
            run_bond_workflow(
                bond=bond,
                settlement_date=args.settlement,
                ytm_decimal=args.ytm,
            )
        elif args.command == "risk":
            run_risk_workflow(offline=args.offline)
        elif args.command == "pca":
            run_pca_workflow(offline=args.offline)
        elif args.command == "backtest":
            run_backtest_workflow(
                offline=args.offline,
                lookback=args.lookback,
                min_observations=args.min_observations,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                transaction_cost_bps_per_face=(
                    args.transaction_cost_bps_per_face
                ),
            )
        else:
            run_relative_value_workflow(offline=args.offline)
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


def _add_offline_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use bundled sample data and make no network request",
    )


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO date {value!r}; expected YYYY-MM-DD"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
