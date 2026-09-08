"""Command-line entry point for the fixed-income analytics project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

COMMANDS = (
    "curve",
    "bond",
    "risk",
    "pca",
    "relative-value",
    "backtest",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; this function has no financial units."""
    parser = argparse.ArgumentParser(
        description="Fixed-Income Yield Curve & Relative-Value Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo", help="Run the eventual end-to-end demonstration"
    )
    demo_parser.add_argument(
        "--offline",
        action="store_true",
        help="Use bundled sample data without network access",
    )

    for command in COMMANDS:
        subparsers.add_parser(command, help=f"Run the eventual {command} workflow")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and return a process status code; no financial units."""
    args = build_parser().parse_args(argv)
    suffix = " in offline mode" if args.command == "demo" and args.offline else ""
    print(f"The '{args.command}' workflow{suffix} is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
