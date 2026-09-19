from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polarix import __version__
from polarix.ingestion.fixtures import SCENARIOS, FixtureConfig
from polarix.orchestration.demo import inspect_run, run_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polarix", description="Auditable financial data pipelines"
    )
    parser.add_argument("--version", action="version", version=f"polarix {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run the offline pipeline using synthetic CME/MT5 fixtures"
    )
    demo.add_argument("--output", type=Path, default=Path(".polarix"))
    demo.add_argument("--run-id")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--seconds", type=int, default=1200)
    demo.add_argument("--scenario", choices=SCENARIOS, default="normal")
    inspect = commands.add_parser(
        "inspect", help="Verify artifact hashes and query feature tables with DuckDB"
    )
    inspect.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run_dir = (
            run_demo(
                args.output, FixtureConfig(args.seed, args.seconds, args.scenario), args.run_id
            )
            if args.command == "demo"
            else args.run_dir
        )
        summary = inspect_run(run_dir)
        print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, allow_nan=False))
        return 0 if summary["status"] == "PASS" else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
