from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.ingestion.cme_reference_ingest import DEFAULT_SYMBOLS
from polarix.quality.cme_reference_quality import (
    CmeQualityConfig,
    build_quality_report,
    write_reports,
)

DEFAULT_INPUT_ROOT = Path(".polarix/data/raw/cme_sample")
DEFAULT_OUTPUT_ROOT = Path(".polarix/data/normalized/cme_reference")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix CME reference quality report")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated canonical tickers (default ES,NQ)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    symbols = tuple((s.strip() for s in args.symbols.split(",") if s.strip()))
    config = CmeQualityConfig(
        date=args.date,
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        reports_root=Path(args.reports_root),
        symbols_requested=symbols,
    )
    report = build_quality_report(config)
    json_path, txt_path = write_reports(config, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "json_report": str(json_path),
                "txt_report": str(txt_path),
            },
            indent=2,
        )
    )
    return 0 if report["decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
