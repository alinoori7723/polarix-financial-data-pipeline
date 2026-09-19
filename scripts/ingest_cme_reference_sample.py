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
from polarix.ingestion.cme_databento_schema import SCHEMA
from polarix.ingestion.cme_reference_ingest import (
    CROSS_DATE_CME_SOURCE_REJECTED,
    DEFAULT_SYMBOLS,
    LEGACY_ROOT_FILES_INCLUDED,
    MISSING_CME_RAW_FOR_DATE,
    MISSING_SAMPLE_DATA,
    CmeIngestError,
    IngestConfig,
    ingest,
)

DEFAULT_INPUT_ROOT = Path(".polarix/data/raw/cme_sample")
DEFAULT_OUTPUT_ROOT = Path(".polarix/data/normalized/cme_reference")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix CME reference ingest")
    p.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument("--date", default=None, help="YYYY-MM-DD; required for write")
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated canonical tickers (default ES,NQ)",
    )
    p.add_argument("--schema", default=SCHEMA, help=f"Schema (only {SCHEMA!r} supported)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--include-legacy-root-files",
        action="store_true",
        help="also include <input_root>/*.parquet (legacy pre-2G.4 layout). Default is fail-closed: only <input_root>/date=<DATE>/*.parquet is read. Triggers a LEGACY_ROOT_FILES_INCLUDED warning when used.",
    )
    return p.parse_args(argv)


def _missing_report(args: argparse.Namespace, *, missing_reason: str) -> dict:
    if missing_reason == MISSING_CME_RAW_FOR_DATE:
        message = f"no CME raw Parquet found under {Path(args.input_root).resolve()}/date={args.date}/; pass --include-legacy-root-files to also accept the legacy <input_root>/*.parquet layout (with a warning), or move the file into the date= partition."
    else:
        message = "no Databento GLBX.MDP3 mbp-1 sample found under input_root; does not invent data. Place a small Databento Parquet in the input_root and re-run."
    return {
        "status": missing_reason,
        "input_root": str(Path(args.input_root).resolve()),
        "date": args.date,
        "message": message,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    symbols = tuple((s.strip() for s in args.symbols.split(",") if s.strip()))
    config = IngestConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        reports_root=Path(args.reports_root),
        date=args.date,
        symbols=symbols,
        schema=args.schema,
        force=args.force,
        dry_run=args.dry_run,
        include_legacy_root_files=args.include_legacy_root_files,
    )
    try:
        result = ingest(config)
    except FileExistsError as exc:
        print(f"[ingest_cme_reference_sample] {exc}", file=sys.stderr)
        return 3
    except CmeIngestError as exc:
        msg = str(exc)
        if CROSS_DATE_CME_SOURCE_REJECTED in msg:
            print(
                json.dumps(
                    {
                        "status": CROSS_DATE_CME_SOURCE_REJECTED,
                        "input_root": str(Path(args.input_root).resolve()),
                        "date": args.date,
                        "message": msg,
                    },
                    indent=2,
                )
            )
            return 5
        print(f"[ingest_cme_reference_sample] {exc}", file=sys.stderr)
        return 1
    if result.missing_sample:
        report = _missing_report(args, missing_reason=result.missing_reason or MISSING_SAMPLE_DATA)
        print(json.dumps(report, indent=2))
        return 2
    if result.schema_error is not None:
        print(
            json.dumps(
                {
                    "status": "SCHEMA_REJECTED",
                    "source_files": [str(p) for p in result.source_files],
                    "schema_error": result.schema_error,
                },
                indent=2,
            )
        )
        return 4
    warnings = []
    if result.legacy_files_included:
        warnings.append(LEGACY_ROOT_FILES_INCLUDED)
    summary = {
        "status": "OK" if not args.dry_run else "DRY_RUN",
        "date": config.date,
        "input_root": str(config.input_root),
        "include_legacy_root_files": bool(config.include_legacy_root_files),
        "symbols_requested": list(symbols),
        "source_files": [str(p) for p in result.source_files],
        "legacy_files_included": [str(p) for p in result.legacy_files_included],
        "rejected_cross_date_files": [str(p) for p in result.rejected_cross_date_files],
        "warnings": warnings,
        "row_counts_by_symbol": result.row_counts_by_symbol,
        "output_files": {
            s: [str(p) for p in fs] for s, fs in result.output_files_by_symbol.items()
        },
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "rejected_rows_count": result.rejected_rows_count,
        "unknown_side_count": result.unknown_side_count,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
