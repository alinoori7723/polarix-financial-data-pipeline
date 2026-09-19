from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.ingestion.cme_downloader import (
    DOWNLOAD_BLOCKED_MISSING_API_KEY,
    DOWNLOAD_BLOCKED_NO_RECORD_LIMIT,
    DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED,
    DOWNLOAD_FAILED,
    DOWNLOAD_OK_TRUNCATED,
    DownloadRequest,
    run_download,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Databento GLBX.MDP3 mbp-1 sample (safe)."
    )
    parser.add_argument("--dataset", default="GLBX.MDP3")
    parser.add_argument("--schema", default="mbp-1")
    parser.add_argument("--symbols", default="ES.c.0,NQ.c.0")
    parser.add_argument("--stype-in", default="continuous")
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2026-05-18T05:30:00Z")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2026-05-18T05:35:00Z")
    parser.add_argument(
        "--output-root",
        default=".polarix/data\\raw\\cme_sample",
        help="Root directory for output. Default: %(default)s",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. Defaults to a deterministic name.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD. When combined with --date-partition-output, the Parquet is written under <output-root>/date=<DATE>/.",
    )
    parser.add_argument(
        "--date-partition-output",
        action="store_true",
        help="Write the Parquet under <output-root>/date=<DATE>/ where DATE is --date (or the calendar date of --start in UTC if --date is absent).",
    )
    parser.add_argument(
        "--max-download-records",
        type=int,
        default=None,
        help="REQUIRED for a real download. Physically mapped to the Databento SDK ``limit`` parameter on timeseries.get_range. If absent, the download is blocked unless --allow-download-without-physical-limit.",
    )
    parser.add_argument(
        "--allow-download-without-physical-limit",
        action="store_true",
        help="Operator override: allow a real download even though no record limit is configured (or the SDK does not accept one). A CRITICAL warning is recorded in the metadata JSON.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Plan the call shape and print planned metadata; do NOT contact Databento.",
    )
    return parser.parse_args(argv)


def _safe_iso_component(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "Z")


def _date_from_start(value: str) -> str:
    m = re.match("^(\\d{4})-(\\d{2})-(\\d{2})", value)
    if not m:
        try:
            t = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            return t.astimezone(_dt.timezone.utc).date().isoformat()
        except Exception:
            return ""
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _resolve_output_path(args: argparse.Namespace, symbols: list[str]) -> Path:
    safe_start = _safe_iso_component(args.start)
    safe_end = _safe_iso_component(args.end)
    sym_part = "_".join((s.replace(".", "") for s in symbols))
    default_name = f"databento_{args.dataset.replace('.', '_')}_{args.schema}_{sym_part}_{safe_start}_{safe_end}.parquet"
    name = args.output_name or default_name
    root = Path(args.output_root)
    if args.date_partition_output:
        date = args.date or _date_from_start(args.start)
        root = root / f"date={date}"
    return root / name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: --symbols is empty.", file=sys.stderr)
        return 2
    output_path = _resolve_output_path(args, symbols)
    request = DownloadRequest(
        dataset=args.dataset,
        schema=args.schema,
        symbols=tuple(symbols),
        stype_in=args.stype_in,
        start_utc=args.start,
        end_utc=args.end,
        output_path=output_path,
        max_download_records=args.max_download_records,
        allow_download_without_physical_limit=args.allow_download_without_physical_limit,
        metadata_only=args.metadata_only,
        dry_run=False,
        api_key_present=bool(os.environ.get("DATABENTO_API_KEY")),
    )
    result = run_download(request)
    summary = {
        "status": result.status,
        "data_completeness_status": result.data_completeness_status,
        "truncation_warning": result.truncation_warning,
        "records_downloaded": result.records_downloaded,
        "output_path": str(result.output_path) if result.output_path else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "file_size_bytes": result.file_size_bytes,
        "physical_limit_applied": result.physical_limit_applied,
        "physical_limit_type": result.physical_limit_type,
        "physical_limit_value": result.physical_limit_value,
        "critical_warnings": list(result.critical_warnings),
        "warnings": list(result.warnings),
        "error": result.error,
        "planned_output_path": str(output_path),
        "request": request.to_dict(),
    }
    print(json.dumps(summary, indent=2, default=str))
    if result.status in (
        DOWNLOAD_BLOCKED_NO_RECORD_LIMIT,
        DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED,
        DOWNLOAD_BLOCKED_MISSING_API_KEY,
    ):
        return 2
    if result.status == DOWNLOAD_FAILED:
        return 1
    if result.status == DOWNLOAD_OK_TRUNCATED:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
