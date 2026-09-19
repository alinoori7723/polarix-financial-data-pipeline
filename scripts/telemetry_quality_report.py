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
from polarix.normalization.normalization import DEFAULT_JOIN_SAFE_THRESHOLD_MS
from polarix.orchestration.run_metadata import RunMetadataError, resolve_run_metadata
from polarix.quality.telemetry_quality import QualityConfig, build_quality_report, write_reports

DEFAULT_RAW_ROOT = Path(".polarix/data/raw/mt5_ticks")
DEFAULT_SILVER_ROOT = Path(".polarix/data/normalized/mt5_ticks")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix Silver telemetry quality report")
    p.add_argument("--date", required=True, help="YYYY-MM-DD source date")
    p.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Bronze partition root (default: %(default)s)",
    )
    p.add_argument(
        "--normalized-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Silver partition root (default: %(default)s)",
    )
    p.add_argument(
        "--reports-root",
        default=str(DEFAULT_REPORTS_ROOT),
        help="Reports directory (default: %(default)s)",
    )
    p.add_argument("--metadata-path", default=None, help="Explicit metadata file")
    p.add_argument(
        "--run-id",
        default=None,
        help="Pick a specific run. With --run-id the report reads ONLY the run-scoped Silver partition symbol=<S>/date=<DATE>/run_id=<RUN_ID>/. Without --run-id the report fails closed (decision=FAIL) when multiple run partitions exist for the date.",
    )
    p.add_argument(
        "--join-safe-threshold-ms",
        type=int,
        default=DEFAULT_JOIN_SAFE_THRESHOLD_MS,
        help="Default %(default)s (CME-safe)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metadata = resolve_run_metadata(
            reports_root=Path(args.reports_root),
            metadata_path=Path(args.metadata_path) if args.metadata_path else None,
            run_id=args.run_id,
        )
    except RunMetadataError as exc:
        print(f"[telemetry_quality_report] FAIL CLOSED: {exc}", file=sys.stderr)
        Path(args.reports_root).mkdir(parents=True, exist_ok=True)
        fail = {
            "source_date": args.date,
            "decision": "FAIL",
            "fatal_warnings": [f"metadata resolution failed: {exc}"],
            "warnings": [],
            "per_symbol": {},
        }
        (Path(args.reports_root) / f"telemetry_quality_{args.date}.json").write_text(
            json.dumps(fail, indent=2, sort_keys=True), encoding="utf-8"
        )
        (Path(args.reports_root) / f"telemetry_quality_{args.date}.txt").write_text(
            f"Decision: FAIL\nReason: {exc}\n", encoding="utf-8"
        )
        return 2
    config = QualityConfig(
        date=args.date,
        raw_root=Path(args.raw_root),
        silver_root=Path(args.normalized_root),
        reports_root=Path(args.reports_root),
        metadata=metadata,
        join_safe_threshold_ms=args.join_safe_threshold_ms,
        run_id=args.run_id,
    )
    report = build_quality_report(config)
    json_path, txt_path = write_reports(config, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "requested_run_id": report.get("requested_run_id"),
                "selected_run_id": report.get("selected_run_id"),
                "silver_layout": report.get("silver_layout"),
                "silver_layout_version": report.get("silver_layout_version"),
                "json_report": str(json_path),
                "txt_report": str(txt_path),
            },
            indent=2,
        )
    )
    return 0 if report["decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
