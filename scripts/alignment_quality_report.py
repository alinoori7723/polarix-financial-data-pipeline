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
from polarix.alignment.alignment_quality import (
    DEFAULT_ALIGNMENT_TOLERANCE_MS,
    DEFAULT_DIAGNOSTIC_TOLERANCES_MS,
    DEFAULT_SAMPLE_UNMATCHED_LIMIT,
    DEFAULT_SYMBOL_MAP,
    AlignmentQualityConfig,
    AlignmentQualityThresholds,
    build_alignment_quality_report,
    parse_symbol_map,
    write_reports,
)

DEFAULT_CME_ROOT = Path(".polarix/data/normalized/cme_reference/reference_trades")
DEFAULT_MT5_ROOT = Path(".polarix/data/normalized/mt5_ticks")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")
DEFAULT_CME_RAW_ROOT = Path(".polarix/data/raw/cme_sample")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix alignment quality report")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--cme-root", default=str(DEFAULT_CME_ROOT))
    p.add_argument(
        "--cme-raw-root",
        default=str(DEFAULT_CME_RAW_ROOT),
        help="directory holding the Databento raw sample and its *.metadata.json sidecars (used to detect TRUNCATED_BY_LIMIT).",
    )
    p.add_argument("--mt5-root", default=str(DEFAULT_MT5_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument(
        "--symbol-map",
        default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items())),
        help="Comma-separated CME=MT5 pairs (default ES=SPX500,NQ=NDX100)",
    )
    p.add_argument(
        "--alignment-tolerance-ms",
        type=int,
        default=DEFAULT_ALIGNMENT_TOLERANCE_MS,
        help="Default 50ms (CME-safe). 1000ms is diagnostic only.",
    )
    p.add_argument(
        "--diagnostic-tolerances-ms",
        default=",".join((str(t) for t in DEFAULT_DIAGNOSTIC_TOLERANCES_MS)),
        help="Comma-separated tolerances for the sensitivity table.",
    )
    p.add_argument("--sample-unmatched-limit", type=int, default=DEFAULT_SAMPLE_UNMATCHED_LIMIT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[alignment_quality_report] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    try:
        diagnostic_tolerances = tuple(
            (int(t.strip()) for t in args.diagnostic_tolerances_ms.split(",") if t.strip())
        )
    except ValueError as exc:
        print(f"[alignment_quality_report] bad --diagnostic-tolerances-ms: {exc}", file=sys.stderr)
        return 2
    raw_root_path = Path(args.cme_raw_root) / f"date={args.date}"
    config = AlignmentQualityConfig(
        date=args.date,
        cme_root=Path(args.cme_root),
        mt5_root=Path(args.mt5_root),
        reports_root=Path(args.reports_root),
        symbol_map=symbol_map,
        alignment_tolerance_ms=args.alignment_tolerance_ms,
        diagnostic_tolerances_ms=diagnostic_tolerances,
        sample_unmatched_limit=args.sample_unmatched_limit,
        thresholds=AlignmentQualityThresholds(),
        write_unmatched_sample=not args.dry_run,
        cme_raw_input_root=raw_root_path if raw_root_path.exists() else None,
    )
    try:
        report, sample = build_alignment_quality_report(config)
    except Exception as exc:
        print(f"[alignment_quality_report] internal error: {exc}", file=sys.stderr)
        return 4
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "decision": report["quality_decision"],
                    "decision_reason": report.get("decision_reason"),
                    "real_overlap_present": report["real_overlap_present"],
                    "overall": report["overall"],
                },
                indent=2,
                default=str,
            )
        )
        return 0 if report["quality_decision"] != "FAIL" else 1
    json_path, txt_path, sample_path = write_reports(config, report, sample)
    print(
        json.dumps(
            {
                "decision": report["quality_decision"],
                "decision_reason": report.get("decision_reason"),
                "real_overlap_present": report["real_overlap_present"],
                "json_report": str(json_path),
                "txt_report": str(txt_path),
                "unmatched_sample": str(sample_path) if sample_path else None,
            },
            indent=2,
        )
    )
    return 0 if report["quality_decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
