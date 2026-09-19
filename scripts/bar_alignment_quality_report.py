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
from polarix.alignment.bar_alignment_quality import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_MIN_JOIN_SAFE_TICK_RATIO,
    DEFAULT_SYMBOL_MAP,
    BarAlignmentConfig,
    BarAlignmentThresholds,
    build_bar_alignment_quality_report,
    parse_symbol_map,
    write_reports,
)
from polarix.features.bar_aggregation import parse_bucket_sizes

DEFAULT_CME_ROOT = Path(".polarix/data/normalized/cme_reference/reference_trades")
DEFAULT_MT5_ROOT = Path(".polarix/data/normalized/mt5_ticks")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix bar-alignment quality report")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--cme-root", default=str(DEFAULT_CME_ROOT))
    p.add_argument("--mt5-root", default=str(DEFAULT_MT5_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument(
        "--symbol-map", default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items()))
    )
    p.add_argument("--bucket-sizes", default=",".join(DEFAULT_BUCKET_SIZES))
    p.add_argument(
        "--min-join-safe-tick-ratio", type=float, default=DEFAULT_MIN_JOIN_SAFE_TICK_RATIO
    )
    p.add_argument(
        "--max-spread-price-threshold",
        type=float,
        default=None,
        help="Optional; if set, bars with mt5_spread_price_max above this are rejected.",
    )
    p.add_argument("--write-buckets", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[bar_alignment_quality_report] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    try:
        bucket_sizes = parse_bucket_sizes(args.bucket_sizes)
    except ValueError as exc:
        print(f"[bar_alignment_quality_report] bad --bucket-sizes: {exc}", file=sys.stderr)
        return 2
    config = BarAlignmentConfig(
        date=args.date,
        cme_root=Path(args.cme_root),
        mt5_root=Path(args.mt5_root),
        reports_root=Path(args.reports_root),
        symbol_map=symbol_map,
        bucket_sizes=bucket_sizes,
        thresholds=BarAlignmentThresholds(
            min_join_safe_tick_ratio=args.min_join_safe_tick_ratio,
            max_spread_price_threshold=args.max_spread_price_threshold,
        ),
        write_buckets=args.write_buckets and (not args.dry_run),
    )
    try:
        result = build_bar_alignment_quality_report(config)
    except Exception as exc:
        print(f"[bar_alignment_quality_report] internal error: {exc}", file=sys.stderr)
        return 4
    if args.dry_run:
        summary = {
            "dry_run": True,
            "decision": result.report["quality_decision"],
            "decision_reason": result.report.get("decision_reason"),
            "real_overlap_present": result.report["real_overlap_present"],
            "overall": result.report["overall"],
            "per_symbol_status": {
                s: e.get("status") for s, e in result.report["per_symbol"].items()
            },
        }
        print(json.dumps(summary, indent=2, default=str))
        return 0 if result.report["quality_decision"] != "FAIL" else 1
    json_path, txt_path, buckets_path = write_reports(config, result)
    summary = {
        "decision": result.report["quality_decision"],
        "decision_reason": result.report.get("decision_reason"),
        "real_overlap_present": result.report["real_overlap_present"],
        "json_report": str(json_path),
        "txt_report": str(txt_path),
        "buckets_parquet": str(buckets_path) if buckets_path else None,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if result.report["quality_decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
