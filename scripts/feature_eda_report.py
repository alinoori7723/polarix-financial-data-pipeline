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
from polarix.features.feature_correlation import DEFAULT_PVALUE_ALPHA, DEFAULT_SMALL_SAMPLE_MIN_ROWS
from polarix.features.feature_eda import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_SYMBOL_PAIRS,
    FeatureEDAConfig,
    FeatureEDAError,
    parse_bucket_size_labels,
    parse_symbol_pairs,
    run_eda,
    write_reports,
)

DEFAULT_FEATURES_ROOT = Path(".polarix/data/features/bar_features")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix feature EDA")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument("--bucket-sizes", default=",".join(DEFAULT_BUCKET_SIZES))
    p.add_argument("--include-diagnostic-5s", action="store_true")
    p.add_argument("--symbol-pairs", default=",".join(DEFAULT_SYMBOL_PAIRS))
    p.add_argument("--max-correlation-features", type=int, default=None)
    p.add_argument("--correlation-method", default="pearson")
    p.add_argument("--small-sample-min-rows", type=int, default=DEFAULT_SMALL_SAMPLE_MIN_ROWS)
    p.add_argument("--pvalue-alpha", type=float, default=DEFAULT_PVALUE_ALPHA)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_pairs = parse_symbol_pairs(args.symbol_pairs)
        bucket_sizes = parse_bucket_size_labels(args.bucket_sizes)
    except ValueError as exc:
        print(f"[feature_eda_report] bad CLI argument: {exc}", file=sys.stderr)
        return 2
    try:
        config = FeatureEDAConfig(
            date=args.date,
            features_root=Path(args.features_root),
            reports_root=Path(args.reports_root),
            symbol_pairs=symbol_pairs,
            bucket_sizes=bucket_sizes,
            include_diagnostic_5s=args.include_diagnostic_5s,
            max_correlation_features=args.max_correlation_features,
            correlation_method=args.correlation_method,
            small_sample_min_rows=args.small_sample_min_rows,
            pvalue_alpha=args.pvalue_alpha,
            force=args.force,
            dry_run=args.dry_run,
        )
    except FeatureEDAError as exc:
        print(f"[feature_eda_report] {exc}", file=sys.stderr)
        return 2
    try:
        result = run_eda(config)
    except FeatureEDAError as exc:
        print(f"[feature_eda_report] {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[feature_eda_report] internal error: {exc}", file=sys.stderr)
        return 4
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "decision": result.report["quality_decision"],
                    "decision_reason": result.report.get("decision_reason"),
                    "total_eligible_rows": result.report.get("total_eligible_rows"),
                    "small_sample_flag": result.report.get("small_sample_flag"),
                    "scipy_available": result.report.get("scipy_available"),
                },
                indent=2,
                default=str,
            )
        )
        return 0 if result.report["quality_decision"] != "FAIL" else 1
    try:
        json_path, txt_path = write_reports(config, result)
    except FeatureEDAError as exc:
        print(f"[feature_eda_report] {exc}", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "decision": result.report["quality_decision"],
                "decision_reason": result.report.get("decision_reason"),
                "json_report": str(json_path),
                "txt_report": str(txt_path),
                **result.paths,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if result.report["quality_decision"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
