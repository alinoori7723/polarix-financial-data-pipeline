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
from polarix.features.bar_aggregation import parse_bucket_sizes
from polarix.features.bar_feature_builder import (
    DEFAULT_BUCKET_SIZES,
    DEFAULT_MIN_JOIN_SAFE_TICK_RATIO,
    DEFAULT_MIN_VOLUME_ALIGNMENT_RATIO,
    DEFAULT_SYMBOL_MAP,
    BuilderConfig,
    build,
    parse_symbol_map,
)

DEFAULT_CME_ROOT = Path(".polarix/data/normalized/cme_reference/reference_trades")
DEFAULT_MT5_ROOT = Path(".polarix/data/normalized/mt5_ticks")
DEFAULT_OUTPUT_ROOT = Path(".polarix/data/features/bar_features")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix Gold candidate feature builder")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--cme-root", default=str(DEFAULT_CME_ROOT))
    p.add_argument("--mt5-root", default=str(DEFAULT_MT5_ROOT))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument(
        "--symbol-map", default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items()))
    )
    p.add_argument("--bucket-sizes", default=",".join(DEFAULT_BUCKET_SIZES))
    p.add_argument("--include-diagnostic-5s", action="store_true")
    p.add_argument(
        "--min-join-safe-tick-ratio", type=float, default=DEFAULT_MIN_JOIN_SAFE_TICK_RATIO
    )
    p.add_argument(
        "--min-volume-alignment-ratio", type=float, default=DEFAULT_MIN_VOLUME_ALIGNMENT_RATIO
    )
    p.add_argument("--max-spread-price-threshold", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[build_bar_features] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    try:
        bucket_sizes = parse_bucket_sizes(args.bucket_sizes)
    except ValueError as exc:
        print(f"[build_bar_features] bad --bucket-sizes: {exc}", file=sys.stderr)
        return 2
    cfg = BuilderConfig(
        date=args.date,
        cme_root=Path(args.cme_root),
        mt5_root=Path(args.mt5_root),
        output_root=Path(args.output_root),
        reports_root=Path(args.reports_root),
        symbol_map=symbol_map,
        bucket_sizes=bucket_sizes,
        include_diagnostic_5s=args.include_diagnostic_5s,
        min_join_safe_tick_ratio=args.min_join_safe_tick_ratio,
        min_volume_alignment_ratio=args.min_volume_alignment_ratio,
        max_spread_price_threshold=args.max_spread_price_threshold,
        dry_run=args.dry_run,
        force=args.force,
    )
    try:
        result = build(cfg)
    except FileExistsError as exc:
        print(f"[build_bar_features] {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[build_bar_features] internal error: {exc}", file=sys.stderr)
        return 4
    if result.missing_input:
        print(
            json.dumps(
                {
                    "status": "MISSING_INPUT_DATA",
                    "cme_root": str(cfg.cme_root),
                    "mt5_root": str(cfg.mt5_root),
                    "message": "No CME reference_trades and/or MT5 Silver for this date. does not invent data.",
                },
                indent=2,
            )
        )
        return 2
    summary = {
        "status": "DRY_RUN" if args.dry_run else "OK",
        "date": args.date,
        "rows_by_pair_bucket": result.rows_by_pair_bucket,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "written_paths": [str(p) for p in result.written_paths],
        "include_diagnostic_5s": args.include_diagnostic_5s,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
