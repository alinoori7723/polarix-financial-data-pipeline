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
from polarix.normalization.normalization import (
    DEFAULT_JOIN_SAFE_THRESHOLD_MS,
    NormalizationConfig,
    NormalizationError,
    normalize,
)
from polarix.orchestration.run_metadata import RunMetadataError, resolve_run_metadata

DEFAULT_RAW_ROOT = Path(".polarix/data/raw/mt5_ticks")
DEFAULT_SILVER_ROOT = Path(".polarix/data/normalized/mt5_ticks")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix Silver normalization")
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
        help="Reports directory used for metadata resolution (default: %(default)s)",
    )
    p.add_argument(
        "--metadata-path",
        default=None,
        help="Explicit live_run_*_summary.json | logger_manifest.json | logger_health.json",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Pick a specific run. Prefers the run-scoped directory at <reports-root>/logger_runs/<RUN_ID>/; falls back to a legacy live_run_<RUN_ID>_summary.json if the run-scoped layout is absent.",
    )
    p.add_argument(
        "--strict-run-metadata",
        dest="strict_run_metadata",
        action="store_true",
        default=True,
        help="Default: refuse to silently use the global rolling logger_manifest.json when run-scoped metadata exists. immutability guarantee.",
    )
    p.add_argument(
        "--no-strict-run-metadata",
        dest="strict_run_metadata",
        action="store_false",
        help="Loosen strict-run-metadata. Useful only in test scenarios.",
    )
    p.add_argument(
        "--allow-latest-metadata",
        action="store_true",
        default=False,
        help="Explicit opt-in: fall back to the legacy global logger_manifest.json / logger_health.json resolution when no verified run-scoped metadata is available. Disabled by default.",
    )
    p.add_argument(
        "--join-safe-threshold-ms",
        type=int,
        default=DEFAULT_JOIN_SAFE_THRESHOLD_MS,
        help="Default %(default)s (CME-safe). 1000ms is NOT join-safe.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        metadata = resolve_run_metadata(
            reports_root=Path(args.reports_root),
            metadata_path=Path(args.metadata_path) if args.metadata_path else None,
            run_id=args.run_id,
            date=args.date,
            allow_latest_metadata=args.allow_latest_metadata,
            strict_run_metadata=args.strict_run_metadata,
        )
    except RunMetadataError as exc:
        tag = getattr(exc, "error_tag", None) or "RUN_METADATA_ERROR"
        print(f"[normalize_telemetry] FAIL CLOSED ({tag}): {exc}", file=sys.stderr)
        return 2
    run_window_start_utc: str | None = None
    run_window_end_utc: str | None = None
    if args.run_id is not None:
        run_window_start_utc, run_window_end_utc = metadata.run_window_utc()
        if run_window_start_utc is None or run_window_end_utc is None:
            print(
                f"[normalize_telemetry] FAIL CLOSED (RUN_WINDOW_UNAVAILABLE): --run-id {args.run_id} was requested but the run window could not be resolved from run metadata (started_at_utc={metadata.run_started_at_utc!r}, ended_at_utc={metadata.run_ended_at_utc!r}, latest_data_file_mtime_utc={metadata.latest_data_file_mtime_utc!r}). Run-scoped normalization without a window would risk cross-run contamination; refusing to proceed.",
                file=sys.stderr,
            )
            return 2
    config = NormalizationConfig(
        raw_root=Path(args.raw_root),
        silver_root=Path(args.normalized_root),
        date=args.date,
        metadata=metadata,
        join_safe_threshold_ms=args.join_safe_threshold_ms,
        force=args.force,
        dry_run=args.dry_run,
        run_id=args.run_id,
        run_window_start_utc=run_window_start_utc,
        run_window_end_utc=run_window_end_utc,
    )
    if args.dry_run:
        try:
            result = normalize(config)
        except NormalizationError as exc:
            print(f"[normalize_telemetry] FAIL CLOSED: {exc}", file=sys.stderr)
            return 2
        plan_dump = {
            "dry_run": True,
            "date": args.date,
            "raw_root": str(config.raw_root),
            "silver_root": str(config.silver_root),
            "verified_offset_min": metadata.verified_offset_min,
            "join_safe_threshold_ms": args.join_safe_threshold_ms,
            "selected_run_id": config.run_id,
            "selected_metadata_path": str(metadata.source_path),
            "selected_metadata_source_type": metadata.source_type,
            "metadata_source_kind": metadata.source_kind,
            "timestamp_semantics_status": metadata.timestamp_semantics_status,
            "metadata_verified_for_normalization": True,
            "silver_layout": "run_scoped" if config.is_run_scoped else "legacy_date_level",
            "silver_layout_version": config.silver_layout_version,
            "run_start_utc": run_window_start_utc,
            "run_end_utc": run_window_end_utc,
            "symbols": [
                {
                    "symbol": p.symbol,
                    "bronze_file_count": len(p.bronze_files),
                    "silver_output_dir": str(p.silver_output),
                }
                for p in result.symbol_plans
            ],
        }
        print(json.dumps(plan_dump, indent=2))
        return 0
    try:
        result = normalize(config)
    except FileExistsError as exc:
        print(f"[normalize_telemetry] {exc}", file=sys.stderr)
        return 3
    except NormalizationError as exc:
        print(f"[normalize_telemetry] FAIL CLOSED: {exc}", file=sys.stderr)
        return 2
    summary = {
        "ok": True,
        "date": args.date,
        "verified_offset_min": metadata.verified_offset_min,
        "join_safe_threshold_ms": args.join_safe_threshold_ms,
        "selected_run_id": config.run_id,
        "selected_metadata_path": str(metadata.source_path),
        "selected_metadata_source_type": metadata.source_type,
        "metadata_source_kind": metadata.source_kind,
        "timestamp_semantics_status": metadata.timestamp_semantics_status,
        "metadata_verified_for_normalization": True,
        "silver_layout": "run_scoped" if config.is_run_scoped else "legacy_date_level",
        "silver_layout_version": config.silver_layout_version,
        "run_start_utc": run_window_start_utc,
        "run_end_utc": run_window_end_utc,
        "rows_excluded_by_window": result.rows_excluded_by_window,
        "manifest_path": str(result.manifest_path),
        "silver_output_dirs": {p.symbol: str(p.silver_output) for p in result.symbol_plans},
        "symbols": {p.symbol: p.rows_written for p in result.symbol_plans},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
