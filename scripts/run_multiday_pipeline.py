from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.orchestration.databento_cost_guard import (
    COST_ESTIMATE_UNAVAILABLE,
    PHYSICAL_LIMIT_SUPPORTED,
    detect_databento_cost_estimate_capability,
    detect_databento_physical_limit_capability,
)
from polarix.orchestration.day_plan import DEFAULT_SYMBOL_MAP
from polarix.orchestration.pipeline_orchestrator import (
    DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS,
    DEFAULT_MAX_DATABENTO_RETRIES_PER_DAY,
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    DEFAULT_MIN_FREE_DISK_GB,
    DEFAULT_POST_ROLL_MINUTES,
    DEFAULT_PRE_ROLL_MINUTES,
    OrchestratorConfig,
    run_multiday,
)
from polarix.orchestration.run_selection import (
    POLICY_LONGEST_VERIFIED,
    VALID_POLICIES,
    parse_run_id_map,
)

DEFAULT_DATA_ROOT = Path(".polarix/data")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix multi-day orchestrator")
    p.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD list")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument("--symbols", default="ES,NQ")
    p.add_argument(
        "--symbol-map", default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items()))
    )
    p.add_argument("--pre-roll-minutes", type=int, default=DEFAULT_PRE_ROLL_MINUTES)
    p.add_argument("--post-roll-minutes", type=int, default=DEFAULT_POST_ROLL_MINUTES)
    p.add_argument("--min-free-disk-gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB)
    p.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    p.add_argument("--download-cme", action="store_true", help="Plan a Databento download per day.")
    p.add_argument(
        "--allow-databento-download",
        action="store_true",
        help="Explicit guard: required IN ADDITION to --download-cme for the orchestrator to actually call Databento.",
    )
    p.add_argument("--continue-on-missing-mt5", action="store_true")
    p.add_argument(
        "--no-continue-on-missing-cme",
        action="store_true",
        help="Stop the day when Databento data is unavailable.",
    )
    p.add_argument(
        "--no-continue-on-day-failure",
        action="store_true",
        help="Stop the entire run on any day failure.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="pass --force to force-capable stages when their output already exists. Without this flag, trusted existing artifacts are skipped (STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED) rather than triggering an output-exists FAIL.",
    )
    p.add_argument(
        "--max-databento-retries-per-day", type=int, default=DEFAULT_MAX_DATABENTO_RETRIES_PER_DAY
    )
    p.add_argument(
        "--run-selection-policy", default=POLICY_LONGEST_VERIFIED, choices=list(VALID_POLICIES)
    )
    p.add_argument(
        "--run-id-map",
        default=None,
        help='Comma-separated "YYYY-MM-DD=live_run_id" pairs for --run-selection-policy explicit',
    )
    p.add_argument("--max-estimated-cost-usd", type=float, default=10.0)
    p.add_argument("--max-estimated-size-gb", type=float, default=5.0)
    p.add_argument("--max-download-size-gb", type=float, default=5.0)
    p.add_argument("--max-download-records", type=int, default=None)
    p.add_argument("--max-download-cost-usd", type=float, default=10.0)
    p.add_argument("--acknowledge-cost-risk", action="store_true")
    p.add_argument("--allow-unestimated-download", action="store_true")
    p.add_argument(
        "--cost-estimate-required", dest="cost_estimate_required", action="store_true", default=True
    )
    p.add_argument(
        "--no-cost-estimate-required", dest="cost_estimate_required", action="store_false"
    )
    p.add_argument(
        "--require-physical-download-limit",
        dest="require_physical_download_limit",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-require-physical-download-limit",
        dest="require_physical_download_limit",
        action="store_false",
    )
    p.add_argument("--allow-download-without-physical-limit", action="store_true")
    p.add_argument(
        "--allow-truncated-cme-sample",
        action="store_true",
        help="explicitly allow downstream stages to consume a CME raw sample whose download was truncated by --max-download-records. Default is fail-closed: truncated samples block downstream.",
    )
    p.add_argument(
        "--download-preflight",
        action="store_true",
        help="Inspect Databento SDK capabilities (cost/size estimate + physical request limit) WITHOUT downloading. Never calls timeseries.get_range; never makes a paid data request.",
    )
    p.add_argument(
        "--child-terminate-timeout-seconds",
        type=float,
        default=DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS,
    )
    p.add_argument(
        "--kill-children-on-interrupt",
        dest="kill_children_on_interrupt",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-kill-children-on-interrupt",
        dest="kill_children_on_interrupt",
        action="store_false",
        help="Debug-only: do not kill child processes on interrupt; the report will warn loudly.",
    )
    return p.parse_args(argv)


def _parse_symbol_map(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"symbol-map entry {item!r} must contain '='")
        k, v = item.split("=", 1)
        k, v = (k.strip(), v.strip())
        if not k or not v:
            raise ValueError(f"symbol-map entry {item!r} has empty key or value")
        out[k] = v
    if not out:
        raise ValueError("symbol-map must contain at least one entry")
    return out


def _expand_dates(args: argparse.Namespace) -> list[str]:
    if args.dates:
        out = [d.strip() for d in args.dates.split(",") if d.strip()]
        if not out:
            raise ValueError("--dates parsed empty")
        return out
    if args.start_date and args.end_date:
        start = _dt.date.fromisoformat(args.start_date)
        end = _dt.date.fromisoformat(args.end_date)
        if end < start:
            raise ValueError("--end-date is before --start-date")
        out = []
        cur = start
        while cur <= end:
            out.append(cur.isoformat())
            cur = cur + _dt.timedelta(days=1)
        return out
    raise ValueError("must pass either --dates or both --start-date and --end-date")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = _parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[run_multiday_pipeline] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    try:
        dates = _expand_dates(args)
    except ValueError as exc:
        print(f"[run_multiday_pipeline] bad date selection: {exc}", file=sys.stderr)
        return 2
    if args.download_cme and (not args.allow_databento_download):
        print(
            "[run_multiday_pipeline] fail-closed: --download-cme requires --allow-databento-download to be set explicitly. Refusing to run.",
            file=sys.stderr,
        )
        return 2
    if (
        args.download_cme
        and args.allow_databento_download
        and (not args.dry_run)
        and (not args.download_preflight)
        and (not args.acknowledge_cost_risk)
    ):
        print(
            "[run_multiday_pipeline] fail-closed: real Databento download requires --acknowledge-cost-risk. Refusing to run.",
            file=sys.stderr,
        )
        return 2
    try:
        run_id_map = parse_run_id_map(args.run_id_map)
    except ValueError as exc:
        print(f"[run_multiday_pipeline] bad --run-id-map: {exc}", file=sys.stderr)
        return 2
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    config = OrchestratorConfig(
        dates=dates,
        data_root=Path(args.data_root),
        reports_root=Path(args.reports_root),
        repo_root=Path(__file__).resolve().parents[1],
        python_exe=Path(sys.executable),
        symbol_map=symbol_map,
        symbols=symbols,
        pre_roll_minutes=args.pre_roll_minutes,
        post_roll_minutes=args.post_roll_minutes,
        min_free_disk_gb=args.min_free_disk_gb,
        min_available_memory_gb=args.min_available_memory_gb,
        download_cme=args.download_cme,
        allow_databento_download=args.allow_databento_download,
        continue_on_missing_mt5=args.continue_on_missing_mt5,
        continue_on_missing_cme=not args.no_continue_on_missing_cme,
        continue_on_day_failure=not args.no_continue_on_day_failure,
        dry_run=args.dry_run,
        force=args.force,
        max_databento_retries_per_day=args.max_databento_retries_per_day,
        run_selection_policy=args.run_selection_policy,
        run_id_map=run_id_map,
        acknowledge_cost_risk=args.acknowledge_cost_risk,
        allow_unestimated_download=args.allow_unestimated_download,
        cost_estimate_required=args.cost_estimate_required,
        require_physical_download_limit=args.require_physical_download_limit,
        allow_download_without_physical_limit=args.allow_download_without_physical_limit,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        max_estimated_size_gb=args.max_estimated_size_gb,
        max_download_size_gb=args.max_download_size_gb,
        max_download_cost_usd=args.max_download_cost_usd,
        max_download_records=args.max_download_records,
        child_terminate_timeout_seconds=args.child_terminate_timeout_seconds,
        kill_children_on_interrupt=args.kill_children_on_interrupt,
        allow_truncated_cme_sample=args.allow_truncated_cme_sample,
        rebuild_existing=args.rebuild_existing,
    )
    if args.download_preflight:
        return _run_download_preflight(args, config)
    result = run_multiday(config)
    per_day_summary = []
    if args.dry_run and result.dry_run_report is not None:
        for day in result.dry_run_report["per_day"]:
            per_day_summary.append(
                {
                    "date": day["date"],
                    "selected_run_id": day["run_selection"].get("selected_run_id"),
                    "mt5_silver_exists": day["mt5_silver_exists"],
                    "cme_raw_present": day["cme_raw_files_present"],
                    "downstream_artifacts_present": day.get("downstream_artifacts_present"),
                    "lineage_status": day.get("lineage_status"),
                    "recommended_databento_window": day["recommended_databento_window_utc"],
                    "recommended_databento_window_hours": day.get(
                        "recommended_databento_window_hours"
                    ),
                    "download_gate_decision": day["download_gate_decision"]["decision"],
                    "cost_estimate_status": day["cost_estimate"]["status"],
                    "physical_limit_status": day["physical_limit"]["status"],
                    "high_volume_window_warning": day.get("high_volume_window_warning"),
                    "next_action": day.get("next_action"),
                }
            )
    else:
        for d in result.per_day:
            per_day_summary.append(
                {
                    "date": d.date,
                    "final_status": d.final_status,
                    "stage_statuses": list(d.stage_statuses),
                    "warnings": list(d.warnings),
                    "errors": list(d.errors),
                }
            )
    out = {
        "dry_run": args.dry_run,
        "dates_requested": list(config.dates),
        "dates_completed": result.summary["dates_completed"],
        "dates_skipped": result.summary["dates_skipped"],
        "dates_failed": result.summary["dates_failed"],
        "interrupted": result.interrupted,
        "child_cleanup_records": list(result.child_cleanup_records),
        "per_day_summary": per_day_summary,
        "summary_json": str(result.summary_json_path) if result.summary_json_path else None,
        "summary_txt": str(result.summary_txt_path) if result.summary_txt_path else None,
        "per_day_reports": [str(p) for p in result.per_day_json_paths],
        "dry_run_report_json": str(result.dry_run_report_json_path)
        if result.dry_run_report_json_path
        else None,
        "dry_run_report_txt": str(result.dry_run_report_txt_path)
        if result.dry_run_report_txt_path
        else None,
        "pipeline_continued_after_failure": result.summary.get("pipeline_continued_after_failure"),
        "global_guard_stopped": result.summary.get("global_guard_stopped"),
    }
    print(json.dumps(out, indent=2, default=str))
    if result.interrupted:
        return 130
    return 0 if not result.summary["dates_failed"] else 1


def _run_download_preflight(args, config) -> int:
    import os as _os

    api_key_present = bool(_os.environ.get("DATABENTO_API_KEY"))
    cost_cap = detect_databento_cost_estimate_capability(api_key_present=api_key_present)
    phys_cap = detect_databento_physical_limit_capability(
        max_download_records=args.max_download_records,
        max_download_size_gb=args.max_download_size_gb,
        max_download_cost_usd=args.max_download_cost_usd,
    )
    blocked_reasons: list[str] = []
    required_overrides: list[str] = []
    if not args.download_cme:
        blocked_reasons.append("MISSING_FLAG_DOWNLOAD_CME")
        required_overrides.append("--download-cme")
    if not args.allow_databento_download:
        blocked_reasons.append("MISSING_FLAG_ALLOW_DATABENTO_DOWNLOAD")
        required_overrides.append("--allow-databento-download")
    if not args.acknowledge_cost_risk:
        blocked_reasons.append("MISSING_FLAG_ACKNOWLEDGE_COST_RISK")
        required_overrides.append("--acknowledge-cost-risk")
    if not api_key_present:
        blocked_reasons.append("MISSING_API_KEY")
        required_overrides.append("env DATABENTO_API_KEY")
    if cost_cap.status == COST_ESTIMATE_UNAVAILABLE:
        if args.cost_estimate_required and (not args.allow_unestimated_download):
            blocked_reasons.append("COST_ESTIMATE_UNAVAILABLE")
            required_overrides.append("--allow-unestimated-download")
    if phys_cap.status != PHYSICAL_LIMIT_SUPPORTED:
        if args.require_physical_download_limit and (
            not args.allow_download_without_physical_limit
        ):
            blocked_reasons.append("PHYSICAL_LIMIT_UNSUPPORTED")
            required_overrides.append("--allow-download-without-physical-limit")
    real_download_would_be_blocked = bool(blocked_reasons)
    report = {
        "mode": "download_preflight",
        "dry_run": True,
        "dates_requested": list(config.dates),
        "api_key_present": api_key_present,
        "cost_estimate_capability_status": cost_cap.status,
        "cost_estimate_capability_source": cost_cap.source,
        "cost_estimate_capability_message": cost_cap.message,
        "physical_limit_capability_status": phys_cap.status,
        "physical_limit_capability_supported_types": list(phys_cap.supported_limit_types),
        "physical_limit_capability_selected_type": phys_cap.selected_limit_type,
        "physical_limit_capability_selected_value": phys_cap.selected_limit_value,
        "physical_limit_capability_target": phys_cap.sdk_call_target,
        "real_download_would_be_blocked": real_download_would_be_blocked,
        "blocked_reasons": blocked_reasons,
        "required_override_flags": required_overrides,
        "operator_flags": {
            "download_cme": args.download_cme,
            "allow_databento_download": args.allow_databento_download,
            "acknowledge_cost_risk": args.acknowledge_cost_risk,
            "allow_unestimated_download": args.allow_unestimated_download,
            "cost_estimate_required": args.cost_estimate_required,
            "require_physical_download_limit": args.require_physical_download_limit,
            "allow_download_without_physical_limit": args.allow_download_without_physical_limit,
        },
        "this_mode": "preflight only: never calls timeseries.get_range; never downloads. Probes attribute / signature surfaces of the installed Databento SDK.",
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if not real_download_would_be_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
