from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from polarix.ingestion import cme_downloader as cme_dl
from polarix.orchestration.databento_cost_guard import (
    DOWNLOAD_APPROVED,
    DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT,
    DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE,
    DOWNLOAD_BLOCKED_BY_COST_GUARD,
    DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD,
    DOWNLOAD_BLOCKED_DRY_RUN,
    DOWNLOAD_BLOCKED_MISSING_ACK,
    DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO,
    DOWNLOAD_BLOCKED_MISSING_API_KEY,
    NOT_REQUESTED_DRY_RUN,
    PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
    CostEstimate,
    GateInputs,
    PhysicalLimitCapability,
    decide_download_gate,
)
from polarix.orchestration.day_plan import DEFAULT_SYMBOL_MAP, build_day_plan
from polarix.orchestration.resource_guards import snapshot
from polarix.orchestration.run_selection import (
    POLICY_LONGEST_VERIFIED,
    RunSelection,
    discover_run_candidates,
    select_run_for_date,
)

LINEAGE_CLEAN_READY = "CLEAN_READY"
LINEAGE_MISSING_CME_RAW = "MISSING_CME_RAW"
LINEAGE_DOWNSTREAM_WITHOUT_RAW_CME = "DOWNSTREAM_WITHOUT_RAW_CME"
LINEAGE_LEGACY_ARTIFACTS_PRESENT = "LEGACY_ARTIFACTS_PRESENT"
LINEAGE_REBUILD_REQUIRED_AFTER_CME_DOWNLOAD = "REBUILD_REQUIRED_AFTER_CME_DOWNLOAD"
LINEAGE_BLOCKED_MISSING_INPUT = "BLOCKED_MISSING_INPUT"
LINEAGE_UNTRUSTED_WARNING = "Downstream artifacts exist but source CME raw is missing in the expected date-partitioned layout; lineage is untrusted until CME raw is downloaded and downstream stages are rebuilt."
WARNING_HIGH_DATA_VOLUME_WINDOW = "HIGH_DATA_VOLUME_WINDOW"
DEFAULT_HIGH_VOLUME_WINDOW_HOURS = 6
STAGE_MARK_RUN = "RUN"
STAGE_MARK_SKIP = "SKIP"
STAGE_MARK_PLAN = "PLAN"
STAGE_MARK_BLOCKED = "BLOCKED"
STAGE_MARK_RUN_AFTER_DOWNLOAD = "RUN_AFTER_DOWNLOAD"
DOWNLOAD_NOT_REQUESTED = "DOWNLOAD_NOT_REQUESTED"
DRY_RUN_DOWNLOAD_BLOCKED = "DRY_RUN_DOWNLOAD_BLOCKED"
DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL = "DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL"
BLOCKED_MISSING_CME_RAW = "BLOCKED_MISSING_CME_RAW"


@dataclass
class DryRunConfig:
    dates: list[str]
    data_root: Path
    reports_root: Path
    repo_root: Path
    symbol_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))
    pre_roll_minutes: int = 5
    post_roll_minutes: int = 5
    run_selection_policy: str = POLICY_LONGEST_VERIFIED
    run_id_map: Mapping[str, str] = field(default_factory=dict)
    download_cme: bool = False
    allow_databento_download: bool = False
    acknowledge_cost_risk: bool = False
    allow_unestimated_download: bool = False
    cost_estimate_required: bool = True
    require_physical_download_limit: bool = True
    allow_download_without_physical_limit: bool = False
    max_estimated_cost_usd: Optional[float] = 10.0
    max_estimated_size_gb: Optional[float] = 5.0
    max_download_size_gb: Optional[float] = 5.0
    max_download_cost_usd: Optional[float] = 10.0
    max_download_records: Optional[int] = None
    api_key_present: bool = False
    high_volume_window_hours: float = DEFAULT_HIGH_VOLUME_WINDOW_HOURS

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        self.repo_root = Path(self.repo_root).resolve()


def _scripts_path(repo_root: Path, name: str) -> str:
    return str(repo_root / "scripts" / name)


def _stage_argv(repo_root: Path, python_exe: str, script: str, args: list[str]) -> list[str]:
    return [python_exe, "-u", _scripts_path(repo_root, script), *args]


def _plan_stages_for_date(
    config: DryRunConfig,
    date: str,
    python_exe: str,
    run_id: Optional[str],
    has_cme_raw: bool,
    has_cme_norm: bool,
    has_align_report: bool,
    has_bar_align: bool,
    has_gold: bool,
    has_eda: bool,
    gate_decision: Optional[str],
) -> list[dict]:
    plan: list[dict] = []

    def stage(
        name: str, script: Optional[str], args: list[str], mark: str, reason: Optional[str]
    ) -> None:
        argv = _stage_argv(config.repo_root, python_exe, script, args) if script is not None else []
        plan.append(
            {
                "name": name,
                "argv": argv,
                "mark": mark,
                "would_run": mark
                in (STAGE_MARK_RUN, STAGE_MARK_RUN_AFTER_DOWNLOAD, STAGE_MARK_PLAN),
                "skip_reason": reason if mark in (STAGE_MARK_SKIP, STAGE_MARK_BLOCKED) else None,
                "reason": reason,
            }
        )

    if has_cme_raw:
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_SKIP, "STAGE_SKIPPED_ALREADY_EXISTS")
    elif not config.download_cme:
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_SKIP, DOWNLOAD_NOT_REQUESTED)
    elif gate_decision == DOWNLOAD_BLOCKED_DRY_RUN:
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_PLAN, DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL)
    elif gate_decision in (DOWNLOAD_APPROVED, DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT):
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_PLAN, DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL)
    elif gate_decision in (
        DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE,
        DOWNLOAD_BLOCKED_BY_COST_GUARD,
        DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD,
        DOWNLOAD_BLOCKED_MISSING_ACK,
        DOWNLOAD_BLOCKED_MISSING_ALLOW_DATABENTO,
        DOWNLOAD_BLOCKED_MISSING_API_KEY,
    ):
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_BLOCKED, gate_decision)
    else:
        stage("CME_DOWNLOAD", None, [], STAGE_MARK_BLOCKED, gate_decision or "BLOCKED_UNKNOWN")
    not has_cme_raw and (
        not config.download_cme
        or any((plan[0].get("mark") == STAGE_MARK_BLOCKED for plan in [plan[:1]]))
    )

    def downstream_mark_and_reason(already_exists: bool):
        if has_cme_raw:
            if already_exists:
                return (STAGE_MARK_SKIP, "STAGE_SKIPPED_ALREADY_EXISTS")
            return (STAGE_MARK_RUN, None)
        if not config.download_cme:
            return (STAGE_MARK_BLOCKED, BLOCKED_MISSING_CME_RAW)
        if gate_decision in (
            DOWNLOAD_APPROVED,
            DOWNLOAD_APPROVED_WITHOUT_PHYSICAL_LIMIT,
            DOWNLOAD_BLOCKED_DRY_RUN,
        ):
            return (STAGE_MARK_RUN_AFTER_DOWNLOAD, None)
        return (STAGE_MARK_BLOCKED, BLOCKED_MISSING_CME_RAW)

    m, r = downstream_mark_and_reason(has_cme_norm)
    stage("CME_INGEST", "ingest_cme_reference_sample.py", ["--date", date], m, r)
    m, r = downstream_mark_and_reason(False)
    stage("CME_QUALITY", "cme_reference_quality_report.py", ["--date", date], m, r)
    m, r = downstream_mark_and_reason(has_align_report)
    stage("ALIGNMENT_QUALITY", "alignment_quality_report.py", ["--date", date], m, r)
    m, r = downstream_mark_and_reason(has_bar_align)
    stage(
        "BAR_ALIGNMENT",
        "bar_alignment_quality_report.py",
        ["--date", date, "--write-buckets"],
        m,
        r,
    )
    m, r = downstream_mark_and_reason(has_gold)
    stage(
        "BUILD_BAR_FEATURES",
        "build_bar_features.py",
        ["--date", date, "--include-diagnostic-5s"],
        m,
        r,
    )
    m, r = downstream_mark_and_reason(False)
    stage("BAR_FEATURE_QUALITY", "bar_feature_quality_report.py", ["--date", date], m, r)
    m, r = downstream_mark_and_reason(has_eda)
    stage("FEATURE_EDA", "feature_eda_report.py", ["--date", date], m, r)
    return plan


def _has_alignment_report(reports_root: Path, date: str) -> bool:
    return (reports_root / f"alignment_quality_{date}.json").exists()


def _has_bar_alignment_report(reports_root: Path, date: str) -> bool:
    return (reports_root / f"bar_alignment_quality_{date}.json").exists()


def build_per_day_plan(
    config: DryRunConfig,
    date: str,
    python_exe: str,
    *,
    cost_estimate: Optional[CostEstimate] = None,
    physical_limit: Optional[PhysicalLimitCapability] = None,
) -> dict:
    if cost_estimate is None:
        cost_estimate = CostEstimate(
            status=NOT_REQUESTED_DRY_RUN,
            source="dry_run",
            message="dry-run: cost estimate not requested",
            max_estimated_cost_usd=config.max_estimated_cost_usd,
            max_estimated_size_gb=config.max_estimated_size_gb,
        )
    if physical_limit is None:
        physical_limit = PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
            message="dry-run: physical-limit capability not requested",
        )
    plan = build_day_plan(
        date,
        data_root=config.data_root,
        reports_root=config.reports_root,
        symbol_map=config.symbol_map,
        pre_roll_minutes=config.pre_roll_minutes,
        post_roll_minutes=config.post_roll_minutes,
    )
    candidates = discover_run_candidates(config.reports_root)
    selection: RunSelection = select_run_for_date(
        date, candidates, policy=config.run_selection_policy, run_id_map=config.run_id_map
    )
    has_cme_raw = bool(plan.cme_raw_files)
    has_cme_norm = bool(plan.cme_normalized_files)
    has_align_report = _has_alignment_report(config.reports_root, date)
    has_bar_align = _has_bar_alignment_report(config.reports_root, date)
    has_gold = bool(plan.gold_feature_files)
    has_eda = plan.eda_report_path is not None
    cme_raw_completeness_status: Optional[str] = None
    cme_raw_truncation_warning: Optional[str] = None
    if has_cme_raw and plan.cme_raw_files:
        try:
            raw_dir = Path(plan.cme_raw_files[0]).parent
            raw_docs = cme_dl.find_metadata_for_raw_dir(raw_dir)
            if raw_docs:
                cme_raw_completeness_status = cme_dl.directory_completeness_status(raw_docs)
        except Exception:
            cme_raw_completeness_status = None
    if cme_raw_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        cme_raw_truncation_warning = "CME raw sample is TRUNCATED_BY_LIMIT; downstream blocked unless --allow-truncated-cme-sample is passed"
    gate = decide_download_gate(
        GateInputs(
            download_cme=config.download_cme,
            allow_databento_download=config.allow_databento_download,
            acknowledge_cost_risk=config.acknowledge_cost_risk,
            allow_unestimated_download=config.allow_unestimated_download,
            cost_estimate_required=config.cost_estimate_required,
            require_physical_download_limit=config.require_physical_download_limit,
            allow_download_without_physical_limit=config.allow_download_without_physical_limit,
            dry_run=True,
            api_key_present=config.api_key_present,
            max_estimated_cost_usd=config.max_estimated_cost_usd,
            max_estimated_size_gb=config.max_estimated_size_gb,
            max_download_records=config.max_download_records,
            max_download_size_gb=config.max_download_size_gb,
            max_download_cost_usd=config.max_download_cost_usd,
        ),
        cost_estimate=cost_estimate,
        physical_limit=physical_limit,
    )
    stages = _plan_stages_for_date(
        config,
        date,
        python_exe,
        run_id=selection.selected_run_id,
        has_cme_raw=has_cme_raw,
        has_cme_norm=has_cme_norm,
        has_align_report=has_align_report,
        has_bar_align=has_bar_align,
        has_gold=has_gold,
        has_eda=has_eda,
        gate_decision=gate.decision,
    )
    snap = snapshot(config.data_root)
    downstream_present = any((has_cme_norm, has_align_report, has_bar_align, has_gold, has_eda))
    lineage_status: list[str] = []
    lineage_warnings: list[str] = []
    if not has_cme_raw:
        lineage_status.append(LINEAGE_MISSING_CME_RAW)
        if downstream_present:
            lineage_status.append(LINEAGE_DOWNSTREAM_WITHOUT_RAW_CME)
            lineage_status.append(LINEAGE_LEGACY_ARTIFACTS_PRESENT)
            lineage_status.append(LINEAGE_REBUILD_REQUIRED_AFTER_CME_DOWNLOAD)
            lineage_warnings.append(LINEAGE_UNTRUSTED_WARNING)
        else:
            lineage_status.append(LINEAGE_BLOCKED_MISSING_INPUT)
    else:
        lineage_status.append(LINEAGE_CLEAN_READY)
    window_hours: Optional[float] = None
    high_volume_warning: Optional[str] = None
    if plan.recommended_cme_start_utc and plan.recommended_cme_end_utc:
        t0 = _dt.datetime.fromisoformat(plan.recommended_cme_start_utc.replace("Z", "+00:00"))
        t1 = _dt.datetime.fromisoformat(plan.recommended_cme_end_utc.replace("Z", "+00:00"))
        window_hours = (t1 - t0).total_seconds() / 3600.0
        if window_hours > config.high_volume_window_hours:
            high_volume_warning = WARNING_HIGH_DATA_VOLUME_WINDOW
    next_action = _compute_next_action(
        download_cme=config.download_cme,
        allow_databento_download=config.allow_databento_download,
        acknowledge_cost_risk=config.acknowledge_cost_risk,
        gate_decision=gate.decision,
        has_cme_raw=has_cme_raw,
        downstream_present=downstream_present,
        high_volume_warning=high_volume_warning,
    )
    return {
        "date": date,
        "mt5_silver_exists": bool(plan.mt5_window_ms),
        "mt5_window_utc": [plan.mt5_min_utc, plan.mt5_max_utc],
        "recommended_databento_window_utc": [
            plan.recommended_cme_start_utc,
            plan.recommended_cme_end_utc,
        ],
        "recommended_databento_window_hours": window_hours,
        "verified_run_scoped_metadata_exists": selection.selected_run_id is not None,
        "run_selection": selection.to_dict(),
        "cme_raw_files_present": has_cme_raw,
        "cme_raw_completeness_status": cme_raw_completeness_status,
        "cme_raw_truncation_warning": cme_raw_truncation_warning,
        "cme_normalized_files_present": has_cme_norm,
        "alignment_report_present": has_align_report,
        "bar_alignment_report_present": has_bar_align,
        "gold_features_present": has_gold,
        "eda_report_present": has_eda,
        "downstream_artifacts_present": downstream_present,
        "lineage_status": lineage_status,
        "lineage_warnings": lineage_warnings,
        "high_volume_window_warning": high_volume_warning,
        "high_volume_window_hours_threshold": config.high_volume_window_hours,
        "planned_stages": stages,
        "would_real_download_be_attempted_if_not_dry_run": config.download_cme
        and config.allow_databento_download
        and config.acknowledge_cost_risk
        and config.api_key_present,
        "cost_estimate": cost_estimate.to_dict(),
        "physical_limit": physical_limit.to_dict(),
        "download_gate_decision": gate.to_dict(),
        "disk_free_gb_now": snap.disk_free_gb,
        "available_memory_gb_now": snap.available_memory_gb,
        "next_action": next_action,
    }


def _compute_next_action(
    *,
    download_cme: bool,
    allow_databento_download: bool,
    acknowledge_cost_risk: bool,
    gate_decision: str,
    has_cme_raw: bool,
    downstream_present: bool,
    high_volume_warning: Optional[str],
) -> str:
    if not has_cme_raw:
        if not download_cme:
            if downstream_present:
                return "Downstream artifacts exist without CME raw; pass --download-cme --allow-databento-download --acknowledge-cost-risk to fetch CME raw and rebuild downstream stages."
            return "Pass --download-cme to plan a Databento fetch for this date."
        if not allow_databento_download:
            return "Add --allow-databento-download to authorize a real download."
        if not acknowledge_cost_risk:
            return "Add --acknowledge-cost-risk to authorize a real download."
        if gate_decision == DOWNLOAD_BLOCKED_DRY_RUN:
            extra = (
                f" (window length exceeds threshold: {high_volume_warning})"
                if high_volume_warning
                else ""
            )
            return f"Dry-run only -- re-run WITHOUT --dry-run to attempt the download once the dry-run report has been reviewed{extra}."
        if gate_decision in (
            DOWNLOAD_BLOCKED_BY_COST_GUARD,
            DOWNLOAD_BLOCKED_BY_COST_ESTIMATE_UNAVAILABLE,
            DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD,
            DOWNLOAD_BLOCKED_MISSING_ACK,
            DOWNLOAD_BLOCKED_MISSING_API_KEY,
        ):
            return f"Download blocked: {gate_decision} -- review thresholds and overrides."
        if high_volume_warning:
            return "Approved by gates, but recommended Databento window exceeds the high-volume threshold; confirm the budget before running."
        return "All gates pass -- ready to run without --dry-run."
    return "CME raw present -- downstream stages will run in order."


def build_multiday_dry_run_report(config: DryRunConfig, *, python_exe: str) -> dict:
    per_day = [build_per_day_plan(config, d, python_exe=python_exe) for d in config.dates]
    return {
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "dates_requested": list(config.dates),
        "data_root": str(config.data_root),
        "reports_root": str(config.reports_root),
        "repo_root": str(config.repo_root),
        "symbol_map": dict(config.symbol_map),
        "run_selection_policy": config.run_selection_policy,
        "run_id_map": dict(config.run_id_map),
        "download_flags": {
            "download_cme": config.download_cme,
            "allow_databento_download": config.allow_databento_download,
            "acknowledge_cost_risk": config.acknowledge_cost_risk,
            "allow_unestimated_download": config.allow_unestimated_download,
            "cost_estimate_required": config.cost_estimate_required,
            "require_physical_download_limit": config.require_physical_download_limit,
            "allow_download_without_physical_limit": config.allow_download_without_physical_limit,
        },
        "thresholds": {
            "max_estimated_cost_usd": config.max_estimated_cost_usd,
            "max_estimated_size_gb": config.max_estimated_size_gb,
            "max_download_size_gb": config.max_download_size_gb,
            "max_download_cost_usd": config.max_download_cost_usd,
            "max_download_records": config.max_download_records,
        },
        "per_day": per_day,
        "this_report": "dry-run only; no Databento call, no network I/O",
    }


def render_dry_run_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Polarix Multi-Day Dry-Run Report")
    lines.append("=" * 80)
    lines.append(f"generated_at_utc      : {report['generated_at_utc']}")
    lines.append(f"data_root             : {report['data_root']}")
    lines.append(f"reports_root          : {report['reports_root']}")
    lines.append(f"symbol_map            : {report['symbol_map']}")
    lines.append(f"run_selection_policy  : {report['run_selection_policy']}")
    lines.append(f"download_flags        : {report['download_flags']}")
    lines.append(f"thresholds            : {report['thresholds']}")
    lines.append("")
    for day in report["per_day"]:
        lines.append("-" * 80)
        lines.append(f"DATE {day['date']}")
        lines.append(f"  mt5_silver_exists           : {day['mt5_silver_exists']}")
        lines.append(
            f"  mt5_window_utc              : {day['mt5_window_utc'][0]} -> {day['mt5_window_utc'][1]}"
        )
        lines.append(
            f"  recommended_databento_window: {day['recommended_databento_window_utc'][0]} -> {day['recommended_databento_window_utc'][1]}"
        )
        lines.append(
            f"  verified_run_metadata_exists: {day['verified_run_scoped_metadata_exists']}"
        )
        sel = day["run_selection"]
        lines.append(f"  selected_run_id             : {sel.get('selected_run_id')}")
        if sel.get("error_reason"):
            lines.append(f"  selection_error_reason      : {sel['error_reason']}")
        if sel.get("warnings"):
            lines.append(f"  selection_warnings          : {sel['warnings']}")
        lines.append(f"  candidates                  : {[c['run_id'] for c in sel['candidates']]}")
        if sel.get("rejected"):
            lines.append(
                "  rejected                    : "
                + ", ".join((f"{r['run_id']} ({r['rejected_reason']})" for r in sel["rejected"]))
            )
        lines.append(f"  cme_raw_present             : {day['cme_raw_files_present']}")
        if day.get("cme_raw_completeness_status"):
            lines.append(f"  cme_raw_completeness        : {day['cme_raw_completeness_status']}")
        if day.get("cme_raw_truncation_warning"):
            lines.append(f"  cme_raw_truncation_warning  : {day['cme_raw_truncation_warning']}")
        lines.append(f"  cme_normalized_present      : {day['cme_normalized_files_present']}")
        lines.append(f"  alignment_report_present    : {day['alignment_report_present']}")
        lines.append(f"  bar_alignment_present       : {day['bar_alignment_report_present']}")
        lines.append(f"  gold_features_present       : {day['gold_features_present']}")
        lines.append(f"  eda_report_present          : {day['eda_report_present']}")
        lines.append(
            f"  real_download_if_real_run   : {day['would_real_download_be_attempted_if_not_dry_run']}"
        )
        gate = day["download_gate_decision"]
        lines.append(f"  download_gate_decision      : {gate['decision']}")
        if gate.get("critical_warnings"):
            lines.append(f"  critical_warnings           : {gate['critical_warnings']}")
        lines.append(f"  cost_estimate_status        : {day['cost_estimate']['status']}")
        lines.append(f"  physical_limit_status       : {day['physical_limit']['status']}")
        lineage = day.get("lineage_status") or []
        if lineage:
            lines.append(f"  lineage_status              : {lineage}")
        for w in day.get("lineage_warnings") or []:
            lines.append(f"  lineage_warning             : {w}")
        if day.get("high_volume_window_warning"):
            lines.append(
                f"  high_volume_window_warning  : {day['high_volume_window_warning']} (window={day.get('recommended_databento_window_hours')}h, threshold={day.get('high_volume_window_hours_threshold')}h)"
            )
        if day.get("next_action"):
            lines.append(f"  next_action                 : {day['next_action']}")
        lines.append("  planned_stages:")
        for s in day["planned_stages"]:
            mark = s.get("mark") or ("RUN" if s["would_run"] else "SKIP")
            reason = s.get("reason") or s.get("skip_reason")
            tail = f"  ({reason})" if reason else ""
            lines.append(f"    [{mark}] {s['name']}{tail}")
        lines.append(f"  disk_free_gb                : {day['disk_free_gb_now']}")
        lines.append(f"  available_memory_gb         : {day['available_memory_gb_now']}")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"
