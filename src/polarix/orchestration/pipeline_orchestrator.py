from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from polarix.ingestion import cme_downloader as cme_dl
from polarix.orchestration import pipeline_status as ps
from polarix.orchestration import stage_dependencies as sd
from polarix.orchestration.databento_download_plan import (
    DownloadOutcome,
    plan_day_download,
    run_day_download,
)
from polarix.orchestration.day_plan import DEFAULT_SYMBOL_MAP, build_day_plan
from polarix.orchestration.multiday_dry_run_report import (
    DryRunConfig,
    build_multiday_dry_run_report,
    render_dry_run_text,
)
from polarix.orchestration.resource_guards import (
    ResourceGuardError,
    ResourceSnapshot,
    assert_disk_free,
    assert_memory_available,
    snapshot,
)
from polarix.orchestration.run_selection import POLICY_LONGEST_VERIFIED, VALID_POLICIES
from polarix.orchestration.subprocess_guard import (
    DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS,
    ChildProcessRegistry,
)

DEFAULT_MIN_FREE_DISK_GB = 10.0
DEFAULT_MIN_AVAILABLE_MEMORY_GB = 2.0
DEFAULT_PRE_ROLL_MINUTES = 5
DEFAULT_POST_ROLL_MINUTES = 5
DEFAULT_MAX_DATABENTO_RETRIES_PER_DAY = 1
DEFAULT_STAGE_TIMEOUT_SECONDS = 3600


@dataclass
class OrchestratorConfig:
    dates: list[str]
    data_root: Path
    reports_root: Path
    repo_root: Path
    python_exe: Path
    symbol_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOL_MAP))
    pre_roll_minutes: int = DEFAULT_PRE_ROLL_MINUTES
    post_roll_minutes: int = DEFAULT_POST_ROLL_MINUTES
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB
    download_cme: bool = False
    allow_databento_download: bool = False
    continue_on_missing_mt5: bool = False
    continue_on_missing_cme: bool = True
    continue_on_day_failure: bool = True
    dry_run: bool = False
    force: bool = False
    max_databento_retries_per_day: int = DEFAULT_MAX_DATABENTO_RETRIES_PER_DAY
    stage_timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS
    run_selection_policy: str = POLICY_LONGEST_VERIFIED
    run_id_map: Mapping[str, str] = field(default_factory=dict)
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
    child_terminate_timeout_seconds: float = DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS
    kill_children_on_interrupt: bool = True
    allow_truncated_cme_sample: bool = False
    rebuild_existing: bool = False

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        self.repo_root = Path(self.repo_root).resolve()
        self.python_exe = Path(self.python_exe).resolve()
        if self.run_selection_policy not in VALID_POLICIES:
            raise ValueError(
                f"run_selection_policy={self.run_selection_policy!r} must be one of {VALID_POLICIES}"
            )


StageRunner = Callable[[list[str], int, Path], subprocess.CompletedProcess]


def _default_stage_runner(argv: list[str], timeout: int, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


@dataclass
class StageRecord:
    name: str
    argv: list[str]
    returncode: Optional[int]
    status: str
    stdout_tail: str
    stderr_tail: str
    started_at_utc: str
    ended_at_utc: str
    disk_before: float
    disk_after: float
    memory_before: float
    memory_after: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "status": self.status,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "disk_before": self.disk_before,
            "disk_after": self.disk_after,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
        }


def _read_decision(reports_root: Path, filename: str) -> Optional[str]:
    p = reports_root / filename
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc.get("quality_decision") or doc.get("decision") or None


@dataclass
class DayReport:
    date: str
    started_at_utc: str
    ended_at_utc: Optional[str] = None
    stages: list[dict] = field(default_factory=list)
    stage_statuses: list[str] = field(default_factory=list)
    download_outcome: Optional[dict] = None
    day_plan_snapshot: Optional[dict] = None
    final_status: str = ps.STATUS_DAY_FAILED
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_produced: list[str] = field(default_factory=list)
    disk_min: Optional[float] = None
    disk_max: Optional[float] = None
    memory_min: Optional[float] = None
    memory_max: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "stages": list(self.stages),
            "stage_statuses": list(self.stage_statuses),
            "download_outcome": self.download_outcome,
            "day_plan_snapshot": self.day_plan_snapshot,
            "final_status": self.final_status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "files_produced": list(self.files_produced),
            "disk_min": self.disk_min,
            "disk_max": self.disk_max,
            "memory_min": self.memory_min,
            "memory_max": self.memory_max,
        }


def _update_resource_extremes(report: DayReport, snap: ResourceSnapshot) -> None:
    report.disk_min = (
        snap.disk_free_gb if report.disk_min is None else min(report.disk_min, snap.disk_free_gb)
    )
    report.disk_max = (
        snap.disk_free_gb if report.disk_max is None else max(report.disk_max, snap.disk_free_gb)
    )
    report.memory_min = (
        snap.available_memory_gb
        if report.memory_min is None
        else min(report.memory_min, snap.available_memory_gb)
    )
    report.memory_max = (
        snap.available_memory_gb
        if report.memory_max is None
        else max(report.memory_max, snap.available_memory_gb)
    )


def _run_stage(
    config: OrchestratorConfig, report: DayReport, name: str, argv: list[str], runner: StageRunner
) -> StageRecord:
    started = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    snap0 = snapshot(config.data_root)
    _update_resource_extremes(report, snap0)
    try:
        assert_disk_free(config.data_root, config.min_free_disk_gb)
        assert_memory_available(config.min_available_memory_gb)
    except ResourceGuardError as exc:
        stage = StageRecord(
            name=name,
            argv=argv,
            returncode=None,
            status=exc.reason,
            stdout_tail="",
            stderr_tail=str(exc),
            started_at_utc=started,
            ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            disk_before=snap0.disk_free_gb,
            disk_after=snap0.disk_free_gb,
            memory_before=snap0.available_memory_gb,
            memory_after=snap0.available_memory_gb,
        )
        report.stages.append(stage.to_dict())
        report.stage_statuses.append(exc.reason)
        return stage
    try:
        result = runner(argv, config.stage_timeout_seconds, config.repo_root)
        rc = result.returncode
        stdout_tail = (result.stdout or "")[-2000:]
        stderr_tail = (result.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as exc:
        rc = None
        stdout_tail = ""
        stderr_tail = f"TIMEOUT after {config.stage_timeout_seconds}s: {exc}"
    except Exception as exc:
        rc = None
        stdout_tail = ""
        stderr_tail = f"runner exception: {exc!r}"
    snap1 = snapshot(config.data_root)
    _update_resource_extremes(report, snap1)
    status = "OK" if rc == 0 else "FAIL"
    stage = StageRecord(
        name=name,
        argv=argv,
        returncode=rc,
        status=status,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        started_at_utc=started,
        ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        disk_before=snap0.disk_free_gb,
        disk_after=snap1.disk_free_gb,
        memory_before=snap0.available_memory_gb,
        memory_after=snap1.available_memory_gb,
    )
    report.stages.append(stage.to_dict())
    report.stage_statuses.append(f"{name}:{status}")
    return stage


def _scripts_path(config: OrchestratorConfig, name: str) -> str:
    return str(config.repo_root / "scripts" / name)


def _stage_argv(config: OrchestratorConfig, script: str, args: list[str]) -> list[str]:
    return [str(config.python_exe), "-u", _scripts_path(config, script), *args]


def process_one_date(
    config: OrchestratorConfig,
    date: str,
    *,
    runner: StageRunner | None = None,
    download_runner: Optional[Callable] = None,
) -> DayReport:
    runner = runner or _default_stage_runner
    report = DayReport(date=date, started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat())
    snap0 = snapshot(config.data_root)
    _update_resource_extremes(report, snap0)
    try:
        assert_disk_free(config.data_root, config.min_free_disk_gb)
        assert_memory_available(config.min_available_memory_gb)
    except ResourceGuardError as exc:
        report.final_status = exc.reason
        report.errors.append(str(exc))
        report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return report
    plan = build_day_plan(
        date,
        data_root=config.data_root,
        reports_root=config.reports_root,
        symbol_map=config.symbol_map,
        pre_roll_minutes=config.pre_roll_minutes,
        post_roll_minutes=config.post_roll_minutes,
    )
    report.day_plan_snapshot = plan.to_dict()
    if plan.mt5_window_ms is None:
        if config.continue_on_missing_mt5:
            report.final_status = ps.STATUS_SKIPPED_NO_MT5_SILVER
            report.warnings.append("no MT5 Silver for this date; skipped")
        else:
            report.final_status = ps.STATUS_DAY_FAILED
            report.errors.append("no MT5 Silver for this date; continue_on_missing_mt5 is false")
        report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return report
    cme_raw_present = bool(plan.cme_raw_files)
    if cme_raw_present and (not config.force):
        report.warnings.append(f"CME raw already present for {date}; skipping download")
    plan_obj = plan_day_download(
        date=date,
        mt5_window_ms=plan.mt5_window_ms,
        output_root=config.data_root / "raw" / "cme_sample",
        pre_roll_minutes=config.pre_roll_minutes,
        post_roll_minutes=config.post_roll_minutes,
        allow_databento_download=config.allow_databento_download,
        download_cme=config.download_cme,
        max_retries=config.max_databento_retries_per_day,
        max_download_records=config.max_download_records,
        allow_download_without_physical_limit=config.allow_download_without_physical_limit,
    )
    report.stage_statuses.append("DATABENTO_PLAN_BUILT")
    if config.dry_run:
        report.download_outcome = {
            "status": "DRY_RUN",
            "plan": plan_obj.to_dict(),
            "would_download": plan_obj.will_download,
        }
    elif cme_raw_present and (not config.force):
        report.download_outcome = {
            "status": ps.STATUS_SKIPPED_CME_EXISTS,
            "plan": plan_obj.to_dict(),
        }
    else:
        try:
            assert_disk_free(config.data_root, config.min_free_disk_gb)
        except ResourceGuardError as exc:
            report.final_status = exc.reason
            report.errors.append(str(exc))
            report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            return report
        outcome: DownloadOutcome = run_day_download(
            plan_obj, repo_root=config.repo_root, runner=download_runner
        )
        report.download_outcome = outcome.to_dict()
        report.stage_statuses.append(outcome.status)
        if outcome.status == ps.STATUS_DATABENTO_RANGE_UNAVAILABLE:
            if config.continue_on_missing_cme:
                report.warnings.append(
                    f"Databento range unavailable for {date}; continuing without CME"
                )
                report.final_status = ps.STATUS_DATABENTO_RANGE_UNAVAILABLE
                report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
                return report
            else:
                report.final_status = ps.STATUS_DAY_FAILED
                report.errors.append("Databento range unavailable; continue_on_missing_cme false")
                report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
                return report
        elif outcome.status == ps.STATUS_DATABENTO_DOWNLOAD_FAILED:
            if not config.continue_on_day_failure:
                report.final_status = ps.STATUS_DAY_FAILED
                report.errors.append("Databento download failed")
                report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
                return report
            report.warnings.append("Databento download failed; continuing")
            report.final_status = ps.STATUS_DATABENTO_DOWNLOAD_FAILED
            report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            return report
    plan_after = build_day_plan(
        date,
        data_root=config.data_root,
        reports_root=config.reports_root,
        symbol_map=config.symbol_map,
        pre_roll_minutes=config.pre_roll_minutes,
        post_roll_minutes=config.post_roll_minutes,
    )
    if not plan_after.cme_raw_files:
        if config.continue_on_missing_cme:
            report.final_status = ps.STATUS_DATABENTO_RANGE_UNAVAILABLE
            report.warnings.append("no CME raw data for this date; downstream stages skipped")
        else:
            report.final_status = ps.STATUS_DAY_FAILED
            report.errors.append("no CME raw data and continue_on_missing_cme is false")
        report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return report
    completeness: Optional[str] = None
    try:
        raw_dir = Path(plan_after.cme_raw_files[0]).parent
        docs = cme_dl.find_metadata_for_raw_dir(raw_dir)
        completeness = cme_dl.directory_completeness_status(docs) if docs else None
    except Exception as exc:
        report.warnings.append(f"failed to read CME raw metadata sidecar(s): {exc}")
    if completeness == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        report.stage_statuses.append(ps.STATUS_CME_RAW_TRUNCATED_BY_LIMIT)
        if not config.allow_truncated_cme_sample and (not config.dry_run):
            report.final_status = ps.STATUS_TRUNCATED_SAMPLE_BLOCKED
            report.errors.append(
                "CME raw sample is TRUNCATED_BY_LIMIT (capped by --max-download-records); downstream stages blocked. Pass --allow-truncated-cme-sample to proceed in exploratory mode."
            )
            report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
            return report
        if config.allow_truncated_cme_sample:
            report.stage_statuses.append(ps.STATUS_TRUNCATED_SAMPLE_ALLOWED_EXPLORATORY)
            report.warnings.append(
                "CME raw sample is TRUNCATED_BY_LIMIT; proceeding under --allow-truncated-cme-sample (exploratory only)"
            )
    elif completeness == cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT:
        report.stage_statuses.append(ps.STATUS_CME_RAW_COMPLETE_WITHIN_LIMIT)
    elif completeness is not None:
        report.stage_statuses.append(ps.STATUS_CME_RAW_UNKNOWN_COMPLETENESS)
        report.warnings.append(
            f"CME raw completeness status is {completeness!r}; downstream stages will run but truncation cannot be ruled out"
        )
    cme_raw_trust = sd.evaluate_cme_raw_trust(
        raw_files=list(plan_after.cme_raw_files),
        allow_truncated_cme_sample=config.allow_truncated_cme_sample,
    )
    cme_symbols = list(config.symbol_map.keys())
    symbol_pairs = [f"{cme}_{mt5}" for cme, mt5 in config.symbol_map.items()]
    artifact_presence: dict[str, bool] = {}
    trust_verdicts: dict[str, sd.StageTrustVerdict] = {}
    for spec in sd.DEFAULT_STAGE_GRAPH:
        present = sd.detect_artifact_present(
            spec,
            date=date,
            data_root=config.data_root,
            reports_root=config.reports_root,
            cme_symbols=cme_symbols,
            symbol_pairs=symbol_pairs,
        )
        artifact_presence[spec.name] = present
        trust_verdicts[spec.name] = sd.evaluate_stage_trust(
            spec,
            artifact_present=present,
            cme_raw_trust=cme_raw_trust,
            upstream_trust=trust_verdicts,
        )
    stage_status: dict[str, str] = {}
    stage_records: dict[str, dict] = {}
    for spec in sd.DEFAULT_STAGE_GRAPH:
        present = artifact_presence[spec.name]
        trust = trust_verdicts[spec.name]
        force_for_legacy = config.force and spec.supports_force
        rebuild_with_force = config.rebuild_existing and present and spec.supports_force
        force_passed = bool(force_for_legacy or rebuild_with_force)
        extra_args: list[str] = []
        if spec.name == "CME_INGEST":
            extra_args = [
                "--input-root",
                str(config.data_root / "raw" / "cme_sample"),
                "--output-root",
                str(config.data_root / "normalized" / "cme_reference"),
            ]
        argv = _stage_argv(
            config, spec.script, [*extra_args, *spec.date_argv_tail(date, with_force=force_passed)]
        )
        upstream_failed = sd.first_upstream_failed(spec, stage_status)
        if upstream_failed is not None:
            rec = _make_stage_record(
                spec,
                argv=argv,
                status=ps.STAGE_SKIPPED_UPSTREAM_FAILED,
                returncode=None,
                stdout_tail="",
                stderr_tail="",
                artifact_present=present,
                artifact_trust_status=trust.artifact_trust_status,
                old_artifact_used=present,
                force_passed=False,
                skip_reason=f"upstream {upstream_failed} not satisfied",
                reason=f"upstream {upstream_failed} not satisfied",
                upstream_failed_stage=upstream_failed,
                dependency_status="UNSATISFIED",
                started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )
            stage_records[spec.name] = rec
            stage_status[spec.name] = rec["status"]
            report.stages.append(rec)
            report.stage_statuses.append(f"{spec.name}:{rec['status']}")
            if present:
                report.warnings.append(
                    f"{spec.name}: skipping although on-disk artifact exists -- upstream {upstream_failed} is not satisfied, so the existing artifact cannot be trusted as authoritative"
                )
            continue
        if config.dry_run:
            rec = _make_stage_record(
                spec,
                argv=argv,
                status="DRY_RUN",
                returncode=None,
                stdout_tail="",
                stderr_tail="",
                artifact_present=present,
                artifact_trust_status=trust.artifact_trust_status,
                old_artifact_used=False,
                force_passed=force_passed,
                skip_reason=None,
                reason="dry-run",
                upstream_failed_stage=None,
                dependency_status="DRY_RUN",
                started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                dry_run=True,
            )
            stage_records[spec.name] = rec
            stage_status[spec.name] = ps.STAGE_OK
            report.stages.append(rec)
            report.stage_statuses.append(f"{spec.name}:DRY_RUN")
            continue
        if present and trust.trust_chain_intact and (not config.rebuild_existing):
            rec = _make_stage_record(
                spec,
                argv=argv,
                status=ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED,
                returncode=None,
                stdout_tail="",
                stderr_tail="",
                artifact_present=True,
                artifact_trust_status=trust.artifact_trust_status,
                old_artifact_used=True,
                force_passed=False,
                skip_reason="trusted artifact already exists",
                reason=trust.reason,
                upstream_failed_stage=None,
                dependency_status="SATISFIED",
                started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )
            stage_records[spec.name] = rec
            stage_status[spec.name] = rec["status"]
            report.stages.append(rec)
            report.stage_statuses.append(f"{spec.name}:{rec['status']}")
            continue
        if present and (not trust.trust_chain_intact) and (not config.rebuild_existing):
            rec = _make_stage_record(
                spec,
                argv=argv,
                status=ps.STAGE_SKIPPED_ALREADY_EXISTS_UNTRUSTED,
                returncode=None,
                stdout_tail="",
                stderr_tail="",
                artifact_present=True,
                artifact_trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
                old_artifact_used=False,
                force_passed=False,
                skip_reason="artifact present but lineage is untrusted",
                reason=trust.reason,
                upstream_failed_stage=None,
                dependency_status="UNSATISFIED",
                started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            )
            stage_records[spec.name] = rec
            stage_status[spec.name] = rec["status"]
            report.stages.append(rec)
            report.stage_statuses.append(f"{spec.name}:{rec['status']}")
            report.warnings.append(
                f"{spec.name}: on-disk artifact is UNTRUSTED ({trust.reason}); pass --rebuild-existing to overwrite it, or restore lineage."
            )
            continue
        stage = _run_stage(config, report, spec.name, argv, runner)
        status = ps.STAGE_OK if stage.status == "OK" else ps.STAGE_FAIL
        rec = _make_stage_record(
            spec,
            argv=argv,
            status=status,
            returncode=stage.returncode,
            stdout_tail=stage.stdout_tail,
            stderr_tail=stage.stderr_tail,
            artifact_present=present,
            artifact_trust_status=trust.artifact_trust_status,
            old_artifact_used=False,
            force_passed=force_passed,
            skip_reason=None,
            reason="executed; --force passed" if force_passed else "executed",
            upstream_failed_stage=None,
            dependency_status="SATISFIED" if status == ps.STAGE_OK else "FAILED",
            started_at_utc=stage.started_at_utc,
            ended_at_utc=stage.ended_at_utc,
            disk_before=stage.disk_before,
            disk_after=stage.disk_after,
            memory_before=stage.memory_before,
            memory_after=stage.memory_after,
        )
        if report.stages and report.stages[-1].get("name") == spec.name:
            report.stages[-1] = rec
        else:
            report.stages.append(rec)
        if report.stage_statuses and report.stage_statuses[-1].startswith(f"{spec.name}:"):
            report.stage_statuses[-1] = f"{spec.name}:{status}"
        else:
            report.stage_statuses.append(f"{spec.name}:{status}")
        stage_records[spec.name] = rec
        stage_status[spec.name] = status
        if status == ps.STAGE_FAIL and (not config.continue_on_day_failure):
            break
    if config.dry_run:
        report.final_status = ps.STATUS_DAY_COMPLETE
    else:
        report.final_status = sd.classify_day_status(
            stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=stage_status
        )
    if not config.dry_run and sd.zombie_artifact_risk(
        stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_records=stage_records
    ):
        report.warnings.append(
            "Phase 2G.3 prevented an invalid DAY_COMPLETE: at least one downstream artifact existed on disk while an upstream required stage was not satisfied. Downstream stages were skipped instead of being allowed to read stale artifacts."
        )
    if not config.dry_run:
        for spec in sd.DEFAULT_STAGE_GRAPH:
            if stage_status.get(spec.name) != ps.STAGE_OK:
                continue
            json_name = {
                "CME_QUALITY": f"cme_reference_quality_{date}.json",
                "ALIGNMENT_QUALITY": f"alignment_quality_{date}.json",
                "BAR_ALIGNMENT": f"bar_alignment_quality_{date}.json",
                "BAR_FEATURE_QUALITY": f"bar_feature_quality_{date}.json",
                "FEATURE_EDA": f"feature_eda_{date}.json",
            }.get(spec.name)
            prefix = {
                "CME_QUALITY": "CME_QUALITY",
                "ALIGNMENT_QUALITY": "ALIGNMENT",
                "BAR_ALIGNMENT": "BAR_ALIGNMENT",
                "BAR_FEATURE_QUALITY": "GOLD_FEATURES",
                "FEATURE_EDA": "EDA",
            }.get(spec.name)
            if not json_name or not prefix:
                continue
            decision = _read_decision(config.reports_root, json_name)
            if decision:
                report.stage_statuses.append(ps.map_decision_to_stage(prefix, decision))
    report.ended_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    return report


def _make_stage_record(
    spec: sd.StageSpec,
    *,
    argv: list[str],
    status: str,
    returncode: Optional[int],
    stdout_tail: str,
    stderr_tail: str,
    artifact_present: bool,
    artifact_trust_status: str,
    old_artifact_used: bool,
    force_passed: bool,
    skip_reason: Optional[str],
    reason: str,
    upstream_failed_stage: Optional[str],
    dependency_status: str,
    started_at_utc: str,
    ended_at_utc: str,
    disk_before: Optional[float] = None,
    disk_after: Optional[float] = None,
    memory_before: Optional[float] = None,
    memory_after: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    return {
        "name": spec.name,
        "argv": list(argv),
        "status": status,
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "artifact_present": artifact_present,
        "artifact_trust_status": artifact_trust_status,
        "old_artifact_used": old_artifact_used,
        "force_passed": force_passed,
        "skip_reason": skip_reason,
        "reason": reason,
        "upstream_failed_stage": upstream_failed_stage,
        "upstream_dependencies": list(spec.depends_on),
        "dependency_status": dependency_status,
        "is_required_stage": spec.is_required,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "disk_before": disk_before,
        "disk_after": disk_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "dry_run": dry_run,
    }


@dataclass
class OrchestratorResult:
    config: OrchestratorConfig
    per_day: list[DayReport]
    summary: dict
    summary_json_path: Optional[Path]
    summary_txt_path: Optional[Path]
    per_day_json_paths: list[Path]
    interrupted: bool = False
    child_cleanup_records: list[dict] = field(default_factory=list)
    dry_run_report: Optional[dict] = None
    dry_run_report_json_path: Optional[Path] = None
    dry_run_report_txt_path: Optional[Path] = None


def _write_per_day_report(reports_root: Path, day: DayReport) -> tuple[Path, Path]:
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"multiday_pipeline_{day.date}.json"
    txt_path = reports_root / f"multiday_pipeline_{day.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(day.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(json_tmp, json_path)
    txt = _render_day_text(day)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(txt, encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    return (json_path, txt_path)


def _render_day_text(day: DayReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"Polarix Multi-Day Pipeline Day Report  --  date={day.date}")
    lines.append("=" * 80)
    lines.append(f"Final status        : {day.final_status}")
    lines.append(f"Started             : {day.started_at_utc}")
    lines.append(f"Ended               : {day.ended_at_utc}")
    lines.append(f"Disk free (min/max) : {day.disk_min} / {day.disk_max}")
    lines.append(f"Memory free (min/max): {day.memory_min} / {day.memory_max}")
    lines.append(f"Stage statuses      : {day.stage_statuses}")
    if day.download_outcome:
        lines.append(f"Download outcome    : {day.download_outcome.get('status')}")
    if day.warnings:
        lines.append("Warnings:")
        for w in day.warnings:
            lines.append(f"  - {w}")
    if day.errors:
        lines.append("Errors:")
        for w in day.errors:
            lines.append(f"  - {w}")
    lines.append("Stages:")
    for stage in day.stages:
        lines.append(
            f"  {stage.get('name')}: status={stage.get('status')} rc={stage.get('returncode')}"
        )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def _render_summary_text(summary: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Polarix Multi-Day Pipeline Summary")
    lines.append("=" * 80)
    lines.append(f"dates_requested      : {summary['dates_requested']}")
    lines.append(f"dates_completed      : {summary['dates_completed']}")
    lines.append(f"dates_skipped        : {summary['dates_skipped']}")
    lines.append(f"dates_failed         : {summary['dates_failed']}")
    lines.append(f"disk_free_min/max    : {summary['disk_free_min']} / {summary['disk_free_max']}")
    lines.append("per_date_statuses    :")
    for d in summary["per_date_status"]:
        lines.append(f"  {d['date']}: {d['final_status']}")
    if summary.get("failures_by_reason"):
        lines.append(f"failures_by_reason   : {summary['failures_by_reason']}")
    lines.append(f"next_recommended_action: {summary.get('next_recommended_action')}")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def run_multiday(
    config: OrchestratorConfig,
    *,
    runner: StageRunner | None = None,
    download_runner: Optional[Callable] = None,
    registry: Optional[ChildProcessRegistry] = None,
) -> OrchestratorResult:
    per_day: list[DayReport] = []
    per_day_json_paths: list[Path] = []
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    interrupted = False
    child_cleanup_records: list[dict] = []
    registry = registry if registry is not None else ChildProcessRegistry()
    dry_run_report: Optional[dict] = None
    dry_run_report_json_path: Optional[Path] = None
    dry_run_report_txt_path: Optional[Path] = None
    if config.dry_run:
        dry_cfg = DryRunConfig(
            dates=list(config.dates),
            data_root=config.data_root,
            reports_root=config.reports_root,
            repo_root=config.repo_root,
            symbol_map=dict(config.symbol_map),
            pre_roll_minutes=config.pre_roll_minutes,
            post_roll_minutes=config.post_roll_minutes,
            run_selection_policy=config.run_selection_policy,
            run_id_map=dict(config.run_id_map),
            download_cme=config.download_cme,
            allow_databento_download=config.allow_databento_download,
            acknowledge_cost_risk=config.acknowledge_cost_risk,
            allow_unestimated_download=config.allow_unestimated_download,
            cost_estimate_required=config.cost_estimate_required,
            require_physical_download_limit=config.require_physical_download_limit,
            allow_download_without_physical_limit=config.allow_download_without_physical_limit,
            max_estimated_cost_usd=config.max_estimated_cost_usd,
            max_estimated_size_gb=config.max_estimated_size_gb,
            max_download_size_gb=config.max_download_size_gb,
            max_download_cost_usd=config.max_download_cost_usd,
            max_download_records=config.max_download_records,
            api_key_present=bool(os.environ.get("DATABENTO_API_KEY")),
        )
        dry_run_report = build_multiday_dry_run_report(dry_cfg, python_exe=str(config.python_exe))
    try:
        for date in config.dates:
            day = process_one_date(config, date, runner=runner, download_runner=download_runner)
            per_day.append(day)
            if not config.dry_run:
                json_path, _txt_path = _write_per_day_report(config.reports_root, day)
                per_day_json_paths.append(json_path)
            if day.final_status == ps.STATUS_DAY_FAILED and (not config.continue_on_day_failure):
                break
            if day.final_status in (ps.STATUS_DISK_GUARD_STOP, ps.STATUS_MEMORY_GUARD_STOP):
                break
    except KeyboardInterrupt:
        interrupted = True
        child_cleanup_records = [
            r.to_dict()
            for r in registry.cleanup(
                timeout_seconds=config.child_terminate_timeout_seconds,
                kill_on_timeout=config.kill_children_on_interrupt,
            )
        ]
        if not per_day:
            per_day.append(
                DayReport(
                    date=config.dates[0] if config.dates else "",
                    started_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                    final_status=ps.STATUS_INTERRUPTED,
                    ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                    warnings=["KeyboardInterrupt before any date started"],
                )
            )
        else:
            per_day[-1].final_status = ps.STATUS_INTERRUPTED
            per_day[-1].warnings.append("KeyboardInterrupt during this date")
    finally:
        if not interrupted and registry.active_count() > 0:
            child_cleanup_records.extend(
                (
                    r.to_dict()
                    for r in registry.cleanup(
                        timeout_seconds=config.child_terminate_timeout_seconds,
                        kill_on_timeout=config.kill_children_on_interrupt,
                    )
                )
            )
    dates_completed = [d.date for d in per_day if d.final_status == ps.STATUS_DAY_COMPLETE]
    dates_skipped = [
        d.date
        for d in per_day
        if d.final_status
        in (ps.STATUS_SKIPPED_NO_MT5_SILVER, ps.STATUS_DATABENTO_RANGE_UNAVAILABLE)
    ]
    dates_failed = [d.date for d in per_day if d.final_status == ps.STATUS_DAY_FAILED]
    dates_partial = [d.date for d in per_day if d.final_status == ps.STATUS_DAY_PARTIAL]
    failures_by_stage: dict[str, int] = {}
    days_with_zombie_artifact_risk: list[str] = []
    for d in per_day:
        for stage in d.stages:
            status = stage.get("status")
            if status == ps.STAGE_FAIL:
                name = stage.get("name", "<unknown>")
                failures_by_stage[name] = failures_by_stage.get(name, 0) + 1
        for w in d.warnings:
            if "Phase 2G.3 prevented an invalid DAY_COMPLETE" in w:
                days_with_zombie_artifact_risk.append(d.date)
                break
    invalid_day_complete_prevented = bool(days_with_zombie_artifact_risk)
    failures_by_reason: dict[str, int] = {}
    for d in per_day:
        if d.final_status in (ps.STATUS_DAY_COMPLETE,):
            continue
        failures_by_reason[d.final_status] = failures_by_reason.get(d.final_status, 0) + 1
    disk_mins = [d.disk_min for d in per_day if d.disk_min is not None]
    disk_maxs = [d.disk_max for d in per_day if d.disk_max is not None]
    global_guard_stopped = any(
        (
            d.final_status in (ps.STATUS_DISK_GUARD_STOP, ps.STATUS_MEMORY_GUARD_STOP)
            for d in per_day
        )
    )
    pipeline_continued_after_failure = (
        (dates_failed or dates_skipped)
        and len(per_day) == len(config.dates)
        and (not global_guard_stopped)
    )
    summary = {
        "dates_requested": list(config.dates),
        "dates_completed": dates_completed,
        "dates_skipped": dates_skipped,
        "dates_failed": dates_failed,
        "dates_partial": dates_partial,
        "failures_by_stage": failures_by_stage,
        "days_with_zombie_artifact_risk": days_with_zombie_artifact_risk,
        "invalid_day_complete_prevented": invalid_day_complete_prevented,
        "failure_by_date": {
            d.date: d.final_status
            for d in per_day
            if d.final_status not in (ps.STATUS_DAY_COMPLETE,)
        },
        "failure_by_reason": failures_by_reason,
        "pipeline_continued_after_failure": bool(pipeline_continued_after_failure),
        "global_guard_stopped": bool(global_guard_stopped),
        "interrupted": interrupted,
        "child_cleanup_records": list(child_cleanup_records),
        "total_cme_raw_downloads_attempted": sum(
            (
                1
                for d in per_day
                if d.download_outcome
                and d.download_outcome.get("status")
                not in (
                    None,
                    "DRY_RUN",
                    ps.STATUS_SKIPPED_CME_EXISTS,
                    ps.STATUS_DATABENTO_DOWNLOAD_BLOCKED_BY_FLAGS,
                    ps.STATUS_DATABENTO_DOWNLOAD_MISSING_API_KEY,
                    ps.STATUS_DATABENTO_DOWNLOAD_MISSING_MT5_WINDOW,
                )
            )
        ),
        "disk_free_min": min(disk_mins) if disk_mins else None,
        "disk_free_max": max(disk_maxs) if disk_maxs else None,
        "per_date_status": [{"date": d.date, "final_status": d.final_status} for d in per_day],
        "failures_by_reason": failures_by_reason,
        "next_recommended_action": "inspect per-date reports under reports_root and act on the listed warnings/errors"
        if dates_failed or dates_skipped
        else "data layer in good shape; collect more dates to lift Phase 2E out of small-sample",
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }
    summary_json_path = summary_txt_path = None
    if not config.dry_run:
        config.reports_root.mkdir(parents=True, exist_ok=True)
        summary_json_path = config.reports_root / f"multiday_pipeline_summary_{timestamp}.json"
        summary_txt_path = config.reports_root / f"multiday_pipeline_summary_{timestamp}.txt"
        jtmp = summary_json_path.with_suffix(".json.tmp")
        jtmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        os.replace(jtmp, summary_json_path)
        ttmp = summary_txt_path.with_suffix(".txt.tmp")
        ttmp.write_text(_render_summary_text(summary), encoding="utf-8")
        os.replace(ttmp, summary_txt_path)
    elif dry_run_report is not None:
        config.reports_root.mkdir(parents=True, exist_ok=True)
        dry_run_report_json_path = (
            config.reports_root / f"multiday_pipeline_dry_run_{timestamp}.json"
        )
        dry_run_report_txt_path = config.reports_root / f"multiday_pipeline_dry_run_{timestamp}.txt"
        jtmp = dry_run_report_json_path.with_suffix(".json.tmp")
        jtmp.write_text(
            json.dumps(dry_run_report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        os.replace(jtmp, dry_run_report_json_path)
        ttmp = dry_run_report_txt_path.with_suffix(".txt.tmp")
        ttmp.write_text(render_dry_run_text(dry_run_report), encoding="utf-8")
        os.replace(ttmp, dry_run_report_txt_path)
    return OrchestratorResult(
        config=config,
        per_day=per_day,
        summary=summary,
        summary_json_path=summary_json_path,
        summary_txt_path=summary_txt_path,
        per_day_json_paths=per_day_json_paths,
        interrupted=interrupted,
        child_cleanup_records=list(child_cleanup_records),
        dry_run_report=dry_run_report,
        dry_run_report_json_path=dry_run_report_json_path,
        dry_run_report_txt_path=dry_run_report_txt_path,
    )
