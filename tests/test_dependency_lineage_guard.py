from __future__ import annotations

import ast
import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.ingestion import cme_downloader as cme_dl
from polarix.orchestration import pipeline_status as ps
from polarix.orchestration import stage_dependencies as sd
from polarix.orchestration.pipeline_orchestrator import (
    OrchestratorConfig,
    process_one_date,
    run_multiday,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed_mt5(data_root: Path, date: str) -> None:
    t0 = int(_dt.datetime.fromisoformat(f"{date}T05:00:00+00:00").timestamp() * 1000)
    for sym in ("SPX500", "NDX100"):
        path = (
            data_root
            / "normalized"
            / "mt5_ticks"
            / f"symbol={sym}"
            / f"date={date}"
            / "part-0001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "symbol": [sym] * 2,
                    "time_msc_utc_ms": pa.array([t0, t0 + 10000], type=pa.int64()),
                }
            ),
            path,
            compression="zstd",
        )


def _write_metadata_sidecar(
    parquet_path: Path,
    *,
    completeness: str,
    records_downloaded: int = 1000000,
    max_download_records: int = 1000000,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        pq.write_table(pa.table({"x": [1]}), parquet_path)
    payload = {
        "requested_start_utc": "2026-05-18T05:00:00Z",
        "requested_end_utc": "2026-05-18T06:00:00Z",
        "dataset": "GLBX.MDP3",
        "schema": "mbp-1",
        "symbols": ["ES.c.0", "NQ.c.0"],
        "stype_in": "continuous",
        "output_path": str(parquet_path),
        "file_size_bytes": parquet_path.stat().st_size,
        "max_download_records": max_download_records,
        "records_downloaded": records_downloaded,
        "data_completeness_status": completeness,
        "truncation_warning": completeness == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT,
        "created_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "databento_sdk_version": "fake-test",
        "api_call_parameters": {"limit": max_download_records},
        "physical_limit_applied": True,
        "physical_limit_type": "record_limit",
        "physical_limit_value": max_download_records,
    }
    sidecar = cme_dl.metadata_path_for(parquet_path)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def _seed_cme_raw(data_root: Path, date: str, *, completeness: str | None) -> Path:
    raw = data_root / "raw" / "cme_sample" / f"date={date}" / "esnq.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), raw)
    if completeness is not None:
        _write_metadata_sidecar(raw, completeness=completeness)
    return raw


def _seed_cme_normalized(data_root: Path, date: str) -> None:
    root = data_root / "normalized" / "cme_reference" / "reference_trades"
    for sym in ("ES", "NQ"):
        path = root / f"symbol={sym}" / f"date={date}" / "part-0001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"x": [1]}), path)


def _seed_report(reports_root: Path, name: str, date: str, decision: str = "PASS") -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / f"{name}_{date}.json").write_text(
        json.dumps({"quality_decision": decision, "date": date}), encoding="utf-8"
    )


def _seed_gold_features(data_root: Path, date: str) -> None:
    root = data_root / "features" / "bar_features"
    for pair in ("ES_SPX500", "NQ_NDX100"):
        p = root / f"symbol_pair={pair}" / f"date={date}" / "bucket=5s" / "part-0001.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"x": [1]}), p)


def _make_config(tmp_path: Path, dates: list[str], **overrides) -> OrchestratorConfig:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    cfg = OrchestratorConfig(
        dates=dates,
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        python_exe=Path("python"),
        dry_run=False,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _runner_failing(stage_scripts: set[str]):
    invoked: list[list[str]] = []

    def runner(argv, timeout, cwd):
        invoked.append(list(argv))
        for script in stage_scripts:
            if any((script in a for a in argv)):
                return subprocess.CompletedProcess(
                    argv, returncode=1, stdout="", stderr=f"[{script}] simulated failure"
                )
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    return (runner, invoked)


def _runner_all_ok():
    invoked: list[list[str]] = []

    def runner(argv, timeout, cwd):
        invoked.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    return (runner, invoked)


def _status_of(day, stage_name: str) -> Optional[str]:
    for s in day.stages:
        if s.get("name") == stage_name:
            return s.get("status")
    return None


def _record_of(day, stage_name: str) -> Optional[dict]:
    for s in day.stages:
        if s.get("name") == stage_name:
            return s
    return None


def test_1_cme_ingest_failure_propagates_to_all_downstream(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, invoked = _runner_failing({"ingest_cme_reference_sample.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert _status_of(day, "CME_INGEST") == ps.STAGE_FAIL
    for downstream in (
        "CME_QUALITY",
        "ALIGNMENT_QUALITY",
        "BAR_ALIGNMENT",
        "BUILD_BAR_FEATURES",
        "BAR_FEATURE_QUALITY",
        "FEATURE_EDA",
    ):
        assert _status_of(day, downstream) == ps.STAGE_SKIPPED_UPSTREAM_FAILED, (
            f"{downstream} must be skipped after CME_INGEST failure"
        )
    invoked_scripts = {Path(a).name for argv in invoked for a in argv if a.endswith(".py")}
    assert "ingest_cme_reference_sample.py" in invoked_scripts
    for script in (
        "cme_reference_quality_report.py",
        "alignment_quality_report.py",
        "bar_alignment_quality_report.py",
        "build_bar_features.py",
        "bar_feature_quality_report.py",
        "feature_eda_report.py",
    ):
        assert script not in invoked_scripts, f"{script} must not be invoked"


def test_2_build_bar_features_failure_skips_quality_and_eda(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, invoked = _runner_failing({"build_bar_features.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert _status_of(day, "BUILD_BAR_FEATURES") == ps.STAGE_FAIL
    assert _status_of(day, "BAR_FEATURE_QUALITY") == ps.STAGE_SKIPPED_UPSTREAM_FAILED
    assert _status_of(day, "FEATURE_EDA") == ps.STAGE_SKIPPED_UPSTREAM_FAILED


def test_3_no_downstream_subprocess_after_required_upstream_failure(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=True)
    runner, invoked = _runner_failing({"ingest_cme_reference_sample.py"})
    process_one_date(cfg, "2026-05-18", runner=runner)
    invoked_scripts = {Path(a).name for argv in invoked for a in argv if a.endswith(".py")}
    for script in (
        "cme_reference_quality_report.py",
        "alignment_quality_report.py",
        "bar_alignment_quality_report.py",
        "build_bar_features.py",
        "bar_feature_quality_report.py",
        "feature_eda_report.py",
    ):
        assert script not in invoked_scripts


def test_4_day_complete_impossible_when_required_stage_failed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, _ = _runner_failing({"alignment_quality_report.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_FAILED
    assert day.final_status != ps.STATUS_DAY_COMPLETE


def test_5_trusted_cme_normalized_satisfies_dependency_no_rebuild(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)
    runner, invoked = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_COMPLETE
    assert invoked == [], "no subprocess should be invoked when every artifact is trusted-skipped"
    for s in day.stages:
        assert s["status"] == ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED


def test_6_untrusted_artifact_does_not_satisfy_dependency(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    cfg = _make_config(
        tmp_path, ["2026-05-18"], allow_truncated_cme_sample=False, rebuild_existing=False
    )
    runner, invoked = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_TRUNCATED_SAMPLE_BLOCKED


def test_7_downstream_quality_does_not_run_on_untrusted_stale_artifact(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=True)
    runner, invoked = _runner_failing({"ingest_cme_reference_sample.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    invoked_scripts = {Path(a).name for argv in invoked for a in argv if a.endswith(".py")}
    assert "cme_reference_quality_report.py" not in invoked_scripts
    assert _status_of(day, "CME_QUALITY") == ps.STAGE_SKIPPED_UPSTREAM_FAILED
    rec = _record_of(day, "CME_QUALITY")
    assert rec["upstream_failed_stage"] == "CME_INGEST"


def test_8_artifact_exists_but_untrusted_is_reported(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)
    runner, _invoked = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    align = _record_of(day, "ALIGNMENT_QUALITY")
    assert align is not None
    assert align["status"] == ps.STAGE_SKIPPED_ALREADY_EXISTS_UNTRUSTED
    assert align["artifact_trust_status"] == ps.ARTIFACT_EXISTS_BUT_UNTRUSTED
    untrusted_raw = sd.CmeRawTrust(
        raw_present=True,
        completeness_status=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT,
        trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
        reason="TRUNCATED, override not set",
    )
    spec = next((s for s in sd.DEFAULT_STAGE_GRAPH if s.name == "CME_INGEST"))
    verdict = sd.evaluate_stage_trust(
        spec, artifact_present=True, cme_raw_trust=untrusted_raw, upstream_trust={}
    )
    assert verdict.artifact_trust_status == ps.ARTIFACT_EXISTS_BUT_UNTRUSTED


def test_9_required_stage_failure_yields_day_failed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, _ = _runner_failing({"build_bar_features.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_FAILED


def test_10_all_trusted_skips_yield_day_complete(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, invoked = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_COMPLETE
    assert invoked == []


def test_11_optional_stage_failure_yields_day_partial(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, _ = _runner_failing({"feature_eda_report.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_PARTIAL
    assert _status_of(day, "FEATURE_EDA") == ps.STAGE_FAIL
    for req in (
        "CME_INGEST",
        "CME_QUALITY",
        "ALIGNMENT_QUALITY",
        "BAR_ALIGNMENT",
        "BUILD_BAR_FEATURES",
    ):
        assert _status_of(day, req) == ps.STAGE_OK


def test_12_trusted_artifact_skipped_without_error_in_default_mode(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)
    invoked: list[list[str]] = []

    def runner(argv, timeout, cwd):
        invoked.append(list(argv))
        if "ingest_cme_reference_sample.py" in argv[2]:
            if "--force" not in argv:
                return subprocess.CompletedProcess(
                    argv,
                    returncode=3,
                    stdout="",
                    stderr="reference_trades already exist; pass --force",
                )
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert _status_of(day, "CME_INGEST") == ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED
    invoked_scripts = {Path(a).name for argv in invoked for a in argv if a.endswith(".py")}
    assert "ingest_cme_reference_sample.py" not in invoked_scripts


def test_13_rebuild_existing_passes_force_to_force_capable_scripts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=True)
    runner, invoked = _runner_all_ok()
    process_one_date(cfg, "2026-05-18", runner=runner)

    def _argv_for(script: str) -> Optional[list[str]]:
        for argv in invoked:
            if any((script in a for a in argv)):
                return argv
        return None

    for force_capable in (
        "ingest_cme_reference_sample.py",
        "build_bar_features.py",
        "feature_eda_report.py",
    ):
        argv = _argv_for(force_capable)
        assert argv is not None, f"{force_capable} should have been invoked"
        assert "--force" in argv, f"{force_capable} should have --force when rebuilding"
    for not_force_capable in ("cme_reference_quality_report.py", "bar_feature_quality_report.py"):
        argv = _argv_for(not_force_capable)
        if argv is not None:
            assert "--force" not in argv


def test_14_output_exists_failure_not_misreported_as_success(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=3, stdout="", stderr="output exists")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_COMPLETE


def test_15_trusted_skip_classification_replaces_output_exists_fail(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)
    runner, _ = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    rec = _record_of(day, "CME_INGEST")
    assert rec["status"] == ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED
    assert rec["force_passed"] is False
    assert rec["artifact_present"] is True
    assert rec["artifact_trust_status"] in (
        ps.ARTIFACT_TRUSTED,
        ps.ARTIFACT_TRUSTED_EXPLORATORY,
        ps.ARTIFACT_TRUSTED_LEGACY_NO_SIDECAR,
    )


def test_16_per_day_report_includes_upstream_failed_stage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    runner, _ = _runner_failing({"ingest_cme_reference_sample.py"})
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    for downstream in ("CME_QUALITY", "ALIGNMENT_QUALITY"):
        rec = _record_of(day, downstream)
        assert rec is not None
        assert rec["upstream_failed_stage"] == (
            "CME_INGEST" if downstream == "CME_QUALITY" else "CME_QUALITY"
        )
        assert rec["dependency_status"] == "UNSATISFIED"


def test_17_per_day_report_includes_artifact_trust_status(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=False)
    runner, _ = _runner_all_ok()
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    rec = _record_of(day, "CME_INGEST")
    assert "artifact_trust_status" in rec
    assert rec["artifact_trust_status"] in (
        ps.ARTIFACT_TRUSTED,
        ps.ARTIFACT_TRUSTED_EXPLORATORY,
        ps.ARTIFACT_TRUSTED_LEGACY_NO_SIDECAR,
    )


def test_18_master_summary_records_failures_by_stage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=False)
    runner, _ = _runner_failing({"build_bar_features.py"})
    result = run_multiday(cfg, runner=runner)
    assert "failures_by_stage" in result.summary
    assert result.summary["failures_by_stage"].get("BUILD_BAR_FEATURES") == 1


def test_19_master_summary_records_dates_failed_not_completed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=False)
    runner, _ = _runner_failing({"ingest_cme_reference_sample.py"})
    result = run_multiday(cfg, runner=runner)
    assert "2026-05-18" in result.summary["dates_failed"]
    assert "2026-05-18" not in result.summary["dates_completed"]


def test_20_master_summary_invalid_day_complete_prevented_flag(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=True)
    runner, _ = _runner_failing({"ingest_cme_reference_sample.py"})
    result = run_multiday(cfg, runner=runner)
    assert result.summary["invalid_day_complete_prevented"] is True
    assert "2026-05-18" in result.summary["days_with_zombie_artifact_risk"]


def test_21_regression_real_run_shape_does_not_yield_day_complete(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    _seed_cme_raw(data, "2026-05-18", completeness=cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT)
    _seed_cme_normalized(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "cme_reference_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "alignment_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_alignment_quality", "2026-05-18")
    _seed_gold_features(data, "2026-05-18")
    _seed_report(tmp_path / "reports", "bar_feature_quality", "2026-05-18")
    _seed_report(tmp_path / "reports", "feature_eda", "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], rebuild_existing=True)
    runner, invoked = _runner_failing(
        {"ingest_cme_reference_sample.py", "build_bar_features.py", "feature_eda_report.py"}
    )
    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status != ps.STATUS_DAY_COMPLETE
    assert day.final_status == ps.STATUS_DAY_FAILED
    assert _status_of(day, "CME_QUALITY") == ps.STAGE_SKIPPED_UPSTREAM_FAILED
    invoked_scripts = {Path(a).name for argv in invoked for a in argv if a.endswith(".py")}
    assert "cme_reference_quality_report.py" not in invoked_scripts
    assert _status_of(day, "BAR_FEATURE_QUALITY") == ps.STAGE_SKIPPED_UPSTREAM_FAILED
    assert any(("Phase 2G.3 prevented" in w for w in day.warnings))


PHASE_2G3_FILES = (
    "src/polarix/orchestration/stage_dependencies.py",
    "src/polarix/orchestration/pipeline_orchestrator.py",
    "src/polarix/orchestration/pipeline_status.py",
    "scripts/run_multiday_pipeline.py",
)


def _texts() -> dict[str, str]:
    return {p: (REPO_ROOT / p).read_text(encoding="utf-8") for p in PHASE_2G3_FILES}


def test_22_no_trading_functions_phase_2g3() -> None:
    forbidden = (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for path, text in _texts().items():
        for tok in forbidden:
            assert tok not in text, f"{path}: forbidden {tok!r}"


def test_23_no_mt5_sdk_imports_phase_2g3() -> None:
    forbidden = ("import MetaTrader5", "from MetaTrader5", "mt5.order_send", "mt5.copy_ticks_from")
    for path, text in _texts().items():
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_24_no_model_training_imports_phase_2g3() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "model.fit(",
        "train_test_split",
    )
    for path, text in _texts().items():
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_25_no_labels_or_targets_phase_2g3() -> None:
    forbidden = ("labels_y", "target_y", "y_true", "y_pred", "make_labels", "build_targets")
    for path, text in _texts().items():
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_26_no_cumulative_cvd_phase_2g3() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _texts().items():
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_27_no_automatic_raw_data_deletion_phase_2g3() -> None:
    forbidden_fn = {"unlink", "rmtree", "remove"}
    for path in PHASE_2G3_FILES:
        try:
            tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                if name in forbidden_fn:
                    pytest.fail(f"{path}:{node.lineno}: deletion call {name!r}")


def test_28_no_weakening_of_cost_or_physical_guard_phase_2g3(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["2026-05-18"])
    assert config.cost_estimate_required
    assert config.require_physical_download_limit
    assert not config.allow_unestimated_download
    assert not config.allow_download_without_physical_limit
    assert not config.acknowledge_cost_risk


def test_29_no_weakening_of_run_scoped_metadata_phase_2g3() -> None:
    text = (REPO_ROOT / "src" / "polarix" / "orchestration" / "pipeline_orchestrator.py").read_text(
        encoding="utf-8"
    )
    assert "from polarix.orchestration.run_selection import" in text


def test_30_no_weakening_of_cross_midnight_assignment_phase_2g3() -> None:
    from polarix.orchestration.databento_download_plan import assert_window_is_single_day

    assert callable(assert_window_is_single_day)


def test_classify_day_status_all_ok() -> None:
    status = {s.name: ps.STAGE_OK for s in sd.DEFAULT_STAGE_GRAPH}
    assert (
        sd.classify_day_status(stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=status)
        == ps.STATUS_DAY_COMPLETE
    )


def test_classify_day_status_required_fail() -> None:
    status = {s.name: ps.STAGE_OK for s in sd.DEFAULT_STAGE_GRAPH}
    status["BUILD_BAR_FEATURES"] = ps.STAGE_FAIL
    assert (
        sd.classify_day_status(stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=status)
        == ps.STATUS_DAY_FAILED
    )


def test_classify_day_status_optional_fail() -> None:
    status = {s.name: ps.STAGE_OK for s in sd.DEFAULT_STAGE_GRAPH}
    status["FEATURE_EDA"] = ps.STAGE_FAIL
    assert (
        sd.classify_day_status(stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=status)
        == ps.STATUS_DAY_PARTIAL
    )


def test_classify_day_status_trusted_skip_counts_as_satisfied() -> None:
    status = {s.name: ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED for s in sd.DEFAULT_STAGE_GRAPH}
    assert (
        sd.classify_day_status(stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=status)
        == ps.STATUS_DAY_COMPLETE
    )


def test_classify_day_status_untrusted_skip_required_is_failed() -> None:
    status = {s.name: ps.STAGE_OK for s in sd.DEFAULT_STAGE_GRAPH}
    status["CME_INGEST"] = ps.STAGE_SKIPPED_ALREADY_EXISTS_UNTRUSTED
    assert (
        sd.classify_day_status(stage_graph=sd.DEFAULT_STAGE_GRAPH, stage_status=status)
        == ps.STATUS_DAY_FAILED
    )
