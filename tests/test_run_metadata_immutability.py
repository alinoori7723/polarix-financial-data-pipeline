from __future__ import annotations

import json
from pathlib import Path

import pytest

from polarix.orchestration.run_metadata import (
    ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA,
    ERROR_UNVERIFIED_RUN_METADATA,
    RUN_SCOPED_DIRNAME,
    RUN_SCOPED_HEALTH,
    RUN_SCOPED_MANIFEST,
    RUN_SCOPED_SUMMARY,
    RunMetadataError,
    list_run_scoped_runs,
    resolve_run_metadata,
    write_run_scoped_metadata,
)


def _verified_manifest_doc(verified_offset_min: int = 120) -> dict:
    return {
        "compression": "zstd",
        "data_root": ".polarix/data",
        "dataset": "mt5_ticks",
        "generated_at_utc": "2026-05-18T19:30:00+00:00",
        "raw_dataset_dir": ".polarix/data\\raw\\mt5_ticks",
        "symbols": ["SPX500", "NDX100"],
        "timestamp_semantics": {
            "status": "OFFSET_VERIFIED_FOR_SESSION",
            "verified_offset_min": verified_offset_min,
        },
    }


def _unverified_manifest_doc() -> dict:
    return {
        "compression": "zstd",
        "data_root": ".polarix/data",
        "dataset": "mt5_ticks",
        "generated_at_utc": "2026-05-18T19:30:00+00:00",
        "raw_dataset_dir": ".polarix/data\\raw\\mt5_ticks",
        "symbols": ["SPX500", "NDX100"],
        "timestamp_semantics": {
            "status": "TIMESTAMP_SEMANTICS_PENDING",
            "verified_offset_min": None,
        },
    }


def _seed_run_scoped(
    reports_root: Path,
    run_id: str,
    *,
    manifest_doc: dict | None = None,
    health_doc: dict | None = None,
    summary_doc: dict | None = None,
) -> Path:
    run_dir = reports_root / RUN_SCOPED_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest_doc is not None:
        (run_dir / RUN_SCOPED_MANIFEST).write_text(json.dumps(manifest_doc), encoding="utf-8")
    if health_doc is not None:
        (run_dir / RUN_SCOPED_HEALTH).write_text(json.dumps(health_doc), encoding="utf-8")
    if summary_doc is not None:
        (run_dir / RUN_SCOPED_SUMMARY).write_text(json.dumps(summary_doc), encoding="utf-8")
    return run_dir


def test_write_run_scoped_metadata_creates_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "logger_manifest.json"
    health = tmp_path / "logger_health.json"
    manifest.write_text(json.dumps(_verified_manifest_doc()), encoding="utf-8")
    health.write_text(
        json.dumps({"account": {"server": "X"}, "broker": {}, "clock_health": {}}), encoding="utf-8"
    )
    run_dir = write_run_scoped_metadata(
        reports_root=tmp_path,
        run_id="live_run_20260518T193000Z",
        manifest_src=manifest,
        health_src=health,
        summary_doc={"final_decision": "PASS"},
    )
    assert run_dir.exists()
    assert (run_dir / RUN_SCOPED_MANIFEST).exists()
    assert (run_dir / RUN_SCOPED_HEALTH).exists()
    assert (run_dir / RUN_SCOPED_SUMMARY).exists()
    runs = list_run_scoped_runs(tmp_path)
    assert runs and runs[0][0] == "live_run_20260518T193000Z"


def test_write_run_scoped_metadata_tolerates_missing_sources(tmp_path: Path) -> None:
    run_dir = write_run_scoped_metadata(
        reports_root=tmp_path,
        run_id="live_run_20260518T193000Z",
        manifest_src=None,
        health_src=None,
        summary_doc={"final_decision": "PARTIAL"},
    )
    assert run_dir.exists()
    assert (run_dir / RUN_SCOPED_SUMMARY).exists()


def test_supervisor_summary_contains_run_metadata_fields(cfg, monkeypatch):
    from tests.test_supervisor import _build_supervisor, _patch_preflight_ok, _write_logger_health

    _patch_preflight_ok(monkeypatch)
    manifest = cfg.reports_root / "logger_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(_verified_manifest_doc(60)), encoding="utf-8")
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup, _holder, _clock, _sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    sup.run()
    summary_path = cfg.reports_root / "live_run_TESTSTAMP_summary.json"
    assert summary_path.exists()
    doc = json.loads(summary_path.read_text(encoding="utf-8"))
    assert doc["run_id"] == "live_run_TESTSTAMP"
    assert "run_metadata_dir" in doc
    assert "manifest_path" in doc
    assert "health_path" in doc
    assert "console_log_path" in doc
    assert "supervisor_health_path" in doc
    assert doc["metadata_verified_for_normalization"] is True
    assert doc["verified_offset_min"] == 60
    assert doc["timestamp_semantics_status"] == "OFFSET_VERIFIED_FOR_SESSION"
    run_dir = cfg.reports_root / RUN_SCOPED_DIRNAME / "live_run_TESTSTAMP"
    assert run_dir.exists()
    assert (run_dir / RUN_SCOPED_MANIFEST).exists()


def test_supervisor_summary_marks_unverified_when_manifest_is_partial(cfg, monkeypatch):
    from tests.test_supervisor import _build_supervisor, _patch_preflight_ok, _write_logger_health

    _patch_preflight_ok(monkeypatch)
    manifest = cfg.reports_root / "logger_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(_unverified_manifest_doc()), encoding="utf-8")
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup, _holder, _clock, _sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    sup.run()
    doc = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert doc["metadata_verified_for_normalization"] is False
    assert doc["timestamp_semantics_status"] == "TIMESTAMP_SEMANTICS_PENDING"
    assert doc["verified_offset_min"] is None


def test_run_id_resolution_survives_global_manifest_poisoning(tmp_path: Path) -> None:
    _seed_run_scoped(
        tmp_path, "live_run_20260518T053000Z", manifest_doc=_verified_manifest_doc(180)
    )
    poisoned = tmp_path / "logger_manifest.json"
    poisoned.write_text(json.dumps(_unverified_manifest_doc()), encoding="utf-8")
    md = resolve_run_metadata(
        reports_root=tmp_path,
        run_id="live_run_20260518T053000Z",
        allow_latest_metadata=False,
        strict_run_metadata=True,
    )
    assert md.verified_offset_min == 180
    assert md.timestamp_semantics_status == "OFFSET_VERIFIED_FOR_SESSION"
    assert md.source_type == "run_scoped"
    assert md.run_id == "live_run_20260518T053000Z"


def test_run_scoped_unverified_run_fails_closed(tmp_path: Path) -> None:
    _seed_run_scoped(tmp_path, "live_run_20260518T193000Z", manifest_doc=_unverified_manifest_doc())
    with pytest.raises(RunMetadataError) as ei:
        resolve_run_metadata(
            reports_root=tmp_path, run_id="live_run_20260518T193000Z", allow_latest_metadata=False
        )
    assert ei.value.error_tag == ERROR_UNVERIFIED_RUN_METADATA


def test_run_scoped_wrong_status_fails_closed(tmp_path: Path) -> None:
    bad = _verified_manifest_doc(180)
    bad["timestamp_semantics"]["status"] = "CALIBRATION_UNSAFE_FOR_JOIN"
    _seed_run_scoped(tmp_path, "live_run_20260518T193000Z", manifest_doc=bad)
    with pytest.raises(RunMetadataError) as ei:
        resolve_run_metadata(
            reports_root=tmp_path, run_id="live_run_20260518T193000Z", allow_latest_metadata=False
        )
    assert ei.value.error_tag == ERROR_UNVERIFIED_RUN_METADATA


def test_multiple_verified_runs_same_date_ambiguous(tmp_path: Path) -> None:
    _seed_run_scoped(
        tmp_path, "live_run_20260518T053000Z", manifest_doc=_verified_manifest_doc(180)
    )
    _seed_run_scoped(
        tmp_path, "live_run_20260518T193000Z", manifest_doc=_verified_manifest_doc(120)
    )
    with pytest.raises(RunMetadataError) as ei:
        resolve_run_metadata(reports_root=tmp_path, date="2026-05-18", allow_latest_metadata=False)
    assert ei.value.error_tag == ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA
    assert "live_run_20260518T053000Z" in str(ei.value)
    assert "live_run_20260518T193000Z" in str(ei.value)


def test_exactly_one_verified_run_used_without_run_id(tmp_path: Path) -> None:
    _seed_run_scoped(
        tmp_path, "live_run_20260518T053000Z", manifest_doc=_verified_manifest_doc(180)
    )
    _seed_run_scoped(tmp_path, "live_run_20260518T193000Z", manifest_doc=_unverified_manifest_doc())
    md = resolve_run_metadata(reports_root=tmp_path, date="2026-05-18", allow_latest_metadata=False)
    assert md.run_id == "live_run_20260518T053000Z"
    assert md.verified_offset_min == 180
    assert md.source_type == "run_scoped"


def test_explicit_metadata_path_to_run_scoped_manifest(tmp_path: Path) -> None:
    run_dir = _seed_run_scoped(
        tmp_path, "live_run_20260518T053000Z", manifest_doc=_verified_manifest_doc(180)
    )
    md = resolve_run_metadata(reports_root=tmp_path, metadata_path=run_dir / RUN_SCOPED_MANIFEST)
    assert md.verified_offset_min == 180
    assert md.source_type == "run_scoped"


def test_explicit_metadata_path_outside_run_scoped_is_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "custom_manifest.json"
    p.write_text(json.dumps(_verified_manifest_doc(60)), encoding="utf-8")
    md = resolve_run_metadata(reports_root=tmp_path, metadata_path=p)
    assert md.verified_offset_min == 60
    assert md.source_type == "explicit_path"


def test_no_run_scoped_and_no_allow_latest_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "logger_manifest.json").write_text(
        json.dumps(_verified_manifest_doc(180)), encoding="utf-8"
    )
    with pytest.raises(RunMetadataError) as ei:
        resolve_run_metadata(reports_root=tmp_path, allow_latest_metadata=False)
    assert ei.value.error_tag == ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA


def test_no_run_scoped_and_allow_latest_uses_global_manifest(tmp_path: Path) -> None:
    (tmp_path / "logger_manifest.json").write_text(
        json.dumps(_verified_manifest_doc(60)), encoding="utf-8"
    )
    md = resolve_run_metadata(reports_root=tmp_path, allow_latest_metadata=True)
    assert md.verified_offset_min == 60
    assert md.source_type == "legacy_latest"


def test_run_scoped_present_must_not_be_silently_overridden_by_global(tmp_path: Path) -> None:
    _seed_run_scoped(
        tmp_path, "live_run_20260518T053000Z", manifest_doc=_verified_manifest_doc(180)
    )
    (tmp_path / "logger_manifest.json").write_text(
        json.dumps(_verified_manifest_doc(60)), encoding="utf-8"
    )
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.verified_offset_min == 180
    assert md.source_type == "run_scoped"


def test_partial_run_metadata_not_accepted_even_with_run_id(tmp_path: Path) -> None:
    _seed_run_scoped(
        tmp_path,
        "live_run_20260518T193000Z",
        manifest_doc=_unverified_manifest_doc(),
        summary_doc={"final_decision": "PARTIAL"},
    )
    with pytest.raises(RunMetadataError) as ei:
        resolve_run_metadata(
            reports_root=tmp_path, run_id="live_run_20260518T193000Z", allow_latest_metadata=False
        )
    assert ei.value.error_tag == ERROR_UNVERIFIED_RUN_METADATA


def test_no_hardcoded_180_in_run_metadata_or_supervisor() -> None:
    import ast

    root = Path(__file__).resolve().parents[1]
    suspect = (
        root / "src" / "polarix" / "orchestration" / "run_metadata.py",
        root / "src" / "polarix" / "orchestration" / "supervisor.py",
    )
    for path in suspect:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 180:
                pytest.fail(
                    f"{path}:{node.lineno}: numeric literal 180 must not appear (broker UTC offset must come from per-run metadata, never hardcoded)"
                )


def test_no_trading_functions_in_phase_2f1_files() -> None:
    root = Path(__file__).resolve().parents[1]
    files = (
        root / "src" / "polarix" / "orchestration" / "run_metadata.py",
        root / "src" / "polarix" / "orchestration" / "supervisor.py",
        root / "scripts" / "normalize_telemetry.py",
    )
    forbidden = (
        "order_send",
        "order_check",
        "order_calc_margin",
        "order_calc_profit",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


@pytest.fixture()
def cfg(tmp_path):
    from polarix.common.config import load_config

    base = load_config((Path(__file__).resolve().parents[1] / "config" / "logger.example.json"))
    redirected = type(base)(
        environment=base.environment,
        broker_profile=base.broker_profile,
        current_observed_broker=base.current_observed_broker,
        data_root=tmp_path / "data",
        reports_root=tmp_path / "reports",
        logs_root=tmp_path / "logs",
        symbols=base.symbols,
        plausible_broker_utc_offsets_minutes=base.plausible_broker_utc_offsets_minutes,
        max_host_clock_offset_ms=base.max_host_clock_offset_ms,
        max_future_jitter_ms=base.max_future_jitter_ms,
        max_live_tick_age_ms=base.max_live_tick_age_ms,
        required_fresh_ticks_for_offset=base.required_fresh_ticks_for_offset,
        flush_max_rows=base.flush_max_rows,
        flush_max_seconds=base.flush_max_seconds,
        closed_market_idle_backoff_seconds_max=base.closed_market_idle_backoff_seconds_max,
        parquet_compression=base.parquet_compression,
        price_scale_default=base.price_scale_default,
        no_trading_functions_allowed=base.no_trading_functions_allowed,
        fail_if_terminal_trade_allowed=base.fail_if_terminal_trade_allowed,
        max_failed_attempts_before_human_review=base.max_failed_attempts_before_human_review,
        raw=base.raw,
    )
    redirected.data_root.mkdir(parents=True, exist_ok=True)
    redirected.reports_root.mkdir(parents=True, exist_ok=True)
    redirected.logs_root.mkdir(parents=True, exist_ok=True)
    redirected.raw_dataset_dir.mkdir(parents=True, exist_ok=True)
    return redirected
