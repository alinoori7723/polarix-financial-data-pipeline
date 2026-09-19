from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from polarix.orchestration.run_metadata import RunMetadataError, resolve_run_metadata


def _write_pass_summary(path: Path, run_id: str, verified_offset_min: int = 120) -> None:
    doc = {
        "final_decision": "PASS",
        "started_at_utc": "2026-05-18T05:20:41.864435+00:00",
        "ended_at_utc": "2026-05-18T09:20:55.958522+00:00",
        "broker_metadata": {
            "account_company": "FundedNext Ltd",
            "account_server": "FundedNext-Server 3",
            "account_login_hash": "abc123",
            "terminal_company": "MetaQuotes Ltd.",
        },
        "clock_status": {"status": "CALIBRATION_COARSE_OK", "offset_ms": -10.0},
        "timestamp_semantics": {
            "status": "OFFSET_VERIFIED_FOR_SESSION",
            "verified_offset_min": verified_offset_min,
        },
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_fail_summary(path: Path) -> None:
    doc = {
        "final_decision": "FAIL",
        "timestamp_semantics": {
            "status": "OFFSET_VERIFIED_FOR_SESSION",
            "verified_offset_min": 999,
        },
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_logger_manifest(
    path: Path, verified_offset_min: int = 60, status: str = "OFFSET_VERIFIED_FOR_SESSION"
) -> None:
    doc = {
        "generated_at_utc": "2026-05-18T09:20:55Z",
        "symbols": ["SPX500", "NDX100"],
        "timestamp_semantics": {"status": status, "verified_offset_min": verified_offset_min},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_logger_health(
    path: Path, verified_offset_min: int = 60, status: str = "OFFSET_VERIFIED_FOR_SESSION"
) -> None:
    doc = {
        "account": {
            "company": "FundedNext Ltd",
            "server": "FundedNext-Server 3",
            "login_hash": "xyz789",
        },
        "broker": {
            "terminal_company": "MetaQuotes Ltd.",
            "terminal_name": "MetaTrader 5",
            "terminal_build": 5836,
        },
        "clock_health": {"status": "CALIBRATION_COARSE_OK", "offset_ms": -10.0},
        "symbols": ["SPX500", "NDX100"],
        "timestamp_semantics": {"status": status, "verified_offset_min": verified_offset_min},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_resolves_from_pass_live_run_summary_newest(tmp_path: Path) -> None:
    older = tmp_path / "live_run_20260517T000000Z_summary.json"
    newer = tmp_path / "live_run_20260518T052041Z_summary.json"
    _write_pass_summary(older, "20260517T000000Z", verified_offset_min=60)
    _write_pass_summary(newer, "20260518T052041Z", verified_offset_min=120)
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.source_kind == "live_run_summary"
    assert md.run_id == "20260518T052041Z"
    assert md.verified_offset_min == 120
    assert md.verified_offset_ms == 120 * 60 * 1000


def test_resolves_picks_pass_and_skips_fail(tmp_path: Path) -> None:
    fail = tmp_path / "live_run_20260519T000000Z_summary.json"
    passing = tmp_path / "live_run_20260518T000000Z_summary.json"
    _write_fail_summary(fail)
    _write_pass_summary(passing, "20260518T000000Z", verified_offset_min=60)
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.run_id == "20260518T000000Z"
    assert md.verified_offset_min == 60


def test_falls_back_to_logger_manifest_when_no_pass_summary(tmp_path: Path) -> None:
    fail = tmp_path / "live_run_20260519T000000Z_summary.json"
    _write_fail_summary(fail)
    _write_logger_manifest(tmp_path / "logger_manifest.json", verified_offset_min=60)
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.source_kind == "logger_manifest"
    assert md.verified_offset_min == 60


def test_falls_back_to_logger_health_when_no_summary_or_manifest(tmp_path: Path) -> None:
    _write_logger_health(tmp_path / "logger_health.json", verified_offset_min=120)
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.source_kind == "logger_health"
    assert md.verified_offset_min == 120
    assert md.broker_server == "FundedNext-Server 3"


def test_fails_closed_when_offset_missing(tmp_path: Path) -> None:
    health = tmp_path / "logger_health.json"
    _write_logger_health(health, verified_offset_min=60)
    doc = json.loads(health.read_text())
    doc["timestamp_semantics"]["verified_offset_min"] = None
    health.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RunMetadataError):
        resolve_run_metadata(reports_root=tmp_path)


def test_fails_closed_when_offset_not_int(tmp_path: Path) -> None:
    health = tmp_path / "logger_health.json"
    _write_logger_health(health, verified_offset_min=60)
    doc = json.loads(health.read_text())
    doc["timestamp_semantics"]["verified_offset_min"] = "180"
    health.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RunMetadataError):
        resolve_run_metadata(reports_root=tmp_path)


def test_fails_closed_when_status_pending(tmp_path: Path) -> None:
    _write_logger_health(
        tmp_path / "logger_health.json",
        verified_offset_min=60,
        status="TIMESTAMP_SEMANTICS_PENDING",
    )
    with pytest.raises(RunMetadataError):
        resolve_run_metadata(reports_root=tmp_path)


def test_fails_closed_when_status_unsafe(tmp_path: Path) -> None:
    _write_logger_health(
        tmp_path / "logger_health.json",
        verified_offset_min=60,
        status="CALIBRATION_UNSAFE_FOR_JOIN",
    )
    with pytest.raises(RunMetadataError):
        resolve_run_metadata(reports_root=tmp_path)


def test_accepts_utc_epoch_verified(tmp_path: Path) -> None:
    _write_logger_health(
        tmp_path / "logger_health.json", verified_offset_min=0, status="UTC_EPOCH_VERIFIED"
    )
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.timestamp_semantics_status == "UTC_EPOCH_VERIFIED"
    assert md.verified_offset_min == 0


def test_no_hardcoded_180_in_normalization_or_metadata() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "polarix"
    for fname in ("orchestration/run_metadata.py", "normalization/normalization.py"):
        text = (root / fname).read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 180:
                pytest.fail(
                    f"{fname}: numeric literal 180 found at line {node.lineno}; broker UTC offset must come from run metadata, never hardcoded"
                )
        if re.search("180\\s*\\*\\s*60\\s*\\*\\s*1000", text):
            pytest.fail(f"{fname}: '180 * 60 * 1000' pattern is forbidden")


def test_explicit_metadata_path_summary(tmp_path: Path) -> None:
    path = tmp_path / "live_run_20260518T100000Z_summary.json"
    _write_pass_summary(path, "20260518T100000Z", verified_offset_min=60)
    _write_logger_manifest(tmp_path / "logger_manifest.json", verified_offset_min=120)
    md = resolve_run_metadata(reports_root=tmp_path, metadata_path=path)
    assert md.source_kind == "live_run_summary"
    assert md.verified_offset_min == 60


def test_explicit_metadata_path_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "logger_manifest.json"
    _write_logger_manifest(manifest, verified_offset_min=60)
    md = resolve_run_metadata(reports_root=tmp_path, metadata_path=manifest)
    assert md.source_kind == "logger_manifest"
    assert md.verified_offset_min == 60


def test_explicit_metadata_path_health(tmp_path: Path) -> None:
    health = tmp_path / "logger_health.json"
    _write_logger_health(health, verified_offset_min=0, status="UTC_EPOCH_VERIFIED")
    md = resolve_run_metadata(reports_root=tmp_path, metadata_path=health)
    assert md.source_kind == "logger_health"
    assert md.verified_offset_min == 0


def test_explicit_run_id_picks_specific_summary(tmp_path: Path) -> None:
    a = tmp_path / "live_run_20260518T000000Z_summary.json"
    b = tmp_path / "live_run_20260518T100000Z_summary.json"
    _write_pass_summary(a, "20260518T000000Z", verified_offset_min=60)
    _write_pass_summary(b, "20260518T100000Z", verified_offset_min=120)
    md = resolve_run_metadata(reports_root=tmp_path, run_id="20260518T000000Z")
    assert md.run_id == "20260518T000000Z"
    assert md.verified_offset_min == 60


def test_missing_reports_root_fails(tmp_path: Path) -> None:
    with pytest.raises(RunMetadataError):
        resolve_run_metadata(reports_root=tmp_path / "does_not_exist")


def test_companion_enrichment_fills_broker_metadata(tmp_path: Path) -> None:
    _write_logger_manifest(tmp_path / "logger_manifest.json", verified_offset_min=60)
    _write_logger_health(tmp_path / "logger_health.json", verified_offset_min=60)
    md = resolve_run_metadata(reports_root=tmp_path)
    assert md.source_kind == "logger_manifest"
    assert md.broker_server == "FundedNext-Server 3"
    assert md.broker_company == "FundedNext Ltd"
