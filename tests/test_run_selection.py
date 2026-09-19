from __future__ import annotations

import json
from pathlib import Path

import pytest

from polarix.orchestration.run_selection import (
    POLICY_EXPLICIT,
    POLICY_FAIL_ON_MULTIPLE,
    POLICY_LATEST_VERIFIED,
    POLICY_LONGEST_VERIFIED,
    REASON_AMBIGUOUS,
    REASON_EXPLICIT_MAP_MISSING_ENTRY,
    REASON_EXPLICIT_MAP_RUN_NOT_FOUND,
    REASON_EXPLICIT_MAP_RUN_NOT_VERIFIED,
    REASON_NO_RUN_SCOPED_DIR,
    REASON_NO_VERIFIED_RUN_METADATA,
    WARNING_CROSS_MIDNIGHT,
    discover_run_candidates,
    parse_run_id_map,
    select_run_for_date,
)


def _seed_run(
    reports_root: Path,
    run_id: str,
    *,
    started_at_utc: str | None,
    ended_at_utc: str | None,
    metadata_verified_for_normalization: bool = True,
    verified_offset_min: int | None = 180,
    timestamp_semantics_status: str | None = "OFFSET_VERIFIED_FOR_SESSION",
    final_decision: str | None = "PASS",
) -> Path:
    run_dir = reports_root / "logger_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "metadata_verified_for_normalization": metadata_verified_for_normalization,
        "verified_offset_min": verified_offset_min,
        "timestamp_semantics_status": timestamp_semantics_status,
        "final_decision": final_decision,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_discover_run_candidates_lists_runs(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:00:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T22:00:00+00:00",
    )
    candidates = discover_run_candidates(tmp_path)
    assert len(candidates) == 2
    assert all((c.run_assignment_date == "2026-05-18" for c in candidates))


def test_cross_midnight_run_assigned_only_to_start_date(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T220000Z",
        started_at_utc="2026-05-18T22:00:00+00:00",
        ended_at_utc="2026-05-19T03:00:00+00:00",
    )
    candidates = discover_run_candidates(tmp_path)
    [c] = candidates
    assert c.run_assignment_date == "2026-05-18"
    assert c.ended_date == "2026-05-19"
    assert c.crosses_midnight is True


def test_no_runs_returns_no_run_scoped_dir(tmp_path: Path) -> None:
    sel = select_run_for_date("2026-05-18", [], policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_NO_RUN_SCOPED_DIR


def test_runs_for_other_date_excluded(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260520T050000Z",
        started_at_utc="2026-05-20T05:00:00+00:00",
        ended_at_utc="2026-05-20T09:00:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_NO_RUN_SCOPED_DIR
    assert any(("RUN_NOT_ASSIGNED_TO_DATE" == r.rejected_reason for r in sel.rejected))


def test_unverified_runs_excluded(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:00:00+00:00",
        metadata_verified_for_normalization=False,
        verified_offset_min=None,
        timestamp_semantics_status="TIMESTAMP_SEMANTICS_PENDING",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_NO_VERIFIED_RUN_METADATA
    assert any((r.rejected_reason == "UNVERIFIED_RUN_METADATA" for r in sel.rejected))


def test_longest_verified_default(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T20:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id == "live_run_20260518T053000Z"


def test_latest_verified_policy(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T20:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LATEST_VERIFIED)
    assert sel.selected_run_id == "live_run_20260518T193000Z"


def test_explicit_policy_uses_map(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T20:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date(
        "2026-05-18",
        cands,
        policy=POLICY_EXPLICIT,
        run_id_map={"2026-05-18": "live_run_20260518T193000Z"},
    )
    assert sel.selected_run_id == "live_run_20260518T193000Z"


def test_explicit_policy_missing_entry_fails_day(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_EXPLICIT, run_id_map={})
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_EXPLICIT_MAP_MISSING_ENTRY


def test_explicit_policy_unknown_run_id_fails(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date(
        "2026-05-18",
        cands,
        policy=POLICY_EXPLICIT,
        run_id_map={"2026-05-18": "live_run_doesnotexist"},
    )
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_EXPLICIT_MAP_RUN_NOT_FOUND


def test_explicit_policy_unverified_target_fails(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
        metadata_verified_for_normalization=False,
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date(
        "2026-05-18",
        cands,
        policy=POLICY_EXPLICIT,
        run_id_map={"2026-05-18": "live_run_20260518T053000Z"},
    )
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_EXPLICIT_MAP_RUN_NOT_VERIFIED


def test_fail_on_multiple_fails_when_multiple_verified(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T20:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_FAIL_ON_MULTIPLE)
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_AMBIGUOUS


def test_fail_on_multiple_passes_with_one(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_FAIL_ON_MULTIPLE)
    assert sel.selected_run_id == "live_run_20260518T053000Z"


def test_cross_midnight_warning_in_selection(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T220000Z",
        started_at_utc="2026-05-18T22:00:00+00:00",
        ended_at_utc="2026-05-19T03:00:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id == "live_run_20260518T220000Z"
    assert WARNING_CROSS_MIDNIGHT in sel.warnings


def test_cross_midnight_run_not_selected_for_next_date(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T220000Z",
        started_at_utc="2026-05-18T22:00:00+00:00",
        ended_at_utc="2026-05-19T03:00:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-19", cands, policy=POLICY_LONGEST_VERIFIED)
    assert sel.selected_run_id is None
    assert sel.error_reason == REASON_NO_RUN_SCOPED_DIR


def test_no_duplicate_assignment_across_two_dates(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T220000Z",
        started_at_utc="2026-05-18T22:00:00+00:00",
        ended_at_utc="2026-05-19T03:00:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    s18 = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    s19 = select_run_for_date("2026-05-19", cands, policy=POLICY_LONGEST_VERIFIED)
    assert s18.selected_run_id == "live_run_20260518T220000Z"
    assert s19.selected_run_id is None


def test_selection_reports_candidates_and_selected(tmp_path: Path) -> None:
    _seed_run(
        tmp_path,
        "live_run_20260518T053000Z",
        started_at_utc="2026-05-18T05:30:00+00:00",
        ended_at_utc="2026-05-18T09:30:00+00:00",
    )
    _seed_run(
        tmp_path,
        "live_run_20260518T193000Z",
        started_at_utc="2026-05-18T19:30:00+00:00",
        ended_at_utc="2026-05-18T20:30:00+00:00",
    )
    cands = discover_run_candidates(tmp_path)
    sel = select_run_for_date("2026-05-18", cands, policy=POLICY_LONGEST_VERIFIED)
    doc = sel.to_dict()
    assert doc["selected_run_id"] == "live_run_20260518T053000Z"
    assert len(doc["candidates"]) == 2
    assert doc["selected_run"]["duration_seconds"] == pytest.approx(4 * 3600)
    assert doc["selected_run_started_at_utc"] is not None


def test_parse_run_id_map_basic() -> None:
    out = parse_run_id_map("2026-05-18=live_run_X,2026-05-19=live_run_Y")
    assert out == {"2026-05-18": "live_run_X", "2026-05-19": "live_run_Y"}


@pytest.mark.parametrize("bad", ["abc", "=live_run_X", "2026-05-18=", ""])
def test_parse_run_id_map_rejects_bad(bad: str) -> None:
    if bad == "":
        assert parse_run_id_map(bad) == {}
        return
    with pytest.raises(ValueError):
        parse_run_id_map(bad)


def test_invalid_policy_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        select_run_for_date("2026-05-18", [], policy="invented_policy")
