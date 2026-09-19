from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from polarix.orchestration.databento_download_plan import (
    STATUS_DOWNLOAD_BLOCKED_BY_FLAGS,
    STATUS_DOWNLOAD_FAILED,
    STATUS_DOWNLOAD_MISSING_API_KEY,
    STATUS_DOWNLOAD_MISSING_MT5_WINDOW,
    STATUS_DOWNLOAD_OK,
    STATUS_DOWNLOAD_SKIPPED_EXISTS,
    STATUS_RANGE_UNAVAILABLE,
    DatabentoDownloadError,
    assert_window_is_single_day,
    plan_day_download,
    run_day_download,
)


def _mt5_ms_for_date(date_iso: str) -> tuple[int, int]:
    import datetime as _dt

    t0 = _dt.datetime.fromisoformat(f"{date_iso}T05:00:00+00:00")
    t1 = t0 + _dt.timedelta(hours=4)
    return (int(t0.timestamp() * 1000), int(t1.timestamp() * 1000))


def test_plan_requires_both_flags_to_unblock(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=False,
        allow_databento_download=False,
        api_key_present=True,
    )
    assert plan.will_download is False
    assert plan.block_reason == STATUS_DOWNLOAD_BLOCKED_BY_FLAGS
    plan2 = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=False,
        api_key_present=True,
    )
    assert plan2.will_download is False
    assert plan2.block_reason == STATUS_DOWNLOAD_BLOCKED_BY_FLAGS
    plan3 = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    assert plan3.will_download is True
    assert plan3.block_reason is None


def test_plan_blocks_when_api_key_missing(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=False,
    )
    assert plan.will_download is False
    assert plan.block_reason == STATUS_DOWNLOAD_MISSING_API_KEY


def test_plan_blocks_when_mt5_window_missing(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=None,
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    assert plan.will_download is False
    assert plan.block_reason == STATUS_DOWNLOAD_MISSING_MT5_WINDOW
    assert plan.start_utc == "" and plan.end_utc == ""


def test_plan_window_is_single_day(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        pre_roll_minutes=5,
        post_roll_minutes=5,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    assert_window_is_single_day(plan.date, plan.start_utc, plan.end_utc)


def test_assert_window_rejects_multi_day_window() -> None:
    with pytest.raises(DatabentoDownloadError) as ei:
        assert_window_is_single_day("2026-05-18", "2026-05-17T00:00:00Z", "2026-05-22T00:00:00Z")
    assert "multi-day" in str(ei.value)


def test_argv_contains_download_script_and_single_date(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    argv = plan.argv
    assert "-u" in argv
    assert any(("download_databento_sample.py" in a for a in argv))
    out_root_idx = argv.index("--output-root")
    out_root_value = argv[out_root_idx + 1]
    assert "date=2026-05-18" in out_root_value


def test_run_day_download_blocked_by_flags(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=False,
        allow_databento_download=False,
        api_key_present=True,
    )

    def runner(argv, timeout):
        raise AssertionError("runner must not be invoked when blocked by flags")

    outcome = run_day_download(plan, repo_root=tmp_path, runner=runner)
    assert outcome.status == STATUS_DOWNLOAD_BLOCKED_BY_FLAGS
    assert outcome.attempts == 0


def test_run_day_download_skip_if_exists(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    out_dir = Path(plan.output_root) / f"date={plan.date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / plan.output_filename).write_bytes(b"existing")

    def runner(argv, timeout):
        raise AssertionError("runner must not be invoked when skip_if_exists")

    outcome = run_day_download(plan, repo_root=tmp_path, runner=runner, skip_if_exists=True)
    assert outcome.status == STATUS_DOWNLOAD_SKIPPED_EXISTS


def test_run_day_download_range_unavailable(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )

    def runner(argv, timeout):
        return subprocess.CompletedProcess(
            argv,
            returncode=2,
            stdout="",
            stderr="ERROR: data_not_available_range outside subscription range",
        )

    outcome = run_day_download(plan, repo_root=tmp_path, runner=runner, skip_if_exists=False)
    assert outcome.status == STATUS_RANGE_UNAVAILABLE
    assert outcome.attempts == 1


def test_run_day_download_failed_status(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
        max_retries=1,
    )
    call_log: list[int] = []

    def runner(argv, timeout):
        call_log.append(1)
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="random error")

    outcome = run_day_download(plan, repo_root=tmp_path, runner=runner, skip_if_exists=False)
    assert outcome.status == STATUS_DOWNLOAD_FAILED
    assert outcome.attempts == 2


def test_run_day_download_success(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=_mt5_ms_for_date("2026-05-18"),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
    )
    out_dir = Path(plan.output_root) / f"date={plan.date}"

    def runner(argv, timeout):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / plan.output_filename).write_bytes(b"x" * 16)
        return subprocess.CompletedProcess(argv, returncode=0, stdout="OK", stderr="")

    outcome = run_day_download(plan, repo_root=tmp_path, runner=runner, skip_if_exists=False)
    assert outcome.status == STATUS_DOWNLOAD_OK
    assert outcome.output_path is not None and outcome.output_path.exists()
