from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PS1_PATH = REPO_ROOT / "scripts" / "run_controlled_live.ps1"


def test_ps1_sets_pythonpath_to_repo_src() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert "$env:PYTHONPATH" in text, "PS1 must set $env:PYTHONPATH"
    assert 'Join-Path $RepoPath "src"' in text or "$RepoPath\\src" in text, (
        "PS1 must derive the src path from $RepoPath"
    )


def test_ps1_prepends_when_pythonpath_already_set() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert re.search("\\$PolarixSrc;\\$env:PYTHONPATH", text), (
        "PS1 must prepend src to an existing PYTHONPATH, not overwrite"
    )


def test_ps1_prints_pythonpath_in_banner() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert re.search('Write-Host\\s+".*PYTHONPATH', text), (
        "PS1 must print PYTHONPATH in the startup banner"
    )


def test_ps1_still_uses_python_unbuffered() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert " -u " in text, "PS1 must still pass -u to python"


def test_ps1_does_not_add_trading_logic() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    for token in (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    ):
        assert token not in text, f"PS1 contains forbidden trading token {token!r}"


def test_ps1_invokes_run_controlled_live_py() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert "run_controlled_live.py" in text


def test_build_launch_env_sets_pythonpath_when_absent(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import _build_launch_env

    base = {"FOO": "bar"}
    env = _build_launch_env(repo_root=tmp_path, base_env=base)
    assert env["FOO"] == "bar"
    assert env["PYTHONPATH"] == str((tmp_path / "src").resolve())


def test_build_launch_env_prepends_when_present_and_missing(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import _build_launch_env

    existing_path = str(tmp_path / "other" / "path")
    base = {"PYTHONPATH": existing_path}
    env = _build_launch_env(repo_root=tmp_path, base_env=base)
    src = str((tmp_path / "src").resolve())
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == src
    assert existing_path in parts


def test_build_launch_env_does_not_duplicate_existing_src(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import _build_launch_env

    src = str((tmp_path / "src").resolve())
    base = {"PYTHONPATH": src + os.pathsep + str(tmp_path / "other")}
    env = _build_launch_env(repo_root=tmp_path, base_env=base)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts.count(src) == 1


def test_build_launch_env_preserves_other_env(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import _build_launch_env

    base = {"KEEP_ME": "yes", "ALSO_KEEP": "1"}
    env = _build_launch_env(repo_root=tmp_path, base_env=base)
    assert env["KEEP_ME"] == "yes"
    assert env["ALSO_KEEP"] == "1"


def test_preflight_import_check_succeeds_against_real_repo() -> None:
    from polarix.orchestration.supervisor import preflight_import_check

    ok, detail = preflight_import_check(sys.executable, REPO_ROOT)
    assert ok, f"import preflight failed unexpectedly: {detail}"
    assert "polarix import OK" in detail


def test_preflight_import_check_fails_when_runner_returns_nonzero(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import preflight_import_check

    def runner(argv, env):
        return subprocess.CompletedProcess(
            argv, returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'polarix'"
        )

    ok, detail = preflight_import_check(sys.executable, REPO_ROOT, runner=runner)
    assert ok is False
    assert "ModuleNotFoundError" in detail


def test_preflight_import_check_passes_env_with_pythonpath(tmp_path: Path) -> None:
    from polarix.orchestration.supervisor import preflight_import_check

    captured: dict = {}

    def runner(argv, env):
        captured["argv"] = list(argv)
        captured["env"] = dict(env)
        return subprocess.CompletedProcess(argv, returncode=0, stdout="ok\n", stderr="")

    ok, _detail = preflight_import_check(sys.executable, REPO_ROOT, runner=runner)
    assert ok is True
    assert "PYTHONPATH" in captured["env"]
    src = str((REPO_ROOT / "src").resolve())
    parts = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert src in parts


def test_supervisor_run_fails_closed_when_import_preflight_fails(cfg, monkeypatch):
    from polarix.orchestration import supervisor as sup_mod
    from tests.test_supervisor import _build_supervisor, _patch_preflight_ok

    _patch_preflight_ok(monkeypatch)

    def failing_import_check(python_exe, repo_root, runner=None):
        return (False, "ModuleNotFoundError: No module named 'polarix'")

    monkeypatch.setattr(sup_mod, "preflight_import_check", failing_import_check)
    sup, holder, _clock, _sleep = _build_supervisor(
        cfg, monkeypatch, proc_kwargs={"responds_to_stop_signal": True}
    )
    rc = sup.run()
    assert rc == 2
    assert "proc" not in holder, "start_logger must NOT run after import preflight fails"
    assert sup.obs.exit_reason == sup_mod.REASON_PREFLIGHT_FAILED
    assert any(("import preflight failed" in n for n in sup.obs.notes))


def test_supervisor_run_succeeds_when_import_preflight_passes(cfg, monkeypatch):
    from polarix.orchestration import supervisor as sup_mod
    from tests.test_supervisor import _build_supervisor, _patch_preflight_ok, _write_logger_health

    _patch_preflight_ok(monkeypatch)
    monkeypatch.setattr(
        sup_mod, "preflight_import_check", lambda *_a, **_k: (True, "polarix import OK")
    )
    sup, holder, _clock, _sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert "proc" in holder, "logger should have been started"


def test_launch_command_unchanged_uses_unbuffered_python_module() -> None:
    from polarix.orchestration.supervisor import _build_launch_command

    cmd = _build_launch_command()
    assert "-u" in cmd
    assert "-m" in cmd
    assert cmd[-1] == "polarix.ingestion.main"


def test_supervisor_module_no_trading_after_fix() -> None:
    text = (REPO_ROOT / "src" / "polarix" / "orchestration" / "supervisor.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    ):
        assert token not in text, f"supervisor.py contains forbidden {token!r}"


def test_run_controlled_live_py_no_trading_after_fix() -> None:
    text = (REPO_ROOT / "scripts" / "run_controlled_live.py").read_text(encoding="utf-8")
    for token in (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    ):
        assert token not in text, f"run_controlled_live.py contains forbidden {token!r}"


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
