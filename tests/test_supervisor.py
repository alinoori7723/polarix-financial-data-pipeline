from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from polarix.common import os_health
from polarix.common.clock_health import STATUS_COARSE_OK, ClockHealth
from polarix.common.config import load_config
from polarix.orchestration.supervisor import (
    REASON_CRITICAL_HEALTH,
    REASON_DISK_PRESSURE,
    REASON_DURATION_ELAPSED,
    REASON_LOGGER_EXITED,
    REASON_MANUAL_STOP,
    REASON_MEMORY_PRESSURE,
    REASON_PREFLIGHT_FAILED,
    REASON_TERMINAL_TRADE_ALLOWED,
    SHUTDOWN_GRACEFUL,
    SHUTDOWN_KILLED,
    SHUTDOWN_TERMINATED,
    Supervisor,
    SupervisorConfig,
    _build_launch_command,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "logger.example.json"


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeSleep:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, dt: float) -> None:
        self.calls.append(dt)
        self.clock.advance(dt)


class FakeLoggerProcess:
    def __init__(
        self,
        command: list[str],
        console_log_path: Path,
        *,
        responds_to_stop_signal: bool = True,
        responds_to_terminate: bool = True,
        crash_immediately: bool = False,
        crash_after_seconds: float | None = None,
        clock: FakeClock | None = None,
        stop_signal_paths: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.console_log_path = console_log_path
        self.responds_to_stop_signal = responds_to_stop_signal
        self.responds_to_terminate = responds_to_terminate
        self.crash_immediately = crash_immediately
        self.crash_after_seconds = crash_after_seconds
        self.clock = clock
        self.stop_signal_paths = stop_signal_paths
        self._started = False
        self._alive = False
        self._returncode: int | None = None
        self._started_at: float = 0.0
        self._kill_called = False
        self._terminate_called = False
        self._pid = 12345

    def start(self) -> None:
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_log_path.touch(exist_ok=True)
        self._started = True
        self._alive = not self.crash_immediately
        if self.crash_immediately:
            self._returncode = 1
        self._started_at = self.clock() if self.clock is not None else 0.0

    def pid(self) -> int | None:
        return self._pid if self._started else None

    def _update_alive(self) -> None:
        if not self._alive:
            return
        if self.crash_after_seconds is not None and self.clock is not None:
            if self.clock() - self._started_at >= self.crash_after_seconds:
                self._alive = False
                self._returncode = 1
                return
        if self.responds_to_stop_signal:
            for p in self.stop_signal_paths:
                if p.exists():
                    self._alive = False
                    self._returncode = 0
                    return
        if self._terminate_called and self.responds_to_terminate:
            self._alive = False
            self._returncode = 143

    def poll(self) -> int | None:
        self._update_alive()
        return self._returncode

    def is_alive(self) -> bool:
        self._update_alive()
        return self._alive

    def returncode(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._terminate_called = True

    def kill(self) -> None:
        self._kill_called = True
        self._alive = False
        self._returncode = -9

    def close(self) -> None:
        pass


@pytest.fixture()
def cfg(tmp_path):
    base = load_config(CONFIG_PATH)
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


def _patch_preflight_ok(monkeypatch):
    from polarix.orchestration import preflight as pf
    from polarix.orchestration import supervisor as sup_mod

    monkeypatch.setattr(
        pf,
        "check_clock_health",
        lambda threshold_ms, **kw: ClockHealth(
            status=STATUS_COARSE_OK, offset_ms=1.0, threshold_ms=threshold_ms, source="fake"
        ),
    )

    def fake_run_preflight(*, cfg, config_path, repo_root, **kw):
        import datetime as _dt

        from polarix.orchestration.preflight import CheckResult, PreflightReport

        now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return PreflightReport(
            started_at_utc=now,
            finished_at_utc=now,
            overall_ok=True,
            checks=[
                CheckResult(name="config_present", ok=True),
                CheckResult(name="host_clock", ok=True),
                CheckResult(name="disk_free", ok=True),
                CheckResult(name="memory_available", ok=True),
                CheckResult(name="mt5_surface", ok=True, detail={"skipped": True}),
            ],
            config_path=str(config_path),
            min_free_disk_gb=kw.get("min_free_disk_gb", 0.0),
            min_available_memory_gb=kw.get("min_available_memory_gb", 0.0),
        )

    monkeypatch.setattr(sup_mod, "run_preflight", fake_run_preflight)


def _patch_preflight_fail(monkeypatch):
    from polarix.orchestration import supervisor as sup_mod

    def fake_run_preflight(*, cfg, config_path, repo_root, **kw):
        import datetime as _dt

        from polarix.orchestration.preflight import CheckResult, PreflightReport

        now = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return PreflightReport(
            started_at_utc=now,
            finished_at_utc=now,
            overall_ok=False,
            checks=[CheckResult(name="disk_free", ok=False, error="forced fail")],
            config_path=str(config_path),
            min_free_disk_gb=kw.get("min_free_disk_gb", 0.0),
            min_available_memory_gb=kw.get("min_available_memory_gb", 0.0),
        )

    monkeypatch.setattr(sup_mod, "run_preflight", fake_run_preflight)


def _write_logger_health(cfg, *, terminal_trade_allowed: bool, account_trade_allowed: bool = False):
    payload = {
        "broker": {
            "terminal_company": "FundedNext",
            "terminal_name": "MT5",
            "terminal_build": 4200,
            "trade_allowed": terminal_trade_allowed,
            "dlls_allowed": False,
        },
        "account": {
            "login_hash": "abcd1234",
            "server": "FundedNext-Server",
            "company": "FundedNext",
            "currency": "USD",
            "leverage": 100,
            "trade_allowed": account_trade_allowed,
            "trade_expert": False,
        },
        "clock_health": {"status": STATUS_COARSE_OK, "offset_ms": 1.0, "threshold_ms": 50},
    }
    (cfg.reports_root / "logger_health.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_supervisor(
    cfg, monkeypatch, *, proc_kwargs: dict | None = None, sup_kwargs: dict | None = None
):
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sup_cfg = SupervisorConfig(
        run_duration_minutes=1,
        check_interval_seconds=10,
        graceful_shutdown_timeout_seconds=5,
        terminate_timeout_seconds=2,
        max_restarts=1,
        market_open_grace_minutes=999,
        mt5_unavailable_grace_minutes=999,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        max_python_rss_mb=100000.0,
        max_mt5_rss_mb=100000.0,
        disk_path=str(cfg.data_root),
    )
    if sup_kwargs:
        for k, v in sup_kwargs.items():
            setattr(sup_cfg, k, v)
    proc_kwargs = dict(proc_kwargs or {})
    proc_kwargs.setdefault("clock", clock)
    proc_kwargs["stop_signal_paths"] = (cfg.logs_root / "stop.signal",)
    holder: dict[str, FakeLoggerProcess] = {}

    def factory(command, console_log_path):
        fake = FakeLoggerProcess(command, console_log_path, **proc_kwargs)
        holder["proc"] = fake
        return fake

    sup = Supervisor(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        sup_cfg=sup_cfg,
        timestamp="TESTSTAMP",
        process_factory=factory,
        clock_monotonic=clock,
        sleep=sleep,
    )
    return (sup, holder, clock, sleep)


def test_launch_command_uses_unbuffered_python_module():
    cmd = _build_launch_command()
    assert "-u" in cmd
    assert "-m" in cmd
    assert cmd[-1] == "polarix.ingestion.main"


def test_supervisor_stop_uses_stop_signal_first(cfg, monkeypatch):
    sup, holder, clock, sleep = _build_supervisor(
        cfg, monkeypatch, proc_kwargs={"responds_to_stop_signal": True}
    )
    sup.start_logger()
    mode = sup.stop_logger_graceful("test")
    assert mode == SHUTDOWN_GRACEFUL
    assert sup.obs.stop_signal_created is True
    proc = holder["proc"]
    assert proc._terminate_called is False
    assert proc._kill_called is False


def test_supervisor_escalates_to_terminate_when_stop_signal_ignored(cfg, monkeypatch):
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": False, "responds_to_terminate": True},
    )
    sup.start_logger()
    mode = sup.stop_logger_graceful("test")
    assert mode == SHUTDOWN_TERMINATED
    assert sup.obs.stop_signal_created is True
    proc = holder["proc"]
    assert proc._terminate_called is True
    assert proc._kill_called is False


def test_supervisor_escalates_to_kill_when_terminate_ignored(cfg, monkeypatch):
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": False, "responds_to_terminate": False},
    )
    sup.start_logger()
    mode = sup.stop_logger_graceful("test")
    assert mode == SHUTDOWN_KILLED
    proc = holder["proc"]
    assert proc._terminate_called is True
    assert proc._kill_called is True


def test_supervisor_stops_when_terminal_trade_allowed_flips_mid_run(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg, monkeypatch, proc_kwargs={"responds_to_stop_signal": True}
    )
    _write_logger_health(cfg, terminal_trade_allowed=True)
    rc = sup.run()
    assert sup.obs.exit_reason == REASON_TERMINAL_TRADE_ALLOWED
    assert rc != 0
    summary = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert any(summary["terminal_trade_allowed_observations"])
    assert summary["final_decision"] == "FAIL"


def test_supervisor_stops_on_disk_pressure(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    from polarix.orchestration import supervisor as sup_mod

    def fake_disk_free_gb(_path):
        return os_health.DiskUsage(path=str(_path), total_gb=100.0, free_gb=0.0, used_gb=100.0)

    monkeypatch.setattr(sup_mod.os_health, "disk_free_gb", fake_disk_free_gb)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"min_free_disk_gb": 1.0},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    rc = sup.run()
    assert sup.obs.exit_reason == REASON_DISK_PRESSURE
    assert rc != 0


def test_supervisor_stops_on_memory_pressure(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    from polarix.orchestration import supervisor as sup_mod

    monkeypatch.setattr(sup_mod.os_health, "process_rss_mb", lambda pid=None: 10000.0)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"max_python_rss_mb": 1.0},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    rc = sup.run()
    assert sup.obs.exit_reason == REASON_MEMORY_PRESSURE
    assert rc != 0


def test_supervisor_restarts_at_most_max_restarts(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sup_cfg = SupervisorConfig(
        run_duration_minutes=10,
        check_interval_seconds=5,
        graceful_shutdown_timeout_seconds=2,
        terminate_timeout_seconds=1,
        max_restarts=1,
        market_open_grace_minutes=999,
        mt5_unavailable_grace_minutes=999,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        max_python_rss_mb=100000.0,
        max_mt5_rss_mb=100000.0,
        disk_path=str(cfg.data_root),
    )
    holder: dict[str, list[FakeLoggerProcess]] = {"procs": []}

    def factory(command, console_log_path):
        fake = FakeLoggerProcess(
            command,
            console_log_path,
            crash_immediately=True,
            clock=clock,
            stop_signal_paths=(cfg.logs_root / "stop.signal",),
        )
        holder["procs"].append(fake)
        return fake

    sup = Supervisor(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        sup_cfg=sup_cfg,
        timestamp="TESTSTAMP",
        process_factory=factory,
        clock_monotonic=clock,
        sleep=sleep,
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert sup.obs.restart_count == 1
    assert sup.obs.exit_reason == REASON_LOGGER_EXITED
    assert len(holder["procs"]) == 2


def test_supervisor_produces_summary_report(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True, "crash_after_seconds": None},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert sup.obs.exit_reason == REASON_DURATION_ELAPSED
    json_path = cfg.reports_root / "live_run_TESTSTAMP_summary.json"
    txt_path = cfg.reports_root / "live_run_TESTSTAMP_summary.txt"
    assert json_path.exists()
    assert txt_path.exists()
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    for k in [
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "exit_reason",
        "preflight_passed",
        "broker_metadata",
        "terminal_trade_allowed_observations",
        "account_trade_allowed_observations",
        "clock_status",
        "disk_free_gb_min",
        "disk_free_gb_max",
        "python_rss_mb_peak",
        "mt5_rss_mb_peak",
        "restart_count",
        "critical_health_files",
        "raw_parquet_file_count",
        "compacted_parquet_file_count",
        "latest_data_file_mtime_utc",
        "shutdown_mode",
        "graceful_shutdown_succeeded",
        "stop_signal_created",
        "logger_returncode",
        "final_decision",
        "launch_command",
        "known_limitations",
    ]:
        assert k in summary, f"missing summary key: {k}"


def test_summary_records_shutdown_mode_graceful(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg, monkeypatch, proc_kwargs={"responds_to_stop_signal": True}
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    summary = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert summary["shutdown_mode"] == SHUTDOWN_GRACEFUL
    assert summary["graceful_shutdown_succeeded"] is True
    assert summary["stop_signal_created"] is True


def test_summary_records_shutdown_mode_killed(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": False, "responds_to_terminate": False},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    summary = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert summary["shutdown_mode"] == SHUTDOWN_KILLED
    assert summary["graceful_shutdown_succeeded"] is False


def test_supervisor_stops_when_critical_health_file_appears(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    _write_logger_health(cfg, terminal_trade_allowed=False)
    appeared: dict[str, bool] = {"flag": False}
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sup_cfg = SupervisorConfig(
        run_duration_minutes=10,
        check_interval_seconds=5,
        graceful_shutdown_timeout_seconds=2,
        terminate_timeout_seconds=1,
        max_restarts=0,
        market_open_grace_minutes=999,
        mt5_unavailable_grace_minutes=999,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        max_python_rss_mb=100000.0,
        max_mt5_rss_mb=100000.0,
        disk_path=str(cfg.data_root),
    )

    def factory(command, console_log_path):
        return FakeLoggerProcess(
            command,
            console_log_path,
            clock=clock,
            stop_signal_paths=(cfg.logs_root / "stop.signal",),
            responds_to_stop_signal=True,
        )

    original_sleep = sleep

    def patched_sleep(dt):
        original_sleep(dt)
        if not appeared["flag"]:
            (cfg.reports_root / "logger_critical-20260101T000000000000Z.json").write_text(
                json.dumps({"severity": "CRITICAL", "event": "KILL_SWITCH_TRIPPED"}),
                encoding="utf-8",
            )
            appeared["flag"] = True

    sup = Supervisor(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        sup_cfg=sup_cfg,
        timestamp="TESTSTAMP",
        process_factory=factory,
        clock_monotonic=clock,
        sleep=patched_sleep,
    )
    sup.run()
    assert sup.obs.exit_reason == REASON_CRITICAL_HEALTH
    assert sup.obs.critical_health_files


def test_supervisor_preflight_failure_writes_failed_report(cfg, monkeypatch):
    _patch_preflight_fail(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(cfg, monkeypatch)
    rc = sup.run()
    assert rc == 2
    assert sup.obs.exit_reason == REASON_PREFLIGHT_FAILED
    failed = cfg.reports_root / "live_run_TESTSTAMP_preflight_failed.json"
    assert failed.exists()
    assert "proc" not in holder


def test_logger_main_loop_exits_on_stop_signal(cfg, monkeypatch):
    from types import SimpleNamespace

    import polarix.ingestion.mt5_readonly as ro_mod
    from polarix.ingestion.main import run_logger

    class _FakeMT5:
        def initialize(self):
            return True

        def shutdown(self):
            return None

        def last_error(self):
            return (0, "ok")

        def terminal_info(self):
            return SimpleNamespace(
                name="FakeTerminal",
                company="FakeBroker",
                path="C:/fake",
                build=4200,
                trade_allowed=False,
                dlls_allowed=False,
                connected=True,
            )

        def account_info(self):
            return SimpleNamespace(
                login=123456,
                server="Fake-Server",
                company="FakeBroker",
                currency="USD",
                leverage=100,
                trade_allowed=False,
                trade_expert=False,
            )

        def symbol_select(self, _name, _enable):
            return True

        def symbol_info(self, _name):
            return None

        def symbol_info_tick(self, _name):
            return None

    monkeypatch.setattr(ro_mod, "mt5", _FakeMT5())
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    cfg.logs_root.mkdir(parents=True, exist_ok=True)
    (cfg.logs_root / "stop.signal").write_text("test", encoding="utf-8")
    rc = run_logger(cfg, run_once=False)
    assert rc == 0


_FORBIDDEN_TRADING_TOKENS = (
    "order_send",
    "order_check",
    "position_close",
    "positions_close",
    "position_modify",
    "trade_request",
    "TradeRequest",
    "MqlTradeRequest",
)
_PS_SCRIPTS = (
    REPO_ROOT / "scripts" / "run_controlled_live.ps1",
    REPO_ROOT / "scripts" / "install_windows_task.ps1",
)


@pytest.mark.parametrize("script", _PS_SCRIPTS)
def test_powershell_scripts_contain_no_trading_logic(script):
    text = script.read_text(encoding="utf-8")
    for token in _FORBIDDEN_TRADING_TOKENS:
        assert token not in text, f"forbidden token {token!r} in {script}"


def test_install_windows_task_does_not_run_in_session_0():
    raw = (REPO_ROOT / "scripts" / "install_windows_task.ps1").read_text(encoding="utf-8")
    non_comment_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0]
        non_comment_lines.append(line)
    text = "\n".join(non_comment_lines)
    forbidden_phrases = (
        "Run whether user is logged on or not",
        "-Password",
        "LogonType ServiceAccount",
        "LogonType S4U",
        "LogonType Password",
    )
    for phrase in forbidden_phrases:
        assert phrase not in text, f"install_windows_task.ps1 sets forbidden phrase: {phrase!r}"
    assert re.search("-LogonType\\s+Interactive", text), "must set -LogonType Interactive"
    assert "-Once" in text, "trigger must be one-time (-Once)"


def test_no_trading_functions_in_new_modules():
    paths = [
        REPO_ROOT / "src" / "polarix" / "orchestration" / "supervisor.py",
        REPO_ROOT / "src" / "polarix" / "orchestration" / "preflight.py",
        REPO_ROOT / "src" / "polarix" / "common" / "os_health.py",
        REPO_ROOT / "scripts" / "run_controlled_live.py",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TRADING_TOKENS:
            assert token not in text, f"forbidden token {token!r} in {p}"


def _sleep_that_creates_stop_signal_at(
    base_sleep: "FakeSleep", clock: "FakeClock", stop_signal_path: Path, *, at_or_after: float
):
    created = {"done": False}

    def _sleep(dt: float) -> None:
        base_sleep(dt)
        if not created["done"] and clock() >= at_or_after:
            stop_signal_path.parent.mkdir(parents=True, exist_ok=True)
            stop_signal_path.write_text(
                json.dumps({"requested_at_utc": "operator", "reason": "operator_stop"}),
                encoding="utf-8",
            )
            created["done"] = True

    return _sleep


def test_operator_stop_signal_during_run_rc0_no_restart_terminal_graceful(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 10},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.sleep = _sleep_that_creates_stop_signal_at(
        sleep, clock, cfg.logs_root / "stop.signal", at_or_after=5.0
    )
    rc = sup.run()
    assert sup.obs.restart_count == 0
    assert sup.obs.exit_reason == REASON_MANUAL_STOP
    assert sup.obs.shutdown_mode == SHUTDOWN_GRACEFUL
    assert sup.obs.logger_returncode == 0
    assert rc == 0
    summary = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert summary["exit_reason"] == REASON_MANUAL_STOP
    assert summary["restart_count"] == 0
    assert summary["shutdown_mode"] == SHUTDOWN_GRACEFUL
    assert summary["graceful_shutdown_succeeded"] is True
    assert summary["logger_graceful_self_exit"] is True
    assert summary["final_decision"] == "PARTIAL"
    health = sup._health_jsonl.read_text(encoding="utf-8")
    assert "logger_exited_unexpectedly" not in health
    assert "logger_exited_gracefully" in health


def test_duration_boundary_rc0_no_restart(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 10},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.sleep = _sleep_that_creates_stop_signal_at(
        sleep, clock, cfg.logs_root / "stop.signal", at_or_after=50.0
    )
    rc = sup.run()
    assert sup.obs.restart_count == 0
    assert sup.obs.exit_reason == REASON_DURATION_ELAPSED
    assert sup.obs.shutdown_mode == SHUTDOWN_GRACEFUL
    assert sup.obs.logger_returncode == 0
    assert rc == 0
    summary = json.loads(
        (cfg.reports_root / "live_run_TESTSTAMP_summary.json").read_text(encoding="utf-8")
    )
    assert summary["exit_reason"] == REASON_DURATION_ELAPSED
    assert summary["restart_count"] == 0
    assert summary["logger_graceful_self_exit"] is True
    assert summary["final_decision"] == "PASS"
    health = sup._health_jsonl.read_text(encoding="utf-8")
    assert "logger_exited_unexpectedly" not in health


def test_logger_nonzero_exit_still_restarts(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sup_cfg = SupervisorConfig(
        run_duration_minutes=10,
        check_interval_seconds=5,
        graceful_shutdown_timeout_seconds=2,
        terminate_timeout_seconds=1,
        max_restarts=1,
        market_open_grace_minutes=999,
        mt5_unavailable_grace_minutes=999,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        max_python_rss_mb=100000.0,
        max_mt5_rss_mb=100000.0,
        disk_path=str(cfg.data_root),
    )
    holder: dict[str, list[FakeLoggerProcess]] = {"procs": []}

    def factory(command, console_log_path):
        fake = FakeLoggerProcess(
            command,
            console_log_path,
            crash_immediately=True,
            clock=clock,
            stop_signal_paths=(cfg.logs_root / "stop.signal",),
        )
        holder["procs"].append(fake)
        return fake

    sup = Supervisor(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        sup_cfg=sup_cfg,
        timestamp="TESTSTAMP",
        process_factory=factory,
        clock_monotonic=clock,
        sleep=sleep,
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert sup.obs.restart_count == 1
    assert sup.obs.exit_reason == REASON_LOGGER_EXITED
    assert len(holder["procs"]) == 2
    assert sup._graceful_terminal_exit is False
    health = sup._health_jsonl.read_text(encoding="utf-8")
    assert "logger_exited_unexpectedly" in health


def test_operator_stop_signal_is_not_cleared_as_stale_during_run(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 10},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.sleep = _sleep_that_creates_stop_signal_at(
        sleep, clock, cfg.logs_root / "stop.signal", at_or_after=5.0
    )
    sup.run()
    assert "proc" in holder
    health_lines = [
        json.loads(line)
        for line in sup._health_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = [e for e in health_lines if e.get("event") == "logger_started"]
    assert len(started) == 1, "logger must be started exactly once (no restart)"
    assert not (cfg.logs_root / "stop.signal").exists()


def test_stale_stop_signal_at_startup_is_cleared_and_logged(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    cfg.logs_root.mkdir(parents=True, exist_ok=True)
    stale = cfg.logs_root / "stop.signal"
    stale.write_text(
        json.dumps({"requested_at_utc": "2026-05-18T09:20:55Z", "reason": "previous_run"}),
        encoding="utf-8",
    )
    assert stale.exists()
    sup, holder, clock, sleep = _build_supervisor(
        cfg, monkeypatch, proc_kwargs={"responds_to_stop_signal": True}
    )
    sup.start_logger()
    assert not stale.exists(), "stale stop.signal must be cleared before launch"
    health = sup._health_jsonl.read_text(encoding="utf-8")
    assert "STALE_STOP_SIGNAL_CLEARED" in health
    assert any(("stale stop.signal cleared" in n for n in sup.obs.notes))


def test_stale_stop_signal_does_not_poison_run(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    cfg.logs_root.mkdir(parents=True, exist_ok=True)
    (cfg.logs_root / "stop.signal").write_text(
        json.dumps({"requested_at_utc": "stale", "reason": "previous_run"}), encoding="utf-8"
    )
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert sup.obs.exit_reason == REASON_DURATION_ELAPSED


def test_graceful_shutdown_removes_stop_signal(cfg, monkeypatch):
    _patch_preflight_ok(monkeypatch)
    sup, holder, clock, sleep = _build_supervisor(
        cfg,
        monkeypatch,
        proc_kwargs={"responds_to_stop_signal": True},
        sup_kwargs={"run_duration_minutes": 1, "check_interval_seconds": 30},
    )
    _write_logger_health(cfg, terminal_trade_allowed=False)
    sup.run()
    assert sup.obs.shutdown_mode == SHUTDOWN_GRACEFUL
    assert not (cfg.logs_root / "stop.signal").exists()


def test_stop_signal_paths_in_tests_are_tmp_only():
    import ast

    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    brand = "Shi" + "vax"
    forbidden_substrings = (brand + "/logs", brand + "\\logs")
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for needle in forbidden_substrings:
                if needle in node.value:
                    bad.append((node.lineno, node.value))
                    break
    assert not bad, (
        f"test_supervisor.py must not reference the production logs path in string literals: {bad!r}; supervisor unit tests must isolate stop.signal to tmp_path"
    )
