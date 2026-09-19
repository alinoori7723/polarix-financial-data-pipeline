from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from polarix import __version__
from polarix.common import os_health
from polarix.common.config import LoggerConfig, ensure_runtime_dirs, load_config
from polarix.ingestion.main import STOP_SIGNAL_FILENAME
from polarix.orchestration.preflight import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    DEFAULT_MIN_FREE_DISK_GB,
    PreflightReport,
    run_preflight,
    write_preflight_failed_report,
)

REASON_DURATION_ELAPSED = "DURATION_ELAPSED"
REASON_MANUAL_STOP = "MANUAL_STOP_SIGNAL"
REASON_TERMINAL_TRADE_ALLOWED = "TERMINAL_TRADE_ALLOWED"
REASON_DISK_PRESSURE = "DISK_PRESSURE_STOP"
REASON_MEMORY_PRESSURE = "MEMORY_PRESSURE_STOP"
REASON_CRITICAL_HEALTH = "CRITICAL_HEALTH_DETECTED"
REASON_LOGGER_EXITED = "LOGGER_EXITED"
REASON_NO_PARQUET_OUTPUT = "NO_PARQUET_OUTPUT"
REASON_MT5_UNAVAILABLE = "MT5_UNAVAILABLE"
REASON_PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
REASON_SUPERVISOR_FAULT = "SUPERVISOR_FAULT"
EXIT_CODE_ZERO_REASONS = (REASON_DURATION_ELAPSED, REASON_MANUAL_STOP)
SHUTDOWN_GRACEFUL = "graceful"
SHUTDOWN_TERMINATED = "terminated"
SHUTDOWN_KILLED = "killed"
SHUTDOWN_NOT_STARTED = "not_started"
SHUTDOWN_ALREADY_EXITED = "already_exited"


@dataclass
class SupervisorConfig:
    run_duration_minutes: int = 240
    check_interval_seconds: int = 30
    graceful_shutdown_timeout_seconds: int = 20
    terminate_timeout_seconds: int = 10
    max_restarts: int = 1
    market_open_grace_minutes: int = 20
    mt5_unavailable_grace_minutes: int = 10
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB
    max_python_rss_mb: float = 1500.0
    max_mt5_rss_mb: float = 1500.0
    disk_path: str = "."


@dataclass
class RunObservations:
    started_at_utc: str
    ended_at_utc: str | None = None
    exit_reason: str | None = None
    preflight_ok: bool = False
    restart_count: int = 0
    shutdown_mode: str = SHUTDOWN_NOT_STARTED
    stop_signal_created: bool = False
    logger_returncode: int | None = None
    terminal_trade_allowed_observations: list[bool] = field(default_factory=list)
    account_trade_allowed_observations: list[bool] = field(default_factory=list)
    clock_status: dict[str, Any] | None = None
    broker_metadata: dict[str, Any] = field(default_factory=dict)
    disk_free_gb_min: float | None = None
    disk_free_gb_max: float | None = None
    python_rss_mb_peak: float | None = None
    mt5_rss_mb_peak: float | None = None
    critical_health_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    config_used: dict[str, Any] = field(default_factory=dict)
    timeouts_used: dict[str, Any] = field(default_factory=dict)
    launch_command: list[str] = field(default_factory=list)


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _utc_compact() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_launch_command(python_exe: str | None = None) -> list[str]:
    py = python_exe or sys.executable
    return [py, "-u", "-m", "polarix.ingestion.main"]


def _build_launch_env(repo_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    src_path = str(Path(repo_root).resolve() / "src")
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts = [p for p in existing.split(os.pathsep) if p]
        if src_path not in parts:
            parts.insert(0, src_path)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        env["PYTHONPATH"] = src_path
    return env


def preflight_import_check(
    python_exe: str | Path,
    repo_root: Path,
    *,
    runner: Callable[[list[str], dict[str, str]], subprocess.CompletedProcess] | None = None,
) -> tuple[bool, str]:
    env = _build_launch_env(Path(repo_root))
    argv = [
        str(python_exe),
        "-c",
        "import polarix, polarix.ingestion.main, polarix.orchestration.supervisor; print('polarix import OK')",
    ]
    if runner is None:

        def runner(a, e):
            return subprocess.run(a, env=e, capture_output=True, text=True, timeout=30)

    try:
        result = runner(argv, env)
    except Exception as exc:
        return (False, f"preflight import check raised: {type(exc).__name__}: {exc}")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-1500:]
        return (False, tail)
    return (True, (result.stdout or "")[-1500:])


def _stop_signal_path(cfg: LoggerConfig) -> Path:
    return cfg.logs_root / STOP_SIGNAL_FILENAME


def _list_critical_health_files(reports_root: Path) -> list[Path]:
    if not reports_root.exists():
        return []
    return sorted(reports_root.glob("logger_critical-*.json"))


def _count_parquet_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum((1 for _ in root.rglob("part-*.parquet")))


def _count_compacted_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum((1 for _ in root.rglob("part-compacted-*.parquet")))


def _latest_data_file_mtime(roots: Iterable[Path]) -> _dt.datetime | None:
    latest: float | None = None
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.parquet"):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if latest is None or mt > latest:
                latest = mt
    if latest is None:
        return None
    return _dt.datetime.fromtimestamp(latest, tz=_dt.timezone.utc)


def _read_logger_health(reports_root: Path) -> dict[str, Any] | None:
    p = reports_root / "logger_health.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


class LoggerProcess:
    def __init__(
        self,
        command: list[str],
        console_log_path: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = list(command)
        self.console_log_path = console_log_path
        self.cwd = cwd
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self) -> None:
        self.console_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.console_log_path, "ab", buffering=0)
        self._proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd) if self.cwd else None,
            env=self.env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def poll(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> int | None:
        return None if self._proc is None else self._proc.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def terminate(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        try:
            if self._log_fh is not None:
                self._log_fh.flush()
                self._log_fh.close()
        finally:
            self._log_fh = None


class Supervisor:
    def __init__(
        self,
        cfg: LoggerConfig,
        config_path: Path,
        repo_root: Path,
        sup_cfg: SupervisorConfig | None = None,
        timestamp: str | None = None,
        process_factory: Callable[[list[str], Path], LoggerProcess] | None = None,
        clock_monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        python_exe: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.repo_root = repo_root
        self.sup_cfg = sup_cfg or SupervisorConfig()
        self.timestamp = timestamp or _utc_compact()
        self.clock = clock_monotonic
        self.sleep = sleep
        self.python_exe = python_exe
        self._process_factory = process_factory or self._default_process_factory
        self._proc: LoggerProcess | None = None
        self._graceful_terminal_exit: bool = False
        self._console_log_path: Path = cfg.reports_root / f"live_run_{self.timestamp}_console.log"
        self._health_jsonl: Path = cfg.logs_root / f"supervisor_health_{self.timestamp}.jsonl"
        self._summary_json: Path = cfg.reports_root / f"live_run_{self.timestamp}_summary.json"
        self._summary_txt: Path = cfg.reports_root / f"live_run_{self.timestamp}_summary.txt"
        self.obs = RunObservations(started_at_utc=_utc_iso())
        self.obs.launch_command = _build_launch_command(self.python_exe)
        self.obs.timeouts_used = {
            "check_interval_seconds": self.sup_cfg.check_interval_seconds,
            "graceful_shutdown_timeout_seconds": self.sup_cfg.graceful_shutdown_timeout_seconds,
            "terminate_timeout_seconds": self.sup_cfg.terminate_timeout_seconds,
            "run_duration_minutes": self.sup_cfg.run_duration_minutes,
            "market_open_grace_minutes": self.sup_cfg.market_open_grace_minutes,
            "mt5_unavailable_grace_minutes": self.sup_cfg.mt5_unavailable_grace_minutes,
        }
        self.obs.config_used = dataclasses.asdict(self.sup_cfg)

    def _default_process_factory(self, command: list[str], console_log: Path) -> LoggerProcess:
        return LoggerProcess(
            command=command,
            console_log_path=console_log,
            cwd=self.repo_root,
            env=_build_launch_env(self.repo_root, os.environ),
        )

    def run_preflight(self) -> PreflightReport:
        ensure_runtime_dirs(self.cfg)
        report = run_preflight(
            cfg=self.cfg,
            config_path=self.config_path,
            repo_root=self.repo_root,
            min_free_disk_gb=self.sup_cfg.min_free_disk_gb,
            min_available_memory_gb=self.sup_cfg.min_available_memory_gb,
            disk_path=self.sup_cfg.disk_path,
        )
        self.obs.preflight_ok = report.overall_ok
        return report

    def _append_health(self, event: dict[str, Any]) -> None:
        self._health_jsonl.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, default=str)
        with open(self._health_jsonl, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _disk_sample(self) -> float | None:
        try:
            usage = os_health.disk_free_gb(self.sup_cfg.disk_path)
        except OSError:
            return None
        free = usage.free_gb
        if self.obs.disk_free_gb_min is None or free < self.obs.disk_free_gb_min:
            self.obs.disk_free_gb_min = free
        if self.obs.disk_free_gb_max is None or free > self.obs.disk_free_gb_max:
            self.obs.disk_free_gb_max = free
        return free

    def _python_rss(self) -> float | None:
        if self._proc is None or self._proc.pid() is None:
            return None
        rss = os_health.process_rss_mb(self._proc.pid())
        if rss is None:
            return None
        if self.obs.python_rss_mb_peak is None or rss > self.obs.python_rss_mb_peak:
            self.obs.python_rss_mb_peak = rss
        return rss

    def _mt5_rss(self) -> float | None:
        rss = os_health.mt5_process_rss_mb()
        if rss is None:
            return None
        if self.obs.mt5_rss_mb_peak is None or rss > self.obs.mt5_rss_mb_peak:
            self.obs.mt5_rss_mb_peak = rss
        return rss

    def _record_health_snapshot(self) -> dict[str, Any]:
        snap = _read_logger_health(self.cfg.reports_root)
        if snap is None:
            return {}
        broker = snap.get("broker") or {}
        account = snap.get("account") or {}
        clock = snap.get("clock_health") or {}
        if broker:
            self.obs.broker_metadata = {
                "terminal_company": broker.get("terminal_company"),
                "terminal_name": broker.get("terminal_name"),
                "terminal_build": broker.get("terminal_build"),
                "account_server": account.get("server"),
                "account_company": account.get("company"),
                "account_login_hash": account.get("login_hash"),
                "account_currency": account.get("currency"),
                "account_leverage": account.get("leverage"),
            }
            if "trade_allowed" in broker:
                self.obs.terminal_trade_allowed_observations.append(bool(broker["trade_allowed"]))
            if "trade_allowed" in account:
                self.obs.account_trade_allowed_observations.append(bool(account["trade_allowed"]))
        if clock:
            self.obs.clock_status = clock
        return snap

    def start_logger(self) -> None:
        command = _build_launch_command(self.python_exe)
        ssp = _stop_signal_path(self.cfg)
        if ssp.exists():
            stale_payload: dict[str, Any] = {"path": str(ssp)}
            try:
                stale_payload["contents"] = ssp.read_text(encoding="utf-8")
            except OSError:
                pass
            try:
                ssp.unlink(missing_ok=True)
                self.obs.notes.append(f"stale stop.signal cleared at startup: {ssp}")
                self._append_health(
                    {
                        "ts": _utc_iso(),
                        "event": "STALE_STOP_SIGNAL_CLEARED",
                        "stop_signal_path": str(ssp),
                        "stale_contents": stale_payload.get("contents"),
                    }
                )
            except OSError as exc:
                self._append_health(
                    {
                        "ts": _utc_iso(),
                        "event": "STALE_STOP_SIGNAL_CLEAR_FAILED",
                        "stop_signal_path": str(ssp),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        self._proc = self._process_factory(command, self._console_log_path)
        self._proc.start()
        self._append_health(
            {
                "ts": _utc_iso(),
                "event": "logger_started",
                "pid": self._proc.pid(),
                "command": command,
            }
        )

    def _create_stop_signal(self) -> None:
        p = _stop_signal_path(self.cfg)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({"requested_at_utc": _utc_iso(), "reason": "supervisor_graceful_stop"}),
                encoding="utf-8",
            )
            self.obs.stop_signal_created = True
        except OSError:
            pass

    def stop_logger_graceful(self, reason: str) -> str:
        if self._proc is None:
            return SHUTDOWN_NOT_STARTED
        if not self._proc.is_alive():
            return SHUTDOWN_ALREADY_EXITED
        self._append_health({"ts": _utc_iso(), "event": "stop_requested", "reason": reason})
        self._create_stop_signal()
        deadline = self.clock() + self.sup_cfg.graceful_shutdown_timeout_seconds
        while self.clock() < deadline:
            if not self._proc.is_alive():
                self._append_health({"ts": _utc_iso(), "event": "logger_exited_gracefully"})
                return SHUTDOWN_GRACEFUL
            self.sleep(0.5)
        self._append_health({"ts": _utc_iso(), "event": "terminate_called"})
        self._proc.terminate()
        deadline = self.clock() + self.sup_cfg.terminate_timeout_seconds
        while self.clock() < deadline:
            if not self._proc.is_alive():
                self._append_health({"ts": _utc_iso(), "event": "logger_exited_after_terminate"})
                return SHUTDOWN_TERMINATED
            self.sleep(0.5)
        self._append_health({"ts": _utc_iso(), "event": "kill_called"})
        self._proc.kill()
        self._proc.wait(timeout=5)
        return SHUTDOWN_KILLED

    def _terminal_trade_allowed_now(self) -> bool | None:
        snap = _read_logger_health(self.cfg.reports_root)
        if snap is None:
            return None
        broker = snap.get("broker") or {}
        if "trade_allowed" not in broker:
            return None
        return bool(broker["trade_allowed"])

    def _evaluate_stop_conditions(
        self, state: dict[str, Any], critical_files_seen: set[str]
    ) -> tuple[bool, str | None]:
        ta = self._terminal_trade_allowed_now()
        if ta is True:
            return (True, REASON_TERMINAL_TRADE_ALLOWED)
        disk_free = state.get("disk_free_gb")
        if disk_free is not None and disk_free < self.sup_cfg.min_free_disk_gb:
            return (True, REASON_DISK_PRESSURE)
        py_rss = state.get("python_rss_mb")
        if py_rss is not None and py_rss > self.sup_cfg.max_python_rss_mb:
            return (True, REASON_MEMORY_PRESSURE)
        mt5_rss = state.get("mt5_rss_mb")
        if mt5_rss is not None and mt5_rss > self.sup_cfg.max_mt5_rss_mb:
            return (True, REASON_MEMORY_PRESSURE)
        current_critical = {str(p) for p in _list_critical_health_files(self.cfg.reports_root)}
        new_critical = current_critical - critical_files_seen
        if new_critical:
            self.obs.critical_health_files = sorted(current_critical)
            critical_files_seen.update(new_critical)
            return (True, REASON_CRITICAL_HEALTH)
        return (False, None)

    def run(self) -> int:
        try:
            preflight = self.run_preflight()
        except Exception as exc:
            self.obs.exit_reason = REASON_SUPERVISOR_FAULT
            self.obs.notes.append(f"preflight raised: {type(exc).__name__}: {exc}")
            self._finalize()
            return 2
        if not preflight.overall_ok:
            self.obs.exit_reason = REASON_PREFLIGHT_FAILED
            write_preflight_failed_report(self.cfg.reports_root, preflight, self.timestamp)
            self.obs.notes.append("preflight failed; logger not started")
            self._finalize()
            return 2
        ok, detail = preflight_import_check(self.python_exe or sys.executable, self.repo_root)
        if not ok:
            self.obs.exit_reason = REASON_PREFLIGHT_FAILED
            self.obs.notes.append(
                f"import preflight failed: polarix could not be imported from {self.python_exe or sys.executable}. Detail: {detail}"
            )
            self._append_health(
                {
                    "ts": _utc_iso(),
                    "event": "IMPORT_PREFLIGHT_FAILED",
                    "python_exe": str(self.python_exe or sys.executable),
                    "repo_root": str(self.repo_root),
                    "detail": detail,
                }
            )
            self._finalize()
            return 2
        try:
            self.start_logger()
        except Exception as exc:
            self.obs.exit_reason = REASON_SUPERVISOR_FAULT
            self.obs.notes.append(f"start raised: {type(exc).__name__}: {exc}")
            self._finalize()
            return 2
        deadline = self.clock() + self.sup_cfg.run_duration_minutes * 60.0
        critical_seen: set[str] = {
            str(p) for p in _list_critical_health_files(self.cfg.reports_root)
        }
        first_parquet_seen = False
        mt5_unavailable_since: float | None = None
        market_open_started_at: float = self.clock()
        try:
            while True:
                now = self.clock()
                if now >= deadline:
                    self.obs.exit_reason = REASON_DURATION_ELAPSED
                    break
                assert self._proc is not None
                if not self._proc.is_alive():
                    rc = self._proc.returncode()
                    self.obs.logger_returncode = rc
                    if rc == 0:
                        stop_signal_present = _stop_signal_path(self.cfg).exists()
                        near_deadline = now >= deadline - self.sup_cfg.check_interval_seconds
                        self._graceful_terminal_exit = True
                        self.obs.exit_reason = (
                            REASON_DURATION_ELAPSED if near_deadline else REASON_MANUAL_STOP
                        )
                        self._append_health(
                            {
                                "ts": _utc_iso(),
                                "event": "logger_exited_gracefully",
                                "returncode": rc,
                                "stop_signal_present": stop_signal_present,
                                "near_duration_boundary": near_deadline,
                                "classified_exit_reason": self.obs.exit_reason,
                            }
                        )
                        self.obs.notes.append(
                            "logger exited with returncode 0 "
                            + (
                                "at/near the configured duration boundary"
                                if self.obs.exit_reason == REASON_DURATION_ELAPSED
                                else "after a stop.signal was observed"
                            )
                            + "; treated as a graceful terminal completion (no restart)."
                        )
                        break
                    self._append_health(
                        {"ts": _utc_iso(), "event": "logger_exited_unexpectedly", "returncode": rc}
                    )
                    if self.obs.restart_count >= self.sup_cfg.max_restarts:
                        self.obs.exit_reason = REASON_LOGGER_EXITED
                        break
                    re_pre = run_preflight(
                        cfg=self.cfg,
                        config_path=self.config_path,
                        repo_root=self.repo_root,
                        min_free_disk_gb=self.sup_cfg.min_free_disk_gb,
                        min_available_memory_gb=self.sup_cfg.min_available_memory_gb,
                        disk_path=self.sup_cfg.disk_path,
                        skip_mt5=True,
                    )
                    if not re_pre.overall_ok:
                        self.obs.exit_reason = REASON_LOGGER_EXITED
                        self.obs.notes.append("restart skipped: re-preflight failed")
                        break
                    self.obs.restart_count += 1
                    self.start_logger()
                    market_open_started_at = self.clock()
                    first_parquet_seen = False
                    mt5_unavailable_since = None
                    continue
                disk_free = self._disk_sample()
                py_rss = self._python_rss()
                mt5_rss = self._mt5_rss()
                snap = self._record_health_snapshot()
                state = {"disk_free_gb": disk_free, "python_rss_mb": py_rss, "mt5_rss_mb": mt5_rss}
                self._append_health(
                    {
                        "ts": _utc_iso(),
                        "event": "monitor_tick",
                        "pid": self._proc.pid(),
                        "disk_free_gb": disk_free,
                        "python_rss_mb": py_rss,
                        "mt5_rss_mb": mt5_rss,
                        "snap_age_known": bool(snap),
                    }
                )
                should_stop, reason = self._evaluate_stop_conditions(state, critical_seen)
                if should_stop and reason:
                    self.obs.exit_reason = reason
                    break
                if not first_parquet_seen:
                    if _count_parquet_files(self.cfg.raw_dataset_dir) > 0:
                        first_parquet_seen = True
                    elif (
                        now - market_open_started_at
                        >= self.sup_cfg.market_open_grace_minutes * 60.0
                    ):
                        if self.obs.restart_count < self.sup_cfg.max_restarts:
                            self.obs.notes.append("no parquet output after grace; restarting once")
                            self.stop_logger_graceful("no_parquet_output_restart")
                            self.obs.restart_count += 1
                            self.start_logger()
                            market_open_started_at = self.clock()
                            continue
                        self.obs.exit_reason = REASON_NO_PARQUET_OUTPUT
                        break
                if snap is None:
                    if mt5_unavailable_since is None:
                        mt5_unavailable_since = now
                    elif (
                        now - mt5_unavailable_since
                        >= self.sup_cfg.mt5_unavailable_grace_minutes * 60.0
                    ):
                        if self.obs.restart_count < self.sup_cfg.max_restarts:
                            self.obs.notes.append("mt5 unavailable beyond grace; restarting once")
                            self.stop_logger_graceful("mt5_unavailable_restart")
                            self.obs.restart_count += 1
                            self.start_logger()
                            mt5_unavailable_since = None
                            continue
                        self.obs.exit_reason = REASON_MT5_UNAVAILABLE
                        break
                else:
                    mt5_unavailable_since = None
                self.sleep(self.sup_cfg.check_interval_seconds)
        except Exception as exc:
            self.obs.notes.append(f"monitor loop fault: {type(exc).__name__}: {exc}")
            self.obs.exit_reason = self.obs.exit_reason or REASON_SUPERVISOR_FAULT
        if self._graceful_terminal_exit:
            self.obs.shutdown_mode = SHUTDOWN_GRACEFUL
            if self._proc is not None:
                self.obs.logger_returncode = self._proc.returncode()
                self._proc.close()
        else:
            mode = self.stop_logger_graceful(reason=self.obs.exit_reason or "duration")
            self.obs.shutdown_mode = mode
            if self._proc is not None:
                self.obs.logger_returncode = self._proc.returncode()
                self._proc.close()
        if self._graceful_terminal_exit or self.obs.shutdown_mode == SHUTDOWN_GRACEFUL:
            try:
                _stop_signal_path(self.cfg).unlink(missing_ok=True)
                self._append_health(
                    {"ts": _utc_iso(), "event": "stop_signal_cleaned_after_graceful_exit"}
                )
            except OSError:
                pass
        return self._finalize()

    def _finalize(self) -> int:
        self.obs.ended_at_utc = _utc_iso()
        final_decision = self._decide_pass_fail()
        summary = self._build_summary(final_decision)
        try:
            self.cfg.reports_root.mkdir(parents=True, exist_ok=True)
            self._summary_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            self._summary_txt.write_text(self._render_text_summary(summary), encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: could not write summary: {exc}", file=sys.stderr)
        try:
            from polarix.orchestration.run_metadata import write_run_scoped_metadata

            manifest_global = self.cfg.reports_root / "logger_manifest.json"
            health_global = self.cfg.reports_root / "logger_health.json"
            write_run_scoped_metadata(
                self.cfg.reports_root,
                f"live_run_{self.timestamp}",
                manifest_src=manifest_global if manifest_global.exists() else None,
                health_src=health_global if health_global.exists() else None,
                summary_doc=summary,
                console_log_src=self._console_log_path if self._console_log_path.exists() else None,
                supervisor_health_src=self._health_jsonl if self._health_jsonl.exists() else None,
            )
        except Exception as exc:
            self.obs.notes.append(
                f"failed to archive run-scoped metadata: {type(exc).__name__}: {exc}"
            )
        if self.obs.exit_reason in (*EXIT_CODE_ZERO_REASONS, None):
            return 0
        if self.obs.exit_reason in (REASON_LOGGER_EXITED,) and self.obs.logger_returncode == 0:
            return 0
        return 1

    def _decide_pass_fail(self) -> str:
        if self.obs.exit_reason in (
            REASON_TERMINAL_TRADE_ALLOWED,
            REASON_DISK_PRESSURE,
            REASON_MEMORY_PRESSURE,
            REASON_CRITICAL_HEALTH,
            REASON_MT5_UNAVAILABLE,
            REASON_NO_PARQUET_OUTPUT,
            REASON_PREFLIGHT_FAILED,
            REASON_SUPERVISOR_FAULT,
        ):
            return "FAIL"
        if self.obs.exit_reason == REASON_LOGGER_EXITED:
            return "PARTIAL"
        if self.obs.exit_reason == REASON_MANUAL_STOP:
            return "PARTIAL"
        if self.obs.restart_count > 0:
            return "PARTIAL"
        return "PASS"

    def _build_summary(self, final_decision: str) -> dict[str, Any]:
        start = _dt.datetime.fromisoformat(self.obs.started_at_utc)
        end = _dt.datetime.fromisoformat(self.obs.ended_at_utc) if self.obs.ended_at_utc else None
        duration_s = (end - start).total_seconds() if end else None
        raw_pq = _count_parquet_files(self.cfg.raw_dataset_dir)
        cp_pq = _count_compacted_files(self.cfg.compacted_dataset_dir)
        latest = _latest_data_file_mtime([self.cfg.raw_dataset_dir, self.cfg.compacted_dataset_dir])
        ts_status: str | None = None
        verified_offset_min_for_summary: int | None = None
        manifest_path = self.cfg.reports_root / "logger_manifest.json"
        if manifest_path.exists():
            try:
                manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
                ts = manifest_doc.get("timestamp_semantics") or {}
                status_value = ts.get("status")
                voff = ts.get("verified_offset_min")
                if isinstance(status_value, str):
                    ts_status = status_value
                if isinstance(voff, int) and (not isinstance(voff, bool)):
                    verified_offset_min_for_summary = voff
            except Exception:
                pass
        metadata_verified_for_normalization = ts_status in (
            "OFFSET_VERIFIED_FOR_SESSION",
            "UTC_EPOCH_VERIFIED",
        ) and isinstance(verified_offset_min_for_summary, int)
        run_id = self.timestamp
        run_metadata_dir = self.cfg.reports_root / "logger_runs" / f"live_run_{run_id}"
        return {
            "supervisor_version": __version__,
            "started_at_utc": self.obs.started_at_utc,
            "ended_at_utc": self.obs.ended_at_utc,
            "duration_seconds": duration_s,
            "exit_reason": self.obs.exit_reason,
            "preflight_passed": self.obs.preflight_ok,
            "broker_metadata": self.obs.broker_metadata,
            "terminal_trade_allowed_observations": self.obs.terminal_trade_allowed_observations,
            "account_trade_allowed_observations": self.obs.account_trade_allowed_observations,
            "clock_status": self.obs.clock_status,
            "disk_free_gb_min": self.obs.disk_free_gb_min,
            "disk_free_gb_max": self.obs.disk_free_gb_max,
            "python_rss_mb_peak": self.obs.python_rss_mb_peak,
            "mt5_rss_mb_peak": self.obs.mt5_rss_mb_peak,
            "restart_count": self.obs.restart_count,
            "critical_health_files": self.obs.critical_health_files,
            "raw_parquet_file_count": raw_pq,
            "compacted_parquet_file_count": cp_pq,
            "latest_data_file_mtime_utc": latest.isoformat() if latest else None,
            "shutdown_mode": self.obs.shutdown_mode,
            "graceful_shutdown_succeeded": self.obs.shutdown_mode == SHUTDOWN_GRACEFUL,
            "stop_signal_created": self.obs.stop_signal_created,
            "logger_graceful_self_exit": self._graceful_terminal_exit,
            "logger_returncode": self.obs.logger_returncode,
            "final_decision": final_decision,
            "launch_command": self.obs.launch_command,
            "timeouts_used": self.obs.timeouts_used,
            "supervisor_config": self.obs.config_used,
            "console_log": str(self._console_log_path),
            "health_jsonl": str(self._health_jsonl),
            "notes": list(self.obs.notes),
            "run_id": f"live_run_{run_id}",
            "run_metadata_dir": str(run_metadata_dir),
            "manifest_path": str(run_metadata_dir / "logger_manifest.json"),
            "health_path": str(run_metadata_dir / "logger_health.json"),
            "console_log_path": str(self._console_log_path),
            "supervisor_health_path": str(self._health_jsonl),
            "timestamp_semantics_status": ts_status,
            "verified_offset_min": verified_offset_min_for_summary,
            "metadata_verified_for_normalization": metadata_verified_for_normalization,
            "known_limitations": [
                "supervisor reads terminal_info.trade_allowed via logger_health.json; the logger writes this snapshot every ~30s, so the supervisor's view lags one snapshot interval at worst",
                "mt5 RSS sampling depends on psutil being installed; if absent the MEMORY_PRESSURE_STOP for the MT5 terminal is best-effort only",
                "graceful stop assumes the logger polls stop.signal at least once per iteration (added in this module)",
            ],
        }

    def _render_text_summary(self, summary: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("Polarix Live Run Supervisor Summary")
        lines.append("=" * 40)
        lines.append(f"started_at_utc:           {summary['started_at_utc']}")
        lines.append(f"ended_at_utc:             {summary['ended_at_utc']}")
        lines.append(f"duration_seconds:         {summary['duration_seconds']}")
        lines.append(f"exit_reason:              {summary['exit_reason']}")
        lines.append(f"final_decision:           {summary['final_decision']}")
        lines.append(f"preflight_passed:         {summary['preflight_passed']}")
        lines.append(f"shutdown_mode:            {summary['shutdown_mode']}")
        lines.append(f"graceful_shutdown_ok:     {summary['graceful_shutdown_succeeded']}")
        lines.append(f"logger_graceful_self_exit:{summary['logger_graceful_self_exit']}")
        lines.append(f"stop_signal_created:      {summary['stop_signal_created']}")
        lines.append(f"restart_count:            {summary['restart_count']}")
        lines.append(f"raw_parquet_file_count:   {summary['raw_parquet_file_count']}")
        lines.append(f"compacted_parquet_files:  {summary['compacted_parquet_file_count']}")
        lines.append(f"latest_data_file_mtime:   {summary['latest_data_file_mtime_utc']}")
        lines.append(
            f"disk_free_gb (min/max):   {summary['disk_free_gb_min']} / {summary['disk_free_gb_max']}"
        )
        lines.append(f"python_rss_mb_peak:       {summary['python_rss_mb_peak']}")
        lines.append(f"mt5_rss_mb_peak:          {summary['mt5_rss_mb_peak']}")
        lines.append(f"logger_returncode:        {summary['logger_returncode']}")
        lines.append(f"console_log:              {summary['console_log']}")
        lines.append(f"health_jsonl:             {summary['health_jsonl']}")
        lines.append(f"launch_command:           {' '.join(summary['launch_command'])}")
        ta = summary.get("terminal_trade_allowed_observations") or []
        lines.append(
            f"terminal.trade_allowed observations: {sum((1 for x in ta if x))} True / {len(ta)} total"
        )
        broker = summary.get("broker_metadata") or {}
        if broker:
            lines.append("broker_metadata:")
            for k, v in sorted(broker.items()):
                lines.append(f"  {k}: {v}")
        if summary.get("notes"):
            lines.append("notes:")
            for n in summary["notes"]:
                lines.append(f"  - {n}")
        if summary.get("known_limitations"):
            lines.append("known_limitations:")
            for n in summary["known_limitations"]:
                lines.append(f"  - {n}")
        return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="polarix-supervisor")
    p.add_argument("--config", default=None, help="path to logger.config.json")
    p.add_argument(
        "--repo-root", default=None, help="repository root (default: parent of this package)"
    )
    p.add_argument("--run-duration-minutes", type=int, default=None)
    p.add_argument("--check-interval-seconds", type=int, default=None)
    p.add_argument("--graceful-shutdown-timeout-seconds", type=int, default=None)
    p.add_argument("--max-restarts", type=int, default=None)
    p.add_argument("--min-free-disk-gb", type=float, default=None)
    p.add_argument("--min-available-memory-gb", type=float, default=None)
    p.add_argument("--max-python-rss-mb", type=float, default=None)
    p.add_argument("--max-mt5-rss-mb", type=float, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg_path = Path(args.config) if args.config else Path("config/logger.config.json")
    cfg = load_config(cfg_path)
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[3]
    )
    sup_cfg = SupervisorConfig()
    if args.run_duration_minutes is not None:
        sup_cfg.run_duration_minutes = args.run_duration_minutes
    if args.check_interval_seconds is not None:
        sup_cfg.check_interval_seconds = args.check_interval_seconds
    if args.graceful_shutdown_timeout_seconds is not None:
        sup_cfg.graceful_shutdown_timeout_seconds = args.graceful_shutdown_timeout_seconds
    if args.max_restarts is not None:
        sup_cfg.max_restarts = args.max_restarts
    if args.min_free_disk_gb is not None:
        sup_cfg.min_free_disk_gb = args.min_free_disk_gb
    if args.min_available_memory_gb is not None:
        sup_cfg.min_available_memory_gb = args.min_available_memory_gb
    if args.max_python_rss_mb is not None:
        sup_cfg.max_python_rss_mb = args.max_python_rss_mb
    if args.max_mt5_rss_mb is not None:
        sup_cfg.max_mt5_rss_mb = args.max_mt5_rss_mb
    sup = Supervisor(cfg=cfg, config_path=cfg_path, repo_root=repo_root, sup_cfg=sup_cfg)
    return sup.run()


if __name__ == "__main__":
    raise SystemExit(main())
