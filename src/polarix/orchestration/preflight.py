from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from polarix.common import os_health
from polarix.common.clock_health import STATUS_COARSE_OK, check_clock_health
from polarix.common.config import LoggerConfig

DEFAULT_MIN_FREE_DISK_GB = 20.0
DEFAULT_MIN_AVAILABLE_MEMORY_GB = 2.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "error": self.error}


@dataclass
class PreflightReport:
    started_at_utc: str
    finished_at_utc: str
    overall_ok: bool
    checks: list[CheckResult]
    config_path: str
    min_free_disk_gb: float
    min_available_memory_gb: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "overall_ok": self.overall_ok,
            "min_free_disk_gb": self.min_free_disk_gb,
            "min_available_memory_gb": self.min_available_memory_gb,
            "config_path": self.config_path,
            "notes": list(self.notes),
            "checks": [c.to_dict() for c in self.checks],
        }


def check_config_present(path: Path) -> CheckResult:
    exists = path.exists() and path.is_file()
    return CheckResult(
        name="config_present",
        ok=exists,
        detail={"path": str(path)},
        error=None if exists else f"config file not found: {path}",
    )


def check_python_can_import_logger() -> CheckResult:
    try:
        mod = importlib.import_module("polarix")
    except Exception as exc:
        return CheckResult(
            name="python_import_polarix", ok=False, error=f"{type(exc).__name__}: {exc}"
        )
    return CheckResult(
        name="python_import_polarix",
        ok=True,
        detail={"version": getattr(mod, "__version__", "unknown")},
    )


def check_no_trading_invariant(repo_root: Path) -> CheckResult:
    script = repo_root / "scripts" / "run_no_trading_invariant_check.py"
    if not script.exists():
        return CheckResult(
            name="no_trading_invariant",
            ok=False,
            detail={"script": str(script)},
            error=f"no-trading invariant scanner not found at {script}",
        )
    try:
        res = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return CheckResult(
            name="no_trading_invariant", ok=False, error=f"{type(exc).__name__}: {exc}"
        )
    ok = res.returncode == 0
    detail: dict[str, Any] = {"returncode": res.returncode}
    try:
        detail["report"] = json.loads(res.stdout)
    except Exception:
        detail["stdout_excerpt"] = (res.stdout or "")[:2000]
        detail["stderr_excerpt"] = (res.stderr or "")[:2000]
    return CheckResult(
        name="no_trading_invariant",
        ok=ok,
        detail=detail,
        error=None if ok else "no-trading invariant failed",
    )


def check_clock(threshold_ms: int) -> CheckResult:
    health = check_clock_health(threshold_ms=threshold_ms)
    return CheckResult(
        name="host_clock",
        ok=health.status == STATUS_COARSE_OK,
        detail={
            "status": health.status,
            "offset_ms": health.offset_ms,
            "threshold_ms": health.threshold_ms,
            "source": health.source,
            "error": health.error,
        },
        error=None
        if health.status == STATUS_COARSE_OK
        else f"clock not safe: {health.status} (offset_ms={health.offset_ms})",
    )


def check_disk(path: str | os.PathLike, min_free_gb: float) -> CheckResult:
    usage = os_health.disk_free_gb(path)
    ok = usage.free_gb >= min_free_gb
    return CheckResult(
        name="disk_free",
        ok=ok,
        detail={
            "path": usage.path,
            "free_gb": round(usage.free_gb, 3),
            "total_gb": round(usage.total_gb, 3),
            "min_free_gb": min_free_gb,
        },
        error=None if ok else f"disk free {usage.free_gb:.2f}GB below threshold {min_free_gb}GB",
    )


def check_memory(min_available_gb: float) -> CheckResult:
    try:
        mem = os_health.available_memory_gb()
    except RuntimeError as exc:
        return CheckResult(name="memory_available", ok=False, error=str(exc))
    ok = mem.available_gb >= min_available_gb
    return CheckResult(
        name="memory_available",
        ok=ok,
        detail={
            "available_gb": round(mem.available_gb, 3),
            "total_gb": round(mem.total_gb, 3),
            "min_available_gb": min_available_gb,
            "percent": mem.percent,
        },
        error=None
        if ok
        else f"available memory {mem.available_gb:.2f}GB below {min_available_gb}GB",
    )


def check_directories_writable(paths: Iterable[Path]) -> CheckResult:
    bad: list[str] = []
    info: list[dict[str, Any]] = []
    for p in paths:
        ok = os_health.directory_writable(p)
        info.append({"path": str(p), "writable": ok})
        if not ok:
            bad.append(str(p))
    return CheckResult(
        name="directories_writable",
        ok=not bad,
        detail={"paths": info},
        error=None if not bad else f"not writable: {bad}",
    )


def check_mt5_surface(symbols: Iterable[str], mt5_module: Any | None = None) -> CheckResult:
    if mt5_module is None:
        try:
            mt5_module = importlib.import_module("MetaTrader5")
        except Exception as exc:
            return CheckResult(
                name="mt5_surface",
                ok=False,
                error=f"MetaTrader5 not importable: {type(exc).__name__}: {exc}",
            )
    detail: dict[str, Any] = {}
    initialized = False
    try:
        if not mt5_module.initialize():
            err = mt5_module.last_error() if hasattr(mt5_module, "last_error") else None
            return CheckResult(
                name="mt5_surface",
                ok=False,
                detail={"last_error": str(err)},
                error="mt5.initialize() returned False",
            )
        initialized = True
        ti = mt5_module.terminal_info()
        if ti is None:
            return CheckResult(name="mt5_surface", ok=False, error="terminal_info() returned None")
        terminal_connected = bool(getattr(ti, "connected", False))
        terminal_trade_allowed = bool(getattr(ti, "trade_allowed", False))
        detail["terminal"] = {
            "connected": terminal_connected,
            "trade_allowed": terminal_trade_allowed,
            "company": str(getattr(ti, "company", "")),
            "name": str(getattr(ti, "name", "")),
            "build": int(getattr(ti, "build", 0) or 0),
        }
        if not terminal_connected:
            return CheckResult(
                name="mt5_surface",
                ok=False,
                detail=detail,
                error="terminal_info.connected is False",
            )
        if terminal_trade_allowed:
            return CheckResult(
                name="mt5_surface",
                ok=False,
                detail=detail,
                error="terminal_info.trade_allowed is True -- AutoTrading must be off",
            )
        ai = mt5_module.account_info()
        if ai is None:
            return CheckResult(
                name="mt5_surface", ok=False, detail=detail, error="account_info() returned None"
            )
        account_trade_allowed = bool(getattr(ai, "trade_allowed", False))
        detail["account"] = {
            "server": str(getattr(ai, "server", "")),
            "company": str(getattr(ai, "company", "")),
            "currency": str(getattr(ai, "currency", "")),
            "leverage": int(getattr(ai, "leverage", 0) or 0),
            "trade_allowed": account_trade_allowed,
            "trade_expert": bool(getattr(ai, "trade_expert", False)),
        }
        notes: list[str] = []
        if account_trade_allowed:
            notes.append("account_info.trade_allowed=True (broker side); logger remains read-only")
        detail["notes"] = notes
        sym_results: dict[str, bool] = {}
        for s in symbols:
            sym_results[s] = bool(mt5_module.symbol_select(s, True))
        detail["symbol_select"] = sym_results
        missing = [s for s, ok in sym_results.items() if not ok]
        if missing:
            return CheckResult(
                name="mt5_surface",
                ok=False,
                detail=detail,
                error=f"failed to select symbols: {missing}",
            )
        return CheckResult(name="mt5_surface", ok=True, detail=detail)
    except Exception as exc:
        return CheckResult(
            name="mt5_surface", ok=False, detail=detail, error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        if initialized:
            try:
                mt5_module.shutdown()
            except Exception:
                pass


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def run_preflight(
    cfg: LoggerConfig,
    config_path: Path,
    repo_root: Path,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    mt5_module: Any | None = None,
    skip_mt5: bool = False,
    disk_path: str = ".",
) -> PreflightReport:
    started = _utc_iso()
    checks: list[CheckResult] = []
    notes: list[str] = []
    checks.append(check_config_present(config_path))
    checks.append(check_python_can_import_logger())
    checks.append(check_no_trading_invariant(repo_root))
    checks.append(check_clock(threshold_ms=cfg.max_host_clock_offset_ms))
    checks.append(check_disk(path=disk_path, min_free_gb=min_free_disk_gb))
    checks.append(check_memory(min_available_gb=min_available_memory_gb))
    checks.append(
        check_directories_writable(
            [cfg.data_root, cfg.reports_root, cfg.logs_root, cfg.raw_dataset_dir]
        )
    )
    if skip_mt5:
        notes.append("mt5_surface skipped by caller (skip_mt5=True)")
        checks.append(CheckResult(name="mt5_surface", ok=True, detail={"skipped": True}))
    else:
        checks.append(check_mt5_surface(symbols=cfg.symbols, mt5_module=mt5_module))
    overall_ok = all((c.ok for c in checks))
    return PreflightReport(
        started_at_utc=started,
        finished_at_utc=_utc_iso(),
        overall_ok=overall_ok,
        checks=checks,
        config_path=str(config_path),
        min_free_disk_gb=min_free_disk_gb,
        min_available_memory_gb=min_available_memory_gb,
        notes=notes,
    )


def write_preflight_failed_report(
    reports_root: Path, report: PreflightReport, timestamp: str
) -> Path:
    reports_root.mkdir(parents=True, exist_ok=True)
    out = reports_root / f"live_run_{timestamp}_preflight_failed.json"
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out
