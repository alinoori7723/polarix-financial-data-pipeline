from __future__ import annotations

import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import psutil

    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False
_BYTES_PER_GB = 1024.0**3
_BYTES_PER_MB = 1024.0**2


@dataclass(frozen=True)
class DiskUsage:
    path: str
    total_gb: float
    free_gb: float
    used_gb: float


@dataclass(frozen=True)
class MemoryUsage:
    total_gb: float
    available_gb: float
    used_gb: float
    percent: float


@dataclass(frozen=True)
class ProcessMemory:
    pid: int | None
    name: str
    rss_mb: float | None
    found: bool


def disk_free_gb(path: str | os.PathLike = ".") -> DiskUsage:
    usage = shutil.disk_usage(str(path))
    return DiskUsage(
        path=str(path),
        total_gb=usage.total / _BYTES_PER_GB,
        free_gb=usage.free / _BYTES_PER_GB,
        used_gb=usage.used / _BYTES_PER_GB,
    )


def available_memory_gb() -> MemoryUsage:
    if not HAS_PSUTIL:
        raise RuntimeError("psutil is required for memory probes")
    vm = psutil.virtual_memory()
    return MemoryUsage(
        total_gb=vm.total / _BYTES_PER_GB,
        available_gb=vm.available / _BYTES_PER_GB,
        used_gb=vm.used / _BYTES_PER_GB,
        percent=float(vm.percent),
    )


def process_rss_mb(pid: int | None = None) -> float | None:
    if not HAS_PSUTIL:
        return None
    try:
        proc = psutil.Process(pid) if pid is not None else psutil.Process()
        return proc.memory_info().rss / _BYTES_PER_MB
    except Exception:
        return None


def find_processes_by_name(names: Iterable[str]) -> list[ProcessMemory]:
    if not HAS_PSUTIL:
        return []
    wanted = {n.lower().rstrip(".exe") for n in names}
    out: list[ProcessMemory] = []
    seen: set[str] = set()
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            raw_name = proc.info.get("name") or ""
            base = raw_name.lower().rstrip(".exe")
            if base in wanted:
                rss = None
                try:
                    rss = proc.memory_info().rss / _BYTES_PER_MB
                except Exception:
                    pass
                out.append(
                    ProcessMemory(pid=proc.info["pid"], name=raw_name, rss_mb=rss, found=True)
                )
                seen.add(base)
        except Exception:
            continue
    for n in wanted - seen:
        out.append(ProcessMemory(pid=None, name=n, rss_mb=None, found=False))
    return out


def mt5_process_rss_mb() -> float | None:
    candidates = ("terminal64.exe", "terminal.exe", "metatrader.exe")
    found = [p for p in find_processes_by_name(candidates) if p.found and p.rss_mb is not None]
    if not found:
        return None
    return max((p.rss_mb or 0.0 for p in found))


def directory_writable(path: str | os.PathLike) -> bool:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = p / f".polarix_writeprobe_{secrets.token_hex(4)}.tmp"
    try:
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False
