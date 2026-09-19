from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GB = 1024**3


class ResourceGuardError(RuntimeError):
    def __init__(
        self,
        reason: str,
        measured: float,
        threshold: float,
        path: Optional[Path] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"{reason}: measured={measured:.3f} threshold={threshold:.3f}"
            + (f" path={path}" if path is not None else "")
            + (f" detail={detail}" if detail else "")
        )
        self.reason = reason
        self.measured = float(measured)
        self.threshold = float(threshold)
        self.path = path
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "measured": self.measured,
            "threshold": self.threshold,
            "path": str(self.path) if self.path is not None else None,
            "detail": self.detail,
        }


def get_disk_free_gb(path: Path | str) -> float:
    p = Path(path)
    candidate = p
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return shutil.disk_usage(str(candidate)).free / GB


def assert_disk_free(path: Path | str, min_free_disk_gb: float) -> float:
    measured = get_disk_free_gb(path)
    if measured < min_free_disk_gb:
        raise ResourceGuardError(
            reason="DISK_GUARD_STOP",
            measured=measured,
            threshold=float(min_free_disk_gb),
            path=Path(path),
        )
    return measured


def get_available_memory_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / GB
    except Exception:
        return 0.0


def assert_memory_available(min_available_memory_gb: float) -> float:
    measured = get_available_memory_gb()
    if measured < min_available_memory_gb:
        raise ResourceGuardError(
            reason="MEMORY_GUARD_STOP", measured=measured, threshold=float(min_available_memory_gb)
        )
    return measured


@dataclass
class ResourceSnapshot:
    disk_free_gb: float
    available_memory_gb: float

    def to_dict(self) -> dict:
        return {"disk_free_gb": self.disk_free_gb, "available_memory_gb": self.available_memory_gb}


def snapshot(disk_path: Path | str) -> ResourceSnapshot:
    return ResourceSnapshot(
        disk_free_gb=get_disk_free_gb(disk_path), available_memory_gb=get_available_memory_gb()
    )
