from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS = 10


@dataclass
class _ActiveChild:
    name: str
    process: object
    argv: list[str]
    started_at_monotonic: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class CleanupRecord:
    name: str
    pid: Optional[int]
    argv: list[str]
    outcome: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pid": self.pid,
            "argv": list(self.argv),
            "outcome": self.outcome,
            "error": self.error,
        }


class ChildProcessRegistry:
    def __init__(self) -> None:
        self._active: list[_ActiveChild] = []

    def add(self, name: str, process: object, argv: Iterable[str]) -> None:
        self._active.append(_ActiveChild(name=name, process=process, argv=list(argv)))

    def remove(self, process: object) -> None:
        self._active = [a for a in self._active if a.process is not process]

    def active_count(self) -> int:
        return len(self._active)

    def active_names(self) -> list[str]:
        return [a.name for a in self._active]

    def __len__(self) -> int:
        return self.active_count()

    def cleanup(
        self,
        *,
        timeout_seconds: float = DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS,
        kill_on_timeout: bool = True,
    ) -> list[CleanupRecord]:
        records: list[CleanupRecord] = []
        if not self._active:
            return records
        active = list(self._active)
        for child in active:
            proc = child.process
            rc = self._safe_poll(proc)
            if rc is not None:
                records.append(
                    CleanupRecord(
                        name=child.name,
                        pid=self._safe_pid(proc),
                        argv=child.argv,
                        outcome="ALREADY_EXITED",
                    )
                )
                continue
            try:
                proc.terminate()
            except Exception as exc:
                records.append(
                    CleanupRecord(
                        name=child.name,
                        pid=self._safe_pid(proc),
                        argv=child.argv,
                        outcome="ERROR",
                        error=f"terminate raised: {exc!r}",
                    )
                )
                continue
        deadline = time.monotonic() + timeout_seconds
        for child in active:
            if any(
                (r.name == child.name and r.outcome in ("ALREADY_EXITED", "ERROR") for r in records)
            ):
                continue
            proc = child.process
            remaining = max(0.0, deadline - time.monotonic())
            try:
                rc = self._safe_wait(proc, timeout=remaining)
            except subprocess.TimeoutExpired:
                rc = None
            if rc is not None:
                records.append(
                    CleanupRecord(
                        name=child.name,
                        pid=self._safe_pid(proc),
                        argv=child.argv,
                        outcome="TERMINATED",
                    )
                )
        for child in active:
            if any((r.name == child.name for r in records)):
                continue
            proc = child.process
            if not kill_on_timeout:
                records.append(
                    CleanupRecord(
                        name=child.name,
                        pid=self._safe_pid(proc),
                        argv=child.argv,
                        outcome="ERROR",
                        error="kill_on_timeout=False and process did not exit within timeout",
                    )
                )
                continue
            try:
                proc.kill()
                try:
                    self._safe_wait(proc, timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                records.append(
                    CleanupRecord(
                        name=child.name, pid=self._safe_pid(proc), argv=child.argv, outcome="KILLED"
                    )
                )
            except Exception as exc:
                records.append(
                    CleanupRecord(
                        name=child.name,
                        pid=self._safe_pid(proc),
                        argv=child.argv,
                        outcome="ERROR",
                        error=f"kill raised: {exc!r}",
                    )
                )
        self._active = []
        return records

    @staticmethod
    def _safe_poll(proc: object) -> Optional[int]:
        try:
            return proc.poll()
        except Exception:
            return None

    @staticmethod
    def _safe_pid(proc: object) -> Optional[int]:
        return getattr(proc, "pid", None)

    @staticmethod
    def _safe_wait(proc: object, timeout: float) -> Optional[int]:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise
        except Exception:
            return None


def run_tracked_subprocess(
    name: str,
    argv: list[str],
    *,
    registry: ChildProcessRegistry,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    popen_factory: Optional[callable] = None,
) -> subprocess.CompletedProcess:
    factory = popen_factory or subprocess.Popen
    proc = factory(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    registry.add(name, proc, argv)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            args=argv, returncode=proc.returncode, stdout=stdout, stderr=stderr
        )
    finally:
        registry.remove(proc)
