from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import pytest

from polarix.orchestration.subprocess_guard import (
    DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS,
    ChildProcessRegistry,
    run_tracked_subprocess,
)


@dataclass
class FakeProc:
    pid: int
    args: list[str]
    behavior: str = "exit_on_terminate"
    rc: Optional[int] = None
    terminate_called: bool = False
    kill_called: bool = False
    terminate_at: Optional[float] = None
    exit_delay_seconds: float = 0.0

    def poll(self) -> Optional[int]:
        if self.terminate_called and self.terminate_at is not None:
            if time.monotonic() >= self.terminate_at + self.exit_delay_seconds:
                if self.behavior in ("exit_on_terminate", "exit_after_terminate"):
                    if self.rc is None:
                        self.rc = -15
        return self.rc

    def terminate(self) -> None:
        self.terminate_called = True
        self.terminate_at = time.monotonic()

    def kill(self) -> None:
        self.kill_called = True
        self.rc = -9

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            rc = self.poll()
            if rc is not None:
                return rc
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout or 0)
            time.sleep(0.01)


def test_registry_tracks_and_removes() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(pid=111, args=["a"])
    reg.add("STAGE_A", p, ["a"])
    assert reg.active_count() == 1
    assert "STAGE_A" in reg.active_names()
    reg.remove(p)
    assert reg.active_count() == 0


def test_cleanup_on_empty_returns_empty() -> None:
    reg = ChildProcessRegistry()
    assert reg.cleanup(timeout_seconds=1) == []


def test_cleanup_already_exited_records_status() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(pid=111, args=["a"], rc=0)
    reg.add("S", p, ["a"])
    records = reg.cleanup(timeout_seconds=1)
    assert len(records) == 1
    assert records[0].outcome == "ALREADY_EXITED"
    assert records[0].pid == 111


def test_cleanup_terminates_normal_children() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(pid=111, args=["a"], behavior="exit_on_terminate")
    reg.add("S", p, ["a"])
    records = reg.cleanup(timeout_seconds=2)
    assert p.terminate_called is True
    assert len(records) == 1
    assert records[0].outcome == "TERMINATED"


def test_cleanup_kills_child_that_ignores_terminate() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(pid=111, args=["a"], behavior="ignore_terminate")
    reg.add("S", p, ["a"])
    records = reg.cleanup(timeout_seconds=0.2, kill_on_timeout=True)
    assert p.terminate_called is True
    assert p.kill_called is True
    assert records[0].outcome == "KILLED"
    assert records[0].argv == ["a"]


def test_cleanup_records_argv_and_pid_for_killed_child() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(
        pid=12345,
        args=["python", "-u", "scripts/foo.py", "--date", "2026-05-18"],
        behavior="ignore_terminate",
    )
    reg.add("FEATURE_EDA", p, p.args)
    [record] = reg.cleanup(timeout_seconds=0.2, kill_on_timeout=True)
    assert record.outcome == "KILLED"
    assert record.pid == 12345
    assert record.argv[2].endswith("foo.py")


def test_cleanup_without_kill_on_timeout_records_error() -> None:
    reg = ChildProcessRegistry()
    p = FakeProc(pid=111, args=["a"], behavior="ignore_terminate")
    reg.add("S", p, ["a"])
    records = reg.cleanup(timeout_seconds=0.1, kill_on_timeout=False)
    assert records[0].outcome == "ERROR"
    assert "did not exit" in (records[0].error or "")
    assert p.kill_called is False


def test_cleanup_clears_registry() -> None:
    reg = ChildProcessRegistry()
    p1 = FakeProc(pid=1, args=["a"], rc=0)
    p2 = FakeProc(pid=2, args=["b"], behavior="ignore_terminate")
    reg.add("A", p1, ["a"])
    reg.add("B", p2, ["b"])
    reg.cleanup(timeout_seconds=0.2)
    assert reg.active_count() == 0


@dataclass
class CompletingFakeProc:
    pid: int
    args: list[str]
    returncode: int = 0
    stdout_text: str = "ok"
    stderr_text: str = ""

    def communicate(self, timeout: Optional[float] = None):
        return (self.stdout_text, self.stderr_text)


def test_run_tracked_subprocess_removes_on_normal_completion() -> None:
    reg = ChildProcessRegistry()

    def factory(argv, cwd, env, stdout, stderr, text):
        return CompletingFakeProc(pid=42, args=argv)

    result = run_tracked_subprocess("S", ["echo", "hi"], registry=reg, popen_factory=factory)
    assert result.returncode == 0
    assert reg.active_count() == 0


def test_run_tracked_subprocess_does_not_orphan_on_exception() -> None:
    reg = ChildProcessRegistry()

    class Raiser:
        pid = 99
        args = ["x"]

        def communicate(self, timeout=None):
            raise RuntimeError("boom")

        def kill(self):
            pass

    def factory(argv, cwd, env, stdout, stderr, text):
        return Raiser()

    with pytest.raises(RuntimeError):
        run_tracked_subprocess("S", ["x"], registry=reg, popen_factory=factory)
    assert reg.active_count() == 0


def test_default_terminate_timeout_constant() -> None:
    assert DEFAULT_CHILD_TERMINATE_TIMEOUT_SECONDS == 10
