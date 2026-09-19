from __future__ import annotations

import os
from pathlib import Path

import pytest

from polarix.common import os_health


def test_disk_free_gb_returns_positive():
    usage = os_health.disk_free_gb("." if os.name == "nt" else "/")
    assert usage.total_gb > 0
    assert usage.free_gb >= 0
    assert usage.free_gb <= usage.total_gb


def test_available_memory_gb_or_runtime_error():
    if not os_health.HAS_PSUTIL:
        with pytest.raises(RuntimeError):
            os_health.available_memory_gb()
        return
    mem = os_health.available_memory_gb()
    assert mem.total_gb > 0
    assert 0 <= mem.available_gb <= mem.total_gb
    assert 0.0 <= mem.percent <= 100.0


def test_process_rss_mb_current_process():
    rss = os_health.process_rss_mb()
    if not os_health.HAS_PSUTIL:
        assert rss is None
    else:
        assert rss is not None and rss > 0


def test_find_processes_by_name_returns_list():
    if not os_health.HAS_PSUTIL:
        assert os_health.find_processes_by_name(["definitely_not_a_real_process_xyz"]) == []
        return
    out = os_health.find_processes_by_name(["definitely_not_a_real_process_xyz"])
    assert isinstance(out, list)
    assert all((not p.found or p.name for p in out))


def test_directory_writable_true_for_tmp(tmp_path: Path):
    assert os_health.directory_writable(tmp_path) is True


def test_directory_writable_false_when_path_is_file(tmp_path: Path):
    target = tmp_path / "afile"
    target.write_text("x")
    assert os_health.directory_writable(target) is False
