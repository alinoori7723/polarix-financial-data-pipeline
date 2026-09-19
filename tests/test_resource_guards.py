from __future__ import annotations

from pathlib import Path

import pytest

from polarix.orchestration import resource_guards
from polarix.orchestration.resource_guards import (
    ResourceGuardError,
    assert_disk_free,
    assert_memory_available,
    get_disk_free_gb,
    snapshot,
)


def test_disk_free_above_threshold_passes(tmp_path: Path) -> None:
    measured = assert_disk_free(tmp_path, min_free_disk_gb=0.0)
    assert measured > 0


def test_disk_free_below_threshold_raises(tmp_path: Path) -> None:
    with pytest.raises(ResourceGuardError) as ei:
        assert_disk_free(tmp_path, min_free_disk_gb=10000000.0)
    err = ei.value
    assert err.reason == "DISK_GUARD_STOP"
    assert err.measured > 0
    assert err.threshold == pytest.approx(10000000.0)
    assert err.path == tmp_path
    payload = err.to_dict()
    assert payload["reason"] == "DISK_GUARD_STOP"
    assert payload["threshold"] == 10000000.0


def test_memory_guard_passes_above_threshold(monkeypatch) -> None:
    monkeypatch.setattr(resource_guards, "get_available_memory_gb", lambda: 8.0)
    assert_memory_available(min_available_memory_gb=2.0)


def test_memory_guard_fails_below_threshold(monkeypatch) -> None:
    monkeypatch.setattr(resource_guards, "get_available_memory_gb", lambda: 0.1)
    with pytest.raises(ResourceGuardError) as ei:
        assert_memory_available(min_available_memory_gb=2.0)
    err = ei.value
    assert err.reason == "MEMORY_GUARD_STOP"
    assert err.measured == pytest.approx(0.1)
    assert err.threshold == pytest.approx(2.0)


def test_get_disk_free_walks_up_to_existing_parent(tmp_path: Path) -> None:
    nonexistent = tmp_path / "a" / "b" / "c"
    assert not nonexistent.exists()
    free = get_disk_free_gb(nonexistent)
    assert free > 0


def test_snapshot_returns_both_values(tmp_path: Path) -> None:
    snap = snapshot(tmp_path)
    assert snap.disk_free_gb > 0
    assert snap.available_memory_gb >= 0
    d = snap.to_dict()
    assert "disk_free_gb" in d and "available_memory_gb" in d
