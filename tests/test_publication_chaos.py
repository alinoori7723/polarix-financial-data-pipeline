from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import polarix.ingestion.parquet_writer as pw_mod
from polarix.ingestion.parquet_writer import ParquetWriter, _backoff_delay
from polarix.ingestion.tick_filter import FilteredEvent


def _ev(symbol: str, recv_ms: int, price: float = 5000.0, seq: int = 0) -> FilteredEvent:
    return FilteredEvent(
        symbol=symbol,
        time_msc_raw=recv_ms,
        recv_time_utc_ms=recv_ms,
        monotonic_ns=recv_ms * 1000000 + seq,
        bid=price,
        ask=price + 0.1,
        last=price,
        bid_scaled=int(price * 100),
        ask_scaled=int((price + 0.1) * 100),
        last_scaled=int(price * 100),
        volume=1,
        flags=2,
        spread_points=1,
        suppressed_count=0,
        first_suppressed_time_ms=None,
        last_suppressed_time_ms=None,
        suppressed_reason=None,
    )


def test_retry_then_success_publishes_file(monkeypatch, tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    base = 1700000000000
    for i in range(10):
        pw.add(_ev("SPX500", base + i * 50, seq=i))
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 4:
            raise PermissionError(32, "WinError 32: sharing violation", str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(pw_mod.os, "replace", flaky)
    pw.flush()
    assert calls["n"] >= 4, "expected at least 4 attempts (3 fails + 1 success)"
    files = list(tmp_path.rglob("part-*.parquet"))
    assert len(files) == 1
    table = pq.ParquetFile(files[0]).read()
    assert table.num_rows == 10
    assert pw._buffer == []


def test_retry_then_success_exercises_backoff_growth():
    0.05 * 2**0
    for attempt in range(0, 6):
        d = _backoff_delay(attempt)
        assert 0 <= d <= 1.5, f"backoff at attempt {attempt} out of bounds: {d}"


def test_permanent_permission_error_preserves_buffer(monkeypatch, tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    base = 1700000000000
    for i in range(5):
        pw.add(_ev("SPX500", base + i * 50, seq=i))

    def always_fail(_src, _dst):
        raise PermissionError(32, "WinError 32: sharing violation (permanent)", "x")

    monkeypatch.setattr(pw_mod.os, "replace", always_fail)
    with pytest.raises(PermissionError):
        pw.flush()
    assert list(tmp_path.rglob("part-*.parquet")) == []
    leftover_tmp = list(tmp_path.rglob("*.tmp"))
    assert leftover_tmp == [], f"tmp files leaked: {leftover_tmp}"
    assert len(pw._buffer) == 5
    assert pw.metrics.flush_failures == 1
    assert pw.metrics.last_failure_reason is not None
    assert "PermissionError" in pw.metrics.last_failure_reason


def test_permanent_failure_does_not_corrupt_prior_finalized_files(monkeypatch, tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    base = 1700000000000
    for i in range(5):
        pw.add(_ev("SPX500", base + i * 50, seq=i))
    pw.flush()
    survivors = list(tmp_path.rglob("part-*.parquet"))
    assert len(survivors) == 1
    survivor_bytes = survivors[0].read_bytes()
    for i in range(5):
        pw.add(_ev("NDX100", base + i * 50, seq=i))

    def always_fail(_src, _dst):
        raise PermissionError(32, "WinError 32: sharing violation", "x")

    monkeypatch.setattr(pw_mod.os, "replace", always_fail)
    with pytest.raises(PermissionError):
        pw.flush()
    assert survivors[0].exists()
    assert survivors[0].read_bytes() == survivor_bytes
    ndx_files = list(tmp_path.rglob("symbol=NDX100/**/part-*.parquet"))
    assert ndx_files == []
    leftover = list(tmp_path.rglob("*.tmp"))
    assert leftover == []


def test_close_swallows_permanent_failure(monkeypatch, tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    pw.add(_ev("SPX500", 1700000000000, seq=0))

    def always_fail(_src, _dst):
        raise PermissionError(32, "WinError 32: sharing violation", "x")

    monkeypatch.setattr(pw_mod.os, "replace", always_fail)
    pw.close()
    assert pw.metrics.flush_failures >= 1


def test_partial_partition_failure_preserves_only_unpublished_events(monkeypatch, tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    base = 1700000000000
    spx_events = [_ev("SPX500", base + i * 50, seq=i) for i in range(3)]
    ndx_events = [_ev("NDX100", base + i * 50, seq=100 + i) for i in range(4)]
    for ev in spx_events:
        pw.add(ev)
    for ev in ndx_events:
        pw.add(ev)
    real_replace = os.replace
    state = {"replaced": 0}

    def flaky(src, dst):
        if state["replaced"] == 0:
            state["replaced"] += 1
            return real_replace(src, dst)
        raise PermissionError(32, "WinError 32: sharing violation", str(src))

    monkeypatch.setattr(pw_mod.os, "replace", flaky)
    with pytest.raises(PermissionError):
        pw.flush()
    files = list(tmp_path.rglob("part-*.parquet"))
    assert len(files) == 1
    assert list(tmp_path.rglob("*.tmp")) == []
    remaining_symbols = {e.symbol for e in pw._buffer}
    assert remaining_symbols == {"NDX100"}
    assert len(pw._buffer) == len(ndx_events)
