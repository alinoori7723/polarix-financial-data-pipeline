from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from polarix.ingestion.parquet_writer import (
    PARQUET_SCHEMA,
    ParquetWriter,
    _backoff_delay,
    safe_replace,
)
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
    )


def test_writer_creates_partitioned_files(tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=10, flush_max_seconds=60)
    base = 1700000000000
    for i in range(20):
        pw.add(_ev("SPX500", base + i * 50, 5000 + i * 0.1, seq=i))
    pw.flush()
    files = list(tmp_path.rglob("part-*.parquet"))
    assert files, "no parquet parts produced"
    for f in files:
        assert "symbol=SPX500" in str(f)
        assert "date=" in str(f)
        assert "hour=" in str(f)


def test_writer_emits_two_symbols_separately(tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=100, flush_max_seconds=60)
    base = 1700000000000
    for i in range(5):
        pw.add(_ev("SPX500", base + i * 100, seq=i))
        pw.add(_ev("NDX100", base + i * 100, seq=i))
    pw.flush()
    spx = list(tmp_path.rglob("symbol=SPX500/**/part-*.parquet"))
    ndx = list(tmp_path.rglob("symbol=NDX100/**/part-*.parquet"))
    assert spx and ndx


def test_writer_schema_matches(tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=5, flush_max_seconds=60)
    base = 1700000000000
    for i in range(5):
        pw.add(_ev("SPX500", base + i * 50, seq=i))
    pw.flush()
    f = next(tmp_path.rglob("part-*.parquet"))
    table = pq.ParquetFile(f).read()
    assert table.schema.equals(PARQUET_SCHEMA, check_metadata=False)


def test_duckdb_can_read_output(tmp_path: Path):
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=5, flush_max_seconds=60)
    base = 1700000000000
    for i in range(5):
        pw.add(_ev("SPX500", base + i * 50, seq=i))
    pw.flush()
    glob = str(tmp_path / "**" / "part-*.parquet").replace("\\", "/")
    con = duckdb.connect()
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()
    assert rows[0] == 5


def test_safe_replace_succeeds_normal(tmp_path: Path):
    src = tmp_path / "a.tmp"
    dst = tmp_path / "a"
    src.write_bytes(b"hello")
    safe_replace(src, dst)
    assert dst.read_bytes() == b"hello"
    assert not src.exists()


def test_safe_replace_retries_on_permission_error(monkeypatch, tmp_path: Path):
    src = tmp_path / "a.tmp"
    dst = tmp_path / "a"
    src.write_bytes(b"hi")
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(s, d):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError("simulated AV hold")
        return real_replace(s, d)

    monkeypatch.setattr("polarix.ingestion.parquet_writer.os.replace", flaky_replace)
    safe_replace(src, dst)
    assert dst.read_bytes() == b"hi"
    assert attempts["n"] >= 3


def test_backoff_is_bounded():
    for attempt in range(20):
        d = _backoff_delay(attempt)
        assert 0 <= d <= 1.5
