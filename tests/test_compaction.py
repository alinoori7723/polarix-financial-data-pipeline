from __future__ import annotations

import datetime as _dt
from pathlib import Path

import duckdb

from polarix.ingestion.compaction import (
    PartitionKey,
    compact_all_closed,
    compact_partition,
    discover_closed_partitions,
)
from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.tick_filter import FilteredEvent

YESTERDAY_UTC_MS = int(
    (_dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=1))
    .replace(hour=12, minute=0, second=0, microsecond=0)
    .timestamp()
    * 1000
)


def _ev(symbol: str, recv_ms: int, seq: int) -> FilteredEvent:
    return FilteredEvent(
        symbol=symbol,
        time_msc_raw=recv_ms,
        recv_time_utc_ms=recv_ms,
        monotonic_ns=recv_ms * 1000000 + seq,
        bid=5000.0 + seq * 0.01,
        ask=5000.1 + seq * 0.01,
        last=5000.0 + seq * 0.01,
        bid_scaled=int((5000.0 + seq * 0.01) * 100),
        ask_scaled=int((5000.1 + seq * 0.01) * 100),
        last_scaled=int((5000.0 + seq * 0.01) * 100),
        volume=1,
        flags=2,
        spread_points=1,
    )


def _write_n_files(raw_root: Path, n_files: int, rows_per_file: int, symbol: str = "SPX500") -> int:
    pw = ParquetWriter(
        raw_dataset_dir=raw_root, flush_max_rows=rows_per_file, flush_max_seconds=3600
    )
    total = 0
    seq = 0
    for file_idx in range(n_files):
        for _ in range(rows_per_file):
            ev = _ev(symbol, YESTERDAY_UTC_MS + seq, seq)
            pw.add(ev)
            seq += 1
            total += 1
        pw.flush()
    return total


def _yesterday() -> str:
    return (_dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=1)).strftime("%Y-%m-%d")


def test_discover_closed_partitions(tmp_path: Path):
    raw_root = tmp_path / "raw"
    _write_n_files(raw_root, n_files=2, rows_per_file=3)
    parts = discover_closed_partitions(raw_root)
    assert len(parts) == 1
    assert parts[0].date == _yesterday()
    assert parts[0].symbol == "SPX500"


def test_compaction_5_files(tmp_path: Path):
    raw_root = tmp_path / "raw"
    compacted_root = tmp_path / "compacted"
    total = _write_n_files(raw_root, n_files=5, rows_per_file=10)
    results = compact_all_closed(raw_root, compacted_root, chunk_files=80)
    assert len(results) == 1
    r = results[0]
    assert r.validation_passed
    assert r.source_rows == total
    assert r.output_rows == total
    assert r.error is None


def test_compaction_500_files(tmp_path: Path):
    raw_root = tmp_path / "raw"
    compacted_root = tmp_path / "compacted"
    total = _write_n_files(raw_root, n_files=500, rows_per_file=4)
    results = compact_all_closed(raw_root, compacted_root, chunk_files=80)
    assert len(results) == 1
    r = results[0]
    assert r.validation_passed
    assert r.source_rows == total
    assert r.output_rows == total


def test_row_count_invariant_via_duckdb(tmp_path: Path):
    raw_root = tmp_path / "raw"
    compacted_root = tmp_path / "compacted"
    total = _write_n_files(raw_root, n_files=20, rows_per_file=5)
    compact_all_closed(raw_root, compacted_root, chunk_files=8)
    con = duckdb.connect()
    glob = str(compacted_root / "**" / "part-compacted-*.parquet").replace("\\", "/")
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
    assert rows == total


def test_sources_not_deleted_when_validation_fails(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "raw"
    compacted_root = tmp_path / "compacted"
    _write_n_files(raw_root, n_files=3, rows_per_file=4)
    partition = discover_closed_partitions(raw_root)[0]
    import polarix.ingestion.compaction as cm

    real_concat = cm.pa.concat_tables

    def lossy_concat(tables, **kwargs):
        t = real_concat(tables, **kwargs)
        return t.slice(0, max(0, t.num_rows - 1))

    monkeypatch.setattr(cm.pa, "concat_tables", lossy_concat)
    result = compact_partition(partition, raw_root, compacted_root)
    assert not result.validation_passed
    assert result.error and "row_count_mismatch" in result.error
    remaining = list((raw_root / "symbol=SPX500" / f"date={_yesterday()}").rglob("part-*.parquet"))
    assert len(remaining) == 3


def test_partition_key_directory_layout():
    p = PartitionKey(symbol="NDX100", date="2024-01-15")
    raw = p.raw_dir(Path("R"))
    comp = p.compacted_dir(Path("C"))
    assert raw.parts[-2:] == ("symbol=NDX100", "date=2024-01-15")
    assert comp.parts[-2:] == ("symbol=NDX100", "date=2024-01-15")
