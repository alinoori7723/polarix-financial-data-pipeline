from __future__ import annotations

from pathlib import Path

import duckdb

from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.synthetic_ticks import TickStream
from polarix.ingestion.tick_filter import TickFilter


def _run_pipeline(tmp_path: Path, ticks, flush_rows: int = 1000):
    tf = TickFilter(price_scale=100, stale_heartbeat_max_gap_ms=1000)
    pw = ParquetWriter(raw_dataset_dir=tmp_path, flush_max_rows=flush_rows, flush_max_seconds=60)
    for t in ticks:
        ev = tf.consider(
            symbol=t.symbol,
            time_msc_raw=t.time_msc_raw,
            recv_time_utc_ms=t.recv_time_utc_ms,
            monotonic_ns=t.monotonic_ns,
            bid=t.bid,
            ask=t.ask,
            last=t.last,
            volume=t.volume,
            flags=t.flags,
            spread_points=2,
        )
        if ev is not None:
            pw.add(ev)
    pw.flush()
    return (tf, pw)


def test_normal_profile_rows_match_emitted(tmp_path: Path):
    stream = TickStream(
        symbols=("SPX500", "NDX100"),
        rate_per_second_total=2000,
        duration_seconds=2,
        start_recv_time_utc_ms=1700000000000,
    )
    tf, pw = _run_pipeline(tmp_path, stream.normal(), flush_rows=500)
    assert pw.metrics.rows_written == tf.total_emitted
    glob = str(tmp_path / "**" / "part-*.parquet").replace("\\", "/")
    con = duckdb.connect()
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
    assert rows == pw.metrics.rows_written


def test_duplicate_burst_collapses_to_one_event(tmp_path: Path):
    stream = TickStream(
        symbols=("SPX500",),
        rate_per_second_total=1,
        duration_seconds=1,
        start_recv_time_utc_ms=1700000000000,
    )
    tf, pw = _run_pipeline(tmp_path, stream.duplicate_burst(n=200), flush_rows=100)
    assert tf.total_seen == 200
    assert tf.total_emitted == 1
    assert tf.total_suppressed == 199


def test_bounded_memory_at_high_rate(tmp_path: Path):
    stream = TickStream(
        symbols=("SPX500", "NDX100"),
        rate_per_second_total=10000,
        duration_seconds=3,
        start_recv_time_utc_ms=1700000000000,
    )
    tf, pw = _run_pipeline(tmp_path, stream.normal(), flush_rows=2000)
    assert pw.metrics.rows_written > 0
    glob = str(tmp_path / "**" / "part-*.parquet").replace("\\", "/")
    rows = duckdb.connect().execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
    assert rows == pw.metrics.rows_written
