from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import duckdb

from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.synthetic_ticks import TickStream
from polarix.ingestion.tick_filter import TickFilter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=int, default=10000)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path(".polarix/data/raw/synthetic"))
    p.add_argument("--report", type=Path, default=Path(".polarix/reports/synthetic_burst.json"))
    args = p.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    stream = TickStream(
        symbols=("SPX500", "NDX100"),
        rate_per_second_total=args.rate,
        duration_seconds=args.duration,
        start_recv_time_utc_ms=1700000000000,
    )
    tf = TickFilter(price_scale=100, stale_heartbeat_max_gap_ms=1000)
    pw = ParquetWriter(raw_dataset_dir=args.out, flush_max_rows=5000, flush_max_seconds=60)
    t_start = time.monotonic()
    seen = 0
    for tick in stream.normal():
        seen += 1
        ev = tf.consider(
            symbol=tick.symbol,
            time_msc_raw=tick.time_msc_raw,
            recv_time_utc_ms=tick.recv_time_utc_ms,
            monotonic_ns=tick.monotonic_ns,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            flags=tick.flags,
            spread_points=2,
        )
        if ev is not None:
            pw.add(ev)
    pw.flush()
    elapsed = time.monotonic() - t_start
    glob = str(args.out / "**" / "part-*.parquet").replace("\\", "/")
    con = duckdb.connect()
    parquet_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
    report = {
        "rate_per_second_total": args.rate,
        "duration_seconds": args.duration,
        "elapsed_wall_seconds": elapsed,
        "ticks_seen": seen,
        "emitted": tf.total_emitted,
        "suppressed": tf.total_suppressed,
        "parquet_rows": parquet_rows,
        "files": pw.metrics.files_written,
        "bytes": pw.metrics.bytes_written,
        "rows_match_emitted": parquet_rows == tf.total_emitted,
        "out_dir": str(args.out),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["rows_match_emitted"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
