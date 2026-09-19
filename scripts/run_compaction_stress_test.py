from __future__ import annotations

import argparse
import datetime as _dt
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

from polarix.ingestion.compaction import compact_all_closed
from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.tick_filter import FilteredEvent


def _make_event(symbol: str, recv_ms: int, seq: int) -> FilteredEvent:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--files", type=int, default=500)
    p.add_argument("--rows-per-file", type=int, default=4)
    p.add_argument("--workdir", type=Path, default=Path(".polarix/data/compaction_stress"))
    p.add_argument("--report", type=Path, default=Path(".polarix/reports/compaction_stress.json"))
    args = p.parse_args()
    if args.workdir.exists():
        shutil.rmtree(args.workdir, ignore_errors=True)
    raw_root = args.workdir / "raw"
    compacted_root = args.workdir / "compacted"
    raw_root.mkdir(parents=True, exist_ok=True)
    compacted_root.mkdir(parents=True, exist_ok=True)
    yesterday_noon = (_dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    base_ms = int(yesterday_noon.timestamp() * 1000)
    pw = ParquetWriter(
        raw_dataset_dir=raw_root, flush_max_rows=args.rows_per_file, flush_max_seconds=3600
    )
    total_rows = 0
    t0 = time.monotonic()
    for f in range(args.files):
        for r in range(args.rows_per_file):
            seq = f * args.rows_per_file + r
            pw.add(_make_event("SPX500", base_ms + seq, seq))
            total_rows += 1
        pw.flush()
    t_write = time.monotonic() - t0
    t1 = time.monotonic()
    results = compact_all_closed(raw_root, compacted_root, chunk_files=80)
    t_compact = time.monotonic() - t1
    parquet_rows = 0
    glob = str(compacted_root / "**" / "part-compacted-*.parquet").replace("\\", "/")
    try:
        parquet_rows = (
            duckdb.connect().execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
        )
    except Exception:
        parquet_rows = -1
    report = {
        "files": args.files,
        "rows_per_file": args.rows_per_file,
        "total_rows_written": total_rows,
        "write_seconds": t_write,
        "compact_seconds": t_compact,
        "results": [
            {
                "partition_symbol": r.partition.symbol,
                "partition_date": r.partition.date,
                "source_files": r.source_files,
                "source_rows": r.source_rows,
                "output_files": r.output_files,
                "output_rows": r.output_rows,
                "min_recv_time_utc_ms": r.min_recv_time_utc_ms,
                "max_recv_time_utc_ms": r.max_recv_time_utc_ms,
                "sources_deleted": r.sources_deleted,
                "sources_left_due_to_lock": r.sources_left_due_to_lock,
                "validation_passed": r.validation_passed,
                "error": r.error,
            }
            for r in results
        ],
        "parquet_rows_after": parquet_rows,
        "workdir": str(args.workdir),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all((r.validation_passed for r in results)):
        return 1
    if parquet_rows != total_rows:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
