from __future__ import annotations

import datetime as _dt
import os
import random
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.ingestion.tick_filter import FilteredEvent

PARQUET_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("time_msc_raw", pa.int64()),
        ("recv_time_utc_ms", pa.int64()),
        ("monotonic_ns", pa.int64()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("last", pa.float64()),
        ("bid_scaled", pa.int64()),
        ("ask_scaled", pa.int64()),
        ("last_scaled", pa.int64()),
        ("volume", pa.int64()),
        ("flags", pa.int64()),
        ("spread_points", pa.int64()),
        ("suppressed_count", pa.int64()),
        ("first_suppressed_time_ms", pa.int64()),
        ("last_suppressed_time_ms", pa.int64()),
        ("suppressed_reason", pa.string()),
    ]
)
_REPLACE_MAX_ATTEMPTS = 6
_REPLACE_BASE_DELAY_S = 0.05
_REPLACE_MAX_DELAY_S = 1.5


def _backoff_delay(attempt: int) -> float:
    cap = min(_REPLACE_MAX_DELAY_S, _REPLACE_BASE_DELAY_S * 2**attempt)
    return random.uniform(0, cap)


def safe_replace(src: Path, dst: Path) -> None:
    last_exc: BaseException | None = None
    for attempt in range(_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(_backoff_delay(attempt))
        except FileNotFoundError as exc:
            last_exc = exc
            time.sleep(_backoff_delay(attempt))
    assert last_exc is not None
    raise last_exc


def _events_to_table(events: Iterable[FilteredEvent]) -> pa.Table:
    cols: dict[str, list] = {f.name: [] for f in PARQUET_SCHEMA}
    for e in events:
        cols["symbol"].append(e.symbol)
        cols["time_msc_raw"].append(e.time_msc_raw)
        cols["recv_time_utc_ms"].append(e.recv_time_utc_ms)
        cols["monotonic_ns"].append(e.monotonic_ns)
        cols["bid"].append(e.bid)
        cols["ask"].append(e.ask)
        cols["last"].append(e.last)
        cols["bid_scaled"].append(e.bid_scaled)
        cols["ask_scaled"].append(e.ask_scaled)
        cols["last_scaled"].append(e.last_scaled)
        cols["volume"].append(e.volume)
        cols["flags"].append(e.flags)
        cols["spread_points"].append(e.spread_points)
        cols["suppressed_count"].append(e.suppressed_count)
        cols["first_suppressed_time_ms"].append(e.first_suppressed_time_ms)
        cols["last_suppressed_time_ms"].append(e.last_suppressed_time_ms)
        cols["suppressed_reason"].append(e.suppressed_reason)
    return pa.Table.from_pydict(cols, schema=PARQUET_SCHEMA)


def _partition_dir(root: Path, symbol: str, recv_time_utc_ms: int) -> Path:
    dt = _dt.datetime.fromtimestamp(recv_time_utc_ms / 1000.0, tz=_dt.timezone.utc)
    return root / f"symbol={symbol}" / f"date={dt:%Y-%m-%d}" / f"hour={dt:%H}"


def _new_part_name(seq: int) -> str:
    return f"part-{seq:08d}-{secrets.token_hex(4)}.parquet"


@dataclass
class WriterMetrics:
    files_written: int = 0
    rows_written: int = 0
    bytes_written: int = 0
    replace_retries: int = 0
    flush_failures: int = 0
    last_failure_reason: str | None = None


@dataclass
class ParquetWriter:
    raw_dataset_dir: Path
    flush_max_rows: int
    flush_max_seconds: int
    compression: str = "zstd"
    before_flush_hook: Callable[[], None] | None = None
    metrics: WriterMetrics = field(default_factory=WriterMetrics)
    _buffer: list[FilteredEvent] = field(default_factory=list)
    _last_flush_monotonic: float = field(default_factory=time.monotonic)
    _seq: int = 0

    def __post_init__(self) -> None:
        self.raw_dataset_dir.mkdir(parents=True, exist_ok=True)

    def add(self, event: FilteredEvent) -> int:
        self._buffer.append(event)
        if self._should_flush_now():
            return self.flush()
        return 0

    def _should_flush_now(self) -> bool:
        if len(self._buffer) >= self.flush_max_rows:
            return True
        if time.monotonic() - self._last_flush_monotonic >= self.flush_max_seconds:
            return len(self._buffer) > 0
        return False

    def flush(self) -> int:
        if not self._buffer:
            self._last_flush_monotonic = time.monotonic()
            return 0
        if self.before_flush_hook is not None:
            self.before_flush_hook()
        pending = list(self._buffer)
        self._buffer.clear()
        partitions: dict[tuple[str, int], list[FilteredEvent]] = {}
        for ev in pending:
            hour_bucket = ev.recv_time_utc_ms // 3600000
            partitions.setdefault((ev.symbol, hour_bucket), []).append(ev)
        rows_total = 0
        published_event_ids: set[int] = set()
        try:
            for (symbol, _hour_bucket), events in partitions.items():
                part_dir = _partition_dir(self.raw_dataset_dir, symbol, events[0].recv_time_utc_ms)
                part_dir.mkdir(parents=True, exist_ok=True)
                self._seq += 1
                final_name = _new_part_name(self._seq)
                final_path = part_dir / final_name
                tmp_path = part_dir / (final_name + ".tmp")
                table = _events_to_table(events)
                pq.write_table(table, tmp_path, compression=self.compression)
                try:
                    safe_replace(tmp_path, final_path)
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
                self.metrics.files_written += 1
                self.metrics.rows_written += len(events)
                try:
                    self.metrics.bytes_written += final_path.stat().st_size
                except OSError:
                    pass
                rows_total += len(events)
                published_event_ids.update((id(e) for e in events))
        except Exception as exc:
            unpublished = [e for e in pending if id(e) not in published_event_ids]
            self._buffer = unpublished + self._buffer
            self.metrics.flush_failures += 1
            self.metrics.last_failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        self._last_flush_monotonic = time.monotonic()
        return rows_total

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
