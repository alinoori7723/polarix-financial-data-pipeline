from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.ingestion.parquet_writer import PARQUET_SCHEMA, safe_replace

DEFAULT_CHUNK_FILES = 80
PART_FILENAME_RE = re.compile("^part-\\d{8}-[0-9a-f]+\\.parquet$")
COMPACTED_FILENAME_RE = re.compile("^part-compacted-\\d{3}\\.parquet$")


@dataclass(frozen=True)
class PartitionKey:
    symbol: str
    date: str

    def raw_dir(self, raw_root: Path) -> Path:
        return raw_root / f"symbol={self.symbol}" / f"date={self.date}"

    def compacted_dir(self, compacted_root: Path) -> Path:
        return compacted_root / f"symbol={self.symbol}" / f"date={self.date}"


@dataclass
class CompactionResult:
    partition: PartitionKey
    source_files: int
    source_rows: int
    output_files: int
    output_rows: int
    min_recv_time_utc_ms: int | None
    max_recv_time_utc_ms: int | None
    sources_deleted: int
    sources_left_due_to_lock: int
    validation_passed: bool
    error: str | None = None


def _chunked(iterable: list[Path], n: int) -> Iterator[list[Path]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def _list_raw_part_files(part_dir: Path) -> list[Path]:
    if not part_dir.exists():
        return []
    out: list[Path] = []
    for child in part_dir.iterdir():
        if child.is_file() and PART_FILENAME_RE.match(child.name):
            out.append(child)
        elif child.is_dir() and child.name.startswith("hour="):
            for f in child.iterdir():
                if f.is_file() and PART_FILENAME_RE.match(f.name):
                    out.append(f)
    out.sort()
    return out


def _today_utc_date() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def discover_closed_partitions(raw_root: Path, today_utc: str | None = None) -> list[PartitionKey]:
    today_utc = today_utc or _today_utc_date()
    out: list[PartitionKey] = []
    if not raw_root.exists():
        return out
    for symbol_dir in raw_root.iterdir():
        if not symbol_dir.is_dir() or not symbol_dir.name.startswith("symbol="):
            continue
        symbol = symbol_dir.name[len("symbol=") :]
        for date_dir in symbol_dir.iterdir():
            if not date_dir.is_dir() or not date_dir.name.startswith("date="):
                continue
            date = date_dir.name[len("date=") :]
            if date < today_utc:
                out.append(PartitionKey(symbol=symbol, date=date))
    out.sort(key=lambda p: (p.symbol, p.date))
    return out


def _try_delete(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except PermissionError:
        return False
    except FileNotFoundError:
        return True


def compact_partition(
    partition: PartitionKey,
    raw_root: Path,
    compacted_root: Path,
    chunk_files: int = DEFAULT_CHUNK_FILES,
    compression: str = "zstd",
    delete_sources_on_success: bool = True,
) -> CompactionResult:
    source_dir = partition.raw_dir(raw_root)
    out_dir = partition.compacted_dir(compacted_root)
    source_files = _list_raw_part_files(source_dir)
    if not source_files:
        return CompactionResult(
            partition=partition,
            source_files=0,
            source_rows=0,
            output_files=0,
            output_rows=0,
            min_recv_time_utc_ms=None,
            max_recv_time_utc_ms=None,
            sources_deleted=0,
            sources_left_due_to_lock=0,
            validation_passed=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    min_ts: int | None = None
    max_ts: int | None = None
    for f in source_files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as exc:
            return CompactionResult(
                partition=partition,
                source_files=len(source_files),
                source_rows=source_rows,
                output_files=0,
                output_rows=0,
                min_recv_time_utc_ms=min_ts,
                max_recv_time_utc_ms=max_ts,
                sources_deleted=0,
                sources_left_due_to_lock=0,
                validation_passed=False,
                error=f"failed to open {f.name}: {exc!r}",
            )
        source_rows += pf.metadata.num_rows
        rg_min: int | None = None
        rg_max: int | None = None
        for rg_idx in range(pf.metadata.num_row_groups):
            col_idx = pf.schema_arrow.get_field_index("recv_time_utc_ms")
            stats = pf.metadata.row_group(rg_idx).column(col_idx).statistics
            if stats is not None and stats.has_min_max:
                lo = int(stats.min)
                hi = int(stats.max)
                rg_min = lo if rg_min is None else min(rg_min, lo)
                rg_max = hi if rg_max is None else max(rg_max, hi)
            else:
                tbl = pf.read_row_group(rg_idx, columns=["recv_time_utc_ms"])
                arr = tbl.column("recv_time_utc_ms").to_pylist()
                if arr:
                    lo = min(arr)
                    hi = max(arr)
                    rg_min = lo if rg_min is None else min(rg_min, lo)
                    rg_max = hi if rg_max is None else max(rg_max, hi)
        if rg_min is not None:
            min_ts = rg_min if min_ts is None else min(min_ts, rg_min)
        if rg_max is not None:
            max_ts = rg_max if max_ts is None else max(max_ts, rg_max)
    output_rows = 0
    output_files = 0
    for idx, chunk in enumerate(_chunked(source_files, chunk_files)):
        tables: list[pa.Table] = []
        for f in chunk:
            t = pq.read_table(f, schema=PARQUET_SCHEMA)
            tables.append(t)
        if not tables:
            continue
        combined = pa.concat_tables(tables, promote_options="default")
        combined = combined.sort_by(
            [
                ("symbol", "ascending"),
                ("recv_time_utc_ms", "ascending"),
                ("monotonic_ns", "ascending"),
            ]
        )
        final_name = f"part-compacted-{idx:03d}.parquet"
        final_path = out_dir / final_name
        tmp_path = out_dir / (final_name + ".tmp")
        pq.write_table(combined, tmp_path, compression=compression)
        safe_replace(tmp_path, final_path)
        output_rows += combined.num_rows
        output_files += 1
    validation_passed = output_rows == source_rows
    if validation_passed and source_rows > 0 and (min_ts is None or max_ts is None):
        validation_passed = False
    sources_deleted = 0
    sources_left_due_to_lock = 0
    error: str | None = None
    if not validation_passed:
        error = f"row_count_mismatch: source={source_rows} output={output_rows}"
    elif delete_sources_on_success:
        for f in source_files:
            if _try_delete(f):
                sources_deleted += 1
            else:
                sources_left_due_to_lock += 1
    return CompactionResult(
        partition=partition,
        source_files=len(source_files),
        source_rows=source_rows,
        output_files=output_files,
        output_rows=output_rows,
        min_recv_time_utc_ms=min_ts,
        max_recv_time_utc_ms=max_ts,
        sources_deleted=sources_deleted,
        sources_left_due_to_lock=sources_left_due_to_lock,
        validation_passed=validation_passed,
        error=error,
    )


def compact_all_closed(
    raw_root: Path,
    compacted_root: Path,
    chunk_files: int = DEFAULT_CHUNK_FILES,
    compression: str = "zstd",
    today_utc: str | None = None,
) -> list[CompactionResult]:
    out: list[CompactionResult] = []
    for partition in discover_closed_partitions(raw_root, today_utc=today_utc):
        out.append(
            compact_partition(
                partition=partition,
                raw_root=raw_root,
                compacted_root=compacted_root,
                chunk_files=chunk_files,
                compression=compression,
            )
        )
    return out
