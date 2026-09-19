from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from polarix.ingestion.cme_databento_schema import (
    AGGRESSOR_BUY,
    AGGRESSOR_UNKNOWN,
    DATASET,
    SCHEMA,
    SIDE_BUY_AGGRESSOR,
    SIDE_SELL_AGGRESSOR,
    VENDOR,
    CmeColumnBinding,
    CmeSchemaError,
    detect_binding,
    map_aggressor,
    to_ns_int64,
)

INGEST_VERSION = "0.1.0"
DEFAULT_SYMBOLS = ("ES", "NQ")
REFERENCE_TRADES_SUBDIR = "reference_trades"
TRADE_ACTION = "T"
MISSING_SAMPLE_DATA = "MISSING_SAMPLE_DATA"
MISSING_CME_RAW_FOR_DATE = "MISSING_CME_RAW_FOR_DATE"
LEGACY_ROOT_FILES_INCLUDED = "LEGACY_ROOT_FILES_INCLUDED"
CROSS_DATE_CME_SOURCE_REJECTED = "CROSS_DATE_CME_SOURCE_REJECTED"
REFERENCE_TRADES_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("vendor", pa.string()),
        ("dataset", pa.string()),
        ("schema", pa.string()),
        ("symbol", pa.string()),
        ("raw_symbol", pa.string()),
        ("instrument_id", pa.int64()),
        ("ts_event", pa.int64()),
        ("ts_recv", pa.int64()),
        ("ts_event_ns", pa.int64()),
        ("ts_recv_ns", pa.int64()),
        ("event_time_utc_ns", pa.int64()),
        ("receive_time_utc_ns", pa.int64()),
        ("action", pa.string()),
        ("side_raw", pa.string()),
        ("aggressor_side", pa.string()),
        ("signed_size", pa.float64()),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("bid_px_00", pa.float64()),
        ("ask_px_00", pa.float64()),
        ("bid_sz_00", pa.int64()),
        ("ask_sz_00", pa.int64()),
        ("spread", pa.float64()),
        ("mid", pa.float64()),
        ("is_trade", pa.bool_()),
        ("is_aggressor_known", pa.bool_()),
        ("is_bbo_available", pa.bool_()),
        ("is_price_valid", pa.bool_()),
        ("is_size_valid", pa.bool_()),
        ("is_reference_trade_valid", pa.bool_()),
        ("ingest_date", pa.string()),
        ("ingest_version", pa.string()),
        ("source_file", pa.string()),
    ]
)


class CmeIngestError(RuntimeError):
    pass


@dataclass
class IngestConfig:
    input_root: Path
    output_root: Path
    reports_root: Path
    date: Optional[str] = None
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    schema: str = SCHEMA
    instrument_id_to_symbol: Mapping[int, str] = field(default_factory=dict)
    force: bool = False
    dry_run: bool = False
    include_legacy_root_files: bool = False

    def __post_init__(self) -> None:
        self.input_root = Path(self.input_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        if self.schema != SCHEMA:
            raise CmeIngestError(f"Phase 2A only supports schema={SCHEMA!r}; got {self.schema!r}")
        self.symbols = tuple(sorted({s.upper() for s in self.symbols}))


@dataclass
class IngestResult:
    config: IngestConfig
    source_files: list[Path]
    output_files_by_symbol: dict[str, list[Path]]
    row_counts_by_symbol: dict[str, int]
    manifest_path: Optional[Path]
    missing_sample: bool
    schema_error: Optional[str]
    rejected_rows_count: int
    unknown_side_count: int
    started_at_utc: str
    ended_at_utc: str
    binding: Optional[CmeColumnBinding]
    missing_reason: Optional[str] = None
    legacy_files_included: list[Path] = field(default_factory=list)
    rejected_cross_date_files: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class _DiscoveryResult:
    selected: list[Path]
    legacy_files_included: list[Path]
    rejected_cross_date_files: list[Path]


def _discover_sample_files(
    input_root: Path, *, date: Optional[str], include_legacy_root_files: bool = False
) -> _DiscoveryResult:
    if not input_root.exists():
        return _DiscoveryResult(selected=[], legacy_files_included=[], rejected_cross_date_files=[])
    if date is None:
        files = sorted(
            (
                Path(p)
                for p in glob.glob(os.path.join(str(input_root), "**", "*.parquet"), recursive=True)
            )
        )
        return _DiscoveryResult(
            selected=files, legacy_files_included=[], rejected_cross_date_files=[]
        )
    date_token = f"date={date}"
    date_dir = input_root / date_token
    partition_files = sorted((Path(p) for p in glob.glob(os.path.join(str(date_dir), "*.parquet"))))
    legacy_files: list[Path] = []
    if include_legacy_root_files:
        legacy_files = sorted(
            (Path(p) for p in glob.glob(os.path.join(str(input_root), "*.parquet")))
        )
    selected = list(partition_files) + list(legacy_files)
    rejected_cross_date: list[Path] = []
    for p in selected:
        for part in p.parts:
            if part.startswith("date=") and part != date_token:
                rejected_cross_date.append(p)
                break
    return _DiscoveryResult(
        selected=selected,
        legacy_files_included=legacy_files,
        rejected_cross_date_files=rejected_cross_date,
    )


def _resolve_symbol_column(
    table: pa.Table, binding: CmeColumnBinding, instrument_id_to_symbol: Mapping[int, str]
) -> pa.Array:
    if binding.raw_symbol is not None:
        col = table.column(binding.raw_symbol)
        return col.combine_chunks().cast(pa.string())
    assert binding.instrument_id is not None
    ids = table.column(binding.instrument_id).to_pylist()
    out = [instrument_id_to_symbol.get(int(i)) if i is not None else None for i in ids]
    return pa.array(out, type=pa.string())


def _filter_to_requested_symbols(
    raw_symbols: pa.Array, requested: tuple[str, ...]
) -> tuple[pa.Array, pa.Array]:
    n = len(raw_symbols)
    requested_sorted = sorted(requested, key=lambda s: -len(s))
    canon = [None] * n
    rs_list = raw_symbols.to_pylist()
    for i, rs in enumerate(rs_list):
        if rs is None:
            continue
        for ticker in requested_sorted:
            if rs == ticker or rs.startswith(ticker):
                canon[i] = ticker
                break
    canon_arr = pa.array(canon, type=pa.string())
    mask = pc.is_valid(canon_arr)
    return (mask, canon_arr)


def _git_hash() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _build_normalized_chunk(
    source_table: pa.Table,
    *,
    source_file: Path,
    binding: CmeColumnBinding,
    canonical_symbol: pa.Array,
    config: IngestConfig,
) -> pa.Table:
    n = source_table.num_rows
    ts_event_ns = to_ns_int64(source_table.column(binding.ts_event), name="ts_event")
    ts_recv_ns = to_ns_int64(source_table.column(binding.ts_recv), name="ts_recv")
    action = source_table.column(binding.action).cast(pa.string())
    side_raw = source_table.column(binding.side).cast(pa.string())
    price = source_table.column(binding.price).cast(pa.float64())
    size = source_table.column(binding.size).cast(pa.int64())
    side_list = side_raw.to_pylist()
    size_list = size.to_pylist()
    aggressor_list: list[str] = []
    signed_size_list: list[Optional[float]] = []
    for s, sz in zip(side_list, size_list):
        agg = map_aggressor(s)
        aggressor_list.append(agg)
        if sz is None or agg == AGGRESSOR_UNKNOWN:
            signed_size_list.append(None)
        elif agg == AGGRESSOR_BUY:
            signed_size_list.append(float(sz))
        else:
            signed_size_list.append(-float(sz))
    aggressor_side = pa.array(aggressor_list, type=pa.string())
    signed_size_arr = pa.array(signed_size_list, type=pa.float64())
    has_bbo = binding.is_bbo_available
    if has_bbo:
        bid_px = source_table.column(binding.bid_px).cast(pa.float64())
        ask_px = source_table.column(binding.ask_px).cast(pa.float64())
        bid_sz = source_table.column(binding.bid_sz).cast(pa.int64())
        ask_sz = source_table.column(binding.ask_sz).cast(pa.int64())
        mid = pc.divide(pc.add(bid_px, ask_px), pa.scalar(2.0, pa.float64()))
        spread = pc.subtract(ask_px, bid_px)
    else:
        bid_px = pa.nulls(n, type=pa.float64())
        ask_px = pa.nulls(n, type=pa.float64())
        bid_sz = pa.nulls(n, type=pa.int64())
        ask_sz = pa.nulls(n, type=pa.int64())
        mid = pa.nulls(n, type=pa.float64())
        spread = pa.nulls(n, type=pa.float64())
    is_trade = pc.equal(action, pa.scalar(TRADE_ACTION, pa.string()))
    is_aggressor_known = pc.or_(
        pc.equal(side_raw, pa.scalar(SIDE_BUY_AGGRESSOR, pa.string())),
        pc.equal(side_raw, pa.scalar(SIDE_SELL_AGGRESSOR, pa.string())),
    )
    is_price_valid = pc.and_(pc.is_valid(price), pc.greater(price, pa.scalar(0.0, pa.float64())))
    is_size_valid = pc.and_(pc.is_valid(size), pc.greater(size, pa.scalar(0, pa.int64())))
    is_reference_trade_valid = pc.and_(pc.and_(is_trade, is_price_valid), is_size_valid)
    if binding.instrument_id is not None:
        instrument_id = source_table.column(binding.instrument_id).cast(pa.int64())
    else:
        instrument_id = pa.nulls(n, type=pa.int64())
    if binding.raw_symbol is not None:
        raw_sym = source_table.column(binding.raw_symbol).cast(pa.string())
    else:
        raw_sym = canonical_symbol

    def _const(value, type_):
        return pa.array([value] * n, type=type_)

    ingest_date = config.date or _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    cols = {
        "source": _const("databento", pa.string()),
        "vendor": _const(VENDOR, pa.string()),
        "dataset": _const(DATASET, pa.string()),
        "schema": _const(SCHEMA, pa.string()),
        "symbol": canonical_symbol,
        "raw_symbol": raw_sym,
        "instrument_id": instrument_id,
        "ts_event": ts_event_ns,
        "ts_recv": ts_recv_ns,
        "ts_event_ns": ts_event_ns,
        "ts_recv_ns": ts_recv_ns,
        "event_time_utc_ns": ts_event_ns,
        "receive_time_utc_ns": ts_recv_ns,
        "action": action,
        "side_raw": side_raw,
        "aggressor_side": aggressor_side,
        "signed_size": signed_size_arr,
        "price": price,
        "size": size,
        "bid_px_00": bid_px,
        "ask_px_00": ask_px,
        "bid_sz_00": bid_sz,
        "ask_sz_00": ask_sz,
        "spread": spread,
        "mid": mid,
        "is_trade": is_trade,
        "is_aggressor_known": is_aggressor_known,
        "is_bbo_available": _const(has_bbo, pa.bool_()),
        "is_price_valid": is_price_valid,
        "is_size_valid": is_size_valid,
        "is_reference_trade_valid": is_reference_trade_valid,
        "ingest_date": _const(ingest_date, pa.string()),
        "ingest_version": _const(INGEST_VERSION, pa.string()),
        "source_file": _const(str(source_file), pa.string()),
    }
    return pa.Table.from_pydict(cols, schema=REFERENCE_TRADES_SCHEMA)


def _silver_output_dir(output_root: Path, symbol: str, date: str) -> Path:
    return output_root / REFERENCE_TRADES_SUBDIR / f"symbol={symbol}" / f"date={date}"


def _existing_part_files(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return sorted(
        (p for p in out_dir.iterdir() if p.name.startswith("part-") and p.suffix == ".parquet")
    )


def ingest(config: IngestConfig) -> IngestResult:
    started_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    discovery = _discover_sample_files(
        config.input_root,
        date=config.date,
        include_legacy_root_files=config.include_legacy_root_files,
    )
    if discovery.rejected_cross_date_files:
        raise CmeIngestError(
            f"{CROSS_DATE_CME_SOURCE_REJECTED}: at least one selected source file belongs to a different date partition than --date={config.date!r}. Rejected files: {[str(p) for p in discovery.rejected_cross_date_files]}"
        )
    source_files = discovery.selected
    if not source_files:
        ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        missing_reason = (
            MISSING_CME_RAW_FOR_DATE if config.date is not None else MISSING_SAMPLE_DATA
        )
        return IngestResult(
            config=config,
            source_files=[],
            output_files_by_symbol={},
            row_counts_by_symbol={},
            manifest_path=None,
            missing_sample=True,
            schema_error=None,
            rejected_rows_count=0,
            unknown_side_count=0,
            started_at_utc=started_at_utc,
            ended_at_utc=ended,
            binding=None,
            missing_reason=missing_reason,
            legacy_files_included=list(discovery.legacy_files_included),
            rejected_cross_date_files=list(discovery.rejected_cross_date_files),
        )
    first = pq.ParquetFile(source_files[0]).read()
    try:
        binding = detect_binding(
            list(first.column_names), instrument_id_to_symbol=config.instrument_id_to_symbol
        )
    except CmeSchemaError as exc:
        return IngestResult(
            config=config,
            source_files=source_files,
            output_files_by_symbol={},
            row_counts_by_symbol={},
            manifest_path=None,
            missing_sample=False,
            schema_error=str(exc),
            rejected_rows_count=0,
            unknown_side_count=0,
            started_at_utc=started_at_utc,
            ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            binding=None,
            missing_reason=None,
            legacy_files_included=list(discovery.legacy_files_included),
            rejected_cross_date_files=list(discovery.rejected_cross_date_files),
        )
    all_chunks: list[pa.Table] = []
    rejected_rows_count = 0
    unknown_side_count = 0
    for path in source_files:
        table = pq.ParquetFile(path).read()
        raw_symbol_arr = _resolve_symbol_column(table, binding, config.instrument_id_to_symbol)
        symbol_mask, canonical_symbol = _filter_to_requested_symbols(raw_symbol_arr, config.symbols)
        action_mask = pc.equal(
            table.column(binding.action).cast(pa.string()), pa.scalar(TRADE_ACTION, pa.string())
        )
        keep_mask = pc.and_(symbol_mask, action_mask)
        rejected_rows_count += int(
            table.num_rows - pc.sum(pc.cast(keep_mask, pa.int64())).as_py() or 0
        )
        kept_table = table.filter(keep_mask)
        if kept_table.num_rows == 0:
            continue
        kept_canonical = canonical_symbol.filter(keep_mask)
        chunk = _build_normalized_chunk(
            kept_table,
            source_file=path,
            binding=binding,
            canonical_symbol=kept_canonical,
            config=config,
        )
        unknown_side_count += int(
            pc.sum(
                pc.cast(
                    pc.equal(
                        chunk.column("aggressor_side"), pa.scalar(AGGRESSOR_UNKNOWN, pa.string())
                    ),
                    pa.int64(),
                )
            ).as_py()
            or 0
        )
        all_chunks.append(chunk)
    output_files_by_symbol: dict[str, list[Path]] = {s: [] for s in config.symbols}
    row_counts_by_symbol: dict[str, int] = {s: 0 for s in config.symbols}
    if config.dry_run:
        for chunk in all_chunks:
            symbols_in_chunk = set(chunk.column("symbol").to_pylist())
            for s in symbols_in_chunk:
                if s is None:
                    continue
                row_counts_by_symbol[s] = row_counts_by_symbol.get(s, 0) + int(
                    pc.sum(
                        pc.cast(
                            pc.equal(chunk.column("symbol"), pa.scalar(s, pa.string())), pa.int64()
                        )
                    ).as_py()
                    or 0
                )
        return IngestResult(
            config=config,
            source_files=source_files,
            output_files_by_symbol=output_files_by_symbol,
            row_counts_by_symbol=row_counts_by_symbol,
            manifest_path=None,
            missing_sample=False,
            schema_error=None,
            rejected_rows_count=rejected_rows_count,
            unknown_side_count=unknown_side_count,
            started_at_utc=started_at_utc,
            ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            binding=binding,
            missing_reason=None,
            legacy_files_included=list(discovery.legacy_files_included),
            rejected_cross_date_files=list(discovery.rejected_cross_date_files),
        )
    date = config.date or _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    manifest_dir = config.output_root / REFERENCE_TRADES_SUBDIR / f"date={date}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if not all_chunks:
        full = REFERENCE_TRADES_SCHEMA.empty_table()
    else:
        full = pa.concat_tables(all_chunks)
    for symbol in config.symbols:
        out_dir = _silver_output_dir(config.output_root, symbol, date)
        if _existing_part_files(out_dir) and (not config.force):
            raise FileExistsError(
                f"reference_trades for symbol={symbol} date={date} already exist; pass --force to overwrite"
            )
        if config.force and out_dir.exists():
            for p in out_dir.iterdir():
                if p.name.startswith("part-") and p.suffix == ".parquet":
                    p.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        sym_mask = pc.equal(full.column("symbol"), pa.scalar(symbol, pa.string()))
        sym_table = full.filter(sym_mask)
        if sym_table.num_rows == 0:
            continue
        seq = 1
        out_name = f"part-{seq:08d}-{secrets.token_hex(4)}.parquet"
        out_path = out_dir / out_name
        tmp_path = out_dir / (out_name + ".tmp")
        pq.write_table(sym_table, tmp_path, compression="zstd")
        os.replace(tmp_path, out_path)
        output_files_by_symbol[symbol].append(out_path)
        row_counts_by_symbol[symbol] = sym_table.num_rows
    manifest = {
        "created_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "source": "databento",
        "vendor": VENDOR,
        "dataset": DATASET,
        "schema": SCHEMA,
        "ingest_version": INGEST_VERSION,
        "code_version": _git_hash(),
        "date": date,
        "source_date": date,
        "input_root": str(config.input_root),
        "include_legacy_root_files": bool(config.include_legacy_root_files),
        "legacy_files_included": [str(p) for p in discovery.legacy_files_included],
        "rejected_cross_date_files": [str(p) for p in discovery.rejected_cross_date_files],
        "isolation_warnings": [LEGACY_ROOT_FILES_INCLUDED]
        if discovery.legacy_files_included
        else [],
        "symbols_requested": list(config.symbols),
        "symbols_with_rows": [s for s, c in row_counts_by_symbol.items() if c > 0],
        "source_files": [str(p) for p in source_files],
        "output_files": {s: [str(p) for p in fs] for s, fs in output_files_by_symbol.items()},
        "row_counts_by_symbol": row_counts_by_symbol,
        "required_columns_present": True,
        "rejected_rows_count": rejected_rows_count,
        "unknown_side_count": unknown_side_count,
        "binding": {
            "ts_event": binding.ts_event,
            "ts_recv": binding.ts_recv,
            "action": binding.action,
            "side": binding.side,
            "price": binding.price,
            "size": binding.size,
            "raw_symbol": binding.raw_symbol,
            "instrument_id": binding.instrument_id,
            "bid_px": binding.bid_px,
            "ask_px": binding.ask_px,
            "bid_sz": binding.bid_sz,
            "ask_sz": binding.ask_sz,
            "is_bbo_available": binding.is_bbo_available,
        },
    }
    manifest_path = manifest_dir / "reference_trades_manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return IngestResult(
        config=config,
        source_files=source_files,
        output_files_by_symbol=output_files_by_symbol,
        row_counts_by_symbol=row_counts_by_symbol,
        manifest_path=manifest_path,
        missing_sample=False,
        schema_error=None,
        rejected_rows_count=rejected_rows_count,
        unknown_side_count=unknown_side_count,
        started_at_utc=started_at_utc,
        ended_at_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        binding=binding,
        missing_reason=None,
        legacy_files_included=list(discovery.legacy_files_included),
        rejected_cross_date_files=list(discovery.rejected_cross_date_files),
    )
