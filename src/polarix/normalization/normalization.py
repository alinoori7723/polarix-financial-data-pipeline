from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.normalization import silver_paths
from polarix.orchestration.run_metadata import RunMetadata

NORMALIZER_VERSION = "0.1.0"
DATA_LAYER = "silver"


class NormalizationError(RuntimeError):
    pass


DEFAULT_JOIN_SAFE_THRESHOLD_MS = 50
LATENCY_OUTLIER_HIGH_MS = 1000
LATENCY_OUTLIER_LOW_MS = -250
SILVER_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("broker_server", pa.string()),
        ("broker_company", pa.string()),
        ("account_login_hash", pa.string()),
        ("source_date", pa.string()),
        ("source_hour", pa.int32()),
        ("time_msc_raw", pa.int64()),
        ("verified_offset_min", pa.int32()),
        ("verified_offset_ms", pa.int64()),
        ("time_msc_utc_ms", pa.int64()),
        ("recv_time_utc_ms", pa.int64()),
        ("residual_ms", pa.int64()),
        ("monotonic_ns", pa.int64()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("mid", pa.float64()),
        ("spread_price", pa.float64()),
        ("spread_points", pa.int64()),
        ("bid_scaled", pa.int64()),
        ("ask_scaled", pa.int64()),
        ("last", pa.float64()),
        ("last_scaled", pa.int64()),
        ("volume", pa.int64()),
        ("flags", pa.int64()),
        ("suppressed_count", pa.int64()),
        ("suppressed_reason", pa.string()),
        ("is_bid_ask_valid", pa.bool_()),
        ("is_scaled_bid_ask_valid", pa.bool_()),
        ("is_latency_outlier", pa.bool_()),
        ("is_join_safe", pa.bool_()),
        ("is_spread_valid", pa.bool_()),
        ("session_hour_utc", pa.int32()),
        ("data_layer", pa.string()),
        ("normalizer_version", pa.string()),
        ("metadata_source_path", pa.string()),
    ]
)


@dataclass
class NormalizationConfig:
    raw_root: Path
    silver_root: Path
    date: str
    metadata: RunMetadata
    join_safe_threshold_ms: int = DEFAULT_JOIN_SAFE_THRESHOLD_MS
    force: bool = False
    dry_run: bool = False
    run_id: Optional[str] = None
    run_window_start_utc: Optional[str] = None
    run_window_end_utc: Optional[str] = None

    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root).resolve()
        self.silver_root = Path(self.silver_root).resolve()
        if self.join_safe_threshold_ms <= 0:
            raise ValueError(
                f"join_safe_threshold_ms must be > 0 -- got {self.join_safe_threshold_ms}"
            )

    @property
    def is_run_scoped(self) -> bool:
        return self.run_id is not None

    @property
    def silver_layout_version(self) -> str:
        return (
            silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED
            if self.is_run_scoped
            else silver_paths.SILVER_LAYOUT_VERSION_LEGACY
        )


@dataclass
class SymbolPlan:
    symbol: str
    bronze_files: list[Path] = field(default_factory=list)
    silver_output: Optional[Path] = None
    rows_written: int = 0


@dataclass
class NormalizationResult:
    config: NormalizationConfig
    symbol_plans: list[SymbolPlan]
    manifest_path: Optional[Path]
    started_at_utc: str
    ended_at_utc: str
    metadata_source_path: str
    verified_offset_min: int
    join_safe_threshold_ms: int
    run_id: Optional[str] = None
    run_window_start_utc: Optional[str] = None
    run_window_end_utc: Optional[str] = None
    silver_layout_version: str = silver_paths.SILVER_LAYOUT_VERSION_LEGACY
    rows_excluded_by_window: int = 0


def _iso_to_ms(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    s = str(iso).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1000)


def _filter_bronze_to_window(
    bronze: pa.Table, start_ms: Optional[int], end_ms: Optional[int]
) -> pa.Table:
    import pyarrow.compute as pc

    if start_ms is None and end_ms is None:
        return bronze
    col = bronze.column("recv_time_utc_ms")
    mask = None
    if start_ms is not None:
        m = pc.greater_equal(col, pa.scalar(int(start_ms), pa.int64()))
        mask = m if mask is None else pc.and_(mask, m)
    if end_ms is not None:
        m = pc.less_equal(col, pa.scalar(int(end_ms), pa.int64()))
        mask = m if mask is None else pc.and_(mask, m)
    return bronze.filter(mask)


def _discover_symbols(raw_root: Path) -> list[str]:
    if not raw_root.exists():
        return []
    out: list[str] = []
    for child in raw_root.iterdir():
        if child.is_dir() and child.name.startswith("symbol="):
            out.append(child.name.split("=", 1)[1])
    out.sort()
    return out


def _discover_bronze_files(raw_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(
        str(raw_root), f"symbol={symbol}", f"date={date}", "hour=*", "part-*.parquet"
    )
    return sorted((Path(p) for p in glob.glob(pattern)))


def _extract_hour_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("hour="):
            return int(part.split("=", 1)[1])
    return -1


def _new_silver_part_name(seq: int) -> str:
    return f"part-{seq:08d}-{secrets.token_hex(4)}.parquet"


def _build_silver_table(
    bronze_table: pa.Table,
    symbol: str,
    source_date: str,
    source_hour: int,
    metadata: RunMetadata,
    join_safe_threshold_ms: int,
) -> pa.Table:
    import pyarrow.compute as pc

    verified_offset_min = metadata.verified_offset_min
    verified_offset_ms = verified_offset_min * 60 * 1000
    n = bronze_table.num_rows
    time_msc_raw = bronze_table.column("time_msc_raw")
    recv_time_utc_ms = bronze_table.column("recv_time_utc_ms")
    bid = bronze_table.column("bid")
    ask = bronze_table.column("ask")
    bid_scaled = bronze_table.column("bid_scaled")
    ask_scaled = bronze_table.column("ask_scaled")
    spread_points = bronze_table.column("spread_points")
    time_msc_utc_ms = pc.subtract(time_msc_raw, pa.scalar(verified_offset_ms, pa.int64()))
    residual_ms = pc.subtract(recv_time_utc_ms, time_msc_utc_ms)
    mid = pc.divide(pc.add(bid, ask), pa.scalar(2.0, pa.float64()))
    spread_price = pc.subtract(ask, bid)
    is_bid_ask_valid = pc.greater_equal(ask, bid)
    is_scaled_bid_ask_valid = pc.greater_equal(ask_scaled, bid_scaled)
    is_spread_valid = pc.and_(
        pc.greater_equal(spread_price, pa.scalar(0.0, pa.float64())),
        pc.greater_equal(spread_points, pa.scalar(0, pa.int64())),
    )
    is_latency_outlier = pc.or_(
        pc.less(residual_ms, pa.scalar(LATENCY_OUTLIER_LOW_MS, pa.int64())),
        pc.greater(residual_ms, pa.scalar(LATENCY_OUTLIER_HIGH_MS, pa.int64())),
    )
    is_join_safe = pc.and_(
        pc.greater_equal(residual_ms, pa.scalar(LATENCY_OUTLIER_LOW_MS, pa.int64())),
        pc.less_equal(residual_ms, pa.scalar(join_safe_threshold_ms, pa.int64())),
    )
    ms_per_hour = pa.scalar(3600000, pa.int64())
    hours_since_epoch = pc.divide(time_msc_utc_ms, ms_per_hour)
    session_hour_utc = pc.cast(
        pc.subtract(
            hours_since_epoch,
            pc.multiply(
                pc.divide(hours_since_epoch, pa.scalar(24, pa.int64())), pa.scalar(24, pa.int64())
            ),
        ),
        pa.int32(),
    )

    def _const_array(value, type_):
        return pa.array([value] * n, type=type_)

    cols = {
        "symbol": bronze_table.column("symbol").cast(pa.string()),
        "broker_server": _const_array(metadata.broker_server, pa.string()),
        "broker_company": _const_array(metadata.broker_company, pa.string()),
        "account_login_hash": _const_array(metadata.account_login_hash, pa.string()),
        "source_date": _const_array(source_date, pa.string()),
        "source_hour": _const_array(source_hour, pa.int32()),
        "time_msc_raw": time_msc_raw.cast(pa.int64()),
        "verified_offset_min": _const_array(verified_offset_min, pa.int32()),
        "verified_offset_ms": _const_array(verified_offset_ms, pa.int64()),
        "time_msc_utc_ms": time_msc_utc_ms.cast(pa.int64()),
        "recv_time_utc_ms": recv_time_utc_ms.cast(pa.int64()),
        "residual_ms": residual_ms.cast(pa.int64()),
        "monotonic_ns": bronze_table.column("monotonic_ns").cast(pa.int64()),
        "bid": bid.cast(pa.float64()),
        "ask": ask.cast(pa.float64()),
        "mid": mid.cast(pa.float64()),
        "spread_price": spread_price.cast(pa.float64()),
        "spread_points": spread_points.cast(pa.int64()),
        "bid_scaled": bid_scaled.cast(pa.int64()),
        "ask_scaled": ask_scaled.cast(pa.int64()),
        "last": bronze_table.column("last").cast(pa.float64()),
        "last_scaled": bronze_table.column("last_scaled").cast(pa.int64()),
        "volume": bronze_table.column("volume").cast(pa.int64()),
        "flags": bronze_table.column("flags").cast(pa.int64()),
        "suppressed_count": bronze_table.column("suppressed_count").cast(pa.int64()),
        "suppressed_reason": bronze_table.column("suppressed_reason").cast(pa.string()),
        "is_bid_ask_valid": is_bid_ask_valid.cast(pa.bool_()),
        "is_scaled_bid_ask_valid": is_scaled_bid_ask_valid.cast(pa.bool_()),
        "is_latency_outlier": is_latency_outlier.cast(pa.bool_()),
        "is_join_safe": is_join_safe.cast(pa.bool_()),
        "is_spread_valid": is_spread_valid.cast(pa.bool_()),
        "session_hour_utc": session_hour_utc.cast(pa.int32()),
        "data_layer": _const_array(DATA_LAYER, pa.string()),
        "normalizer_version": _const_array(NORMALIZER_VERSION, pa.string()),
        "metadata_source_path": _const_array(str(metadata.source_path), pa.string()),
    }
    return pa.Table.from_pydict(cols, schema=SILVER_SCHEMA)


def _read_bronze_file(path: Path) -> pa.Table:
    pf = pq.ParquetFile(path)
    return pf.read()


def _symbol_silver_output_dir(config: NormalizationConfig, symbol: str) -> Path:
    if config.is_run_scoped:
        assert config.run_id is not None
        return silver_paths.run_scoped_silver_dir(
            config.silver_root, symbol, config.date, config.run_id
        )
    return silver_paths.legacy_silver_dir(config.silver_root, symbol, config.date)


def plan_normalization(config: NormalizationConfig) -> list[SymbolPlan]:
    symbols = _discover_symbols(config.raw_root)
    plans: list[SymbolPlan] = []
    for symbol in symbols:
        files = _discover_bronze_files(config.raw_root, symbol, config.date)
        if not files:
            continue
        out_dir = _symbol_silver_output_dir(config, symbol)
        plans.append(SymbolPlan(symbol=symbol, bronze_files=files, silver_output=out_dir))
    return plans


def _git_hash() -> Optional[str]:
    try:
        import subprocess

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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _silver_output_already_exists(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any((p.name.startswith("part-") and p.suffix == ".parquet" for p in out_dir.iterdir()))


def normalize(config: NormalizationConfig) -> NormalizationResult:
    started_at_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    plans = plan_normalization(config)
    run_start_ms: Optional[int] = None
    run_end_ms: Optional[int] = None
    if config.is_run_scoped:
        run_start_ms = _iso_to_ms(config.run_window_start_utc)
        run_end_ms = _iso_to_ms(config.run_window_end_utc)
        if run_start_ms is None or run_end_ms is None:
            raise NormalizationError(
                f"run-scoped normalization for run_id={config.run_id!r} requires a resolvable run window; got start={config.run_window_start_utc!r} end={config.run_window_end_utc!r}. Refusing to write run-scoped Silver without a window -- it would risk cross-run contamination."
            )
        if run_end_ms < run_start_ms:
            raise NormalizationError(
                f"run window end ({config.run_window_end_utc}) precedes start ({config.run_window_start_utc}) for run_id={config.run_id!r}"
            )
    if config.dry_run:
        return NormalizationResult(
            config=config,
            symbol_plans=plans,
            manifest_path=None,
            started_at_utc=started_at_utc,
            ended_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            metadata_source_path=str(config.metadata.source_path),
            verified_offset_min=config.metadata.verified_offset_min,
            join_safe_threshold_ms=config.join_safe_threshold_ms,
            run_id=config.run_id,
            run_window_start_utc=config.run_window_start_utc,
            run_window_end_utc=config.run_window_end_utc,
            silver_layout_version=config.silver_layout_version,
        )
    if config.is_run_scoped:
        manifest_root = config.silver_root / f"date={config.date}" / f"run_id={config.run_id}"
    else:
        manifest_root = config.silver_root / f"date={config.date}"
    _ensure_dir(manifest_root)
    rows_excluded_by_window = 0
    seq = 0
    written_outputs: list[dict] = []
    for plan in plans:
        assert plan.silver_output is not None
        if _silver_output_already_exists(plan.silver_output) and (not config.force):
            raise FileExistsError(
                f"Silver output already exists at {plan.silver_output}; pass --force to overwrite"
            )
        if config.force and plan.silver_output.exists():
            for p in plan.silver_output.iterdir():
                if p.name.startswith("part-") and p.suffix == ".parquet":
                    p.unlink()
        _ensure_dir(plan.silver_output)
        by_hour: dict[int, list[Path]] = {}
        for f in plan.bronze_files:
            by_hour.setdefault(_extract_hour_from_path(f), []).append(f)
        rows_written_symbol = 0
        for hour in sorted(by_hour):
            chunks: list[pa.Table] = []
            for f in by_hour[hour]:
                bronze = _read_bronze_file(f)
                if bronze.num_rows == 0:
                    continue
                if config.is_run_scoped:
                    before = bronze.num_rows
                    bronze = _filter_bronze_to_window(bronze, run_start_ms, run_end_ms)
                    rows_excluded_by_window += before - bronze.num_rows
                    if bronze.num_rows == 0:
                        continue
                silver = _build_silver_table(
                    bronze,
                    symbol=plan.symbol,
                    source_date=config.date,
                    source_hour=hour,
                    metadata=config.metadata,
                    join_safe_threshold_ms=config.join_safe_threshold_ms,
                )
                chunks.append(silver)
            if not chunks:
                continue
            table = pa.concat_tables(chunks)
            seq += 1
            out_name = f"part-h{hour:02d}-{seq:08d}-{secrets.token_hex(4)}.parquet"
            out_path = plan.silver_output / out_name
            tmp_path = plan.silver_output / (out_name + ".tmp")
            pq.write_table(table, tmp_path, compression="zstd")
            os.replace(tmp_path, out_path)
            rows_written_symbol += table.num_rows
            written_outputs.append(
                {
                    "symbol": plan.symbol,
                    "hour": hour,
                    "output_file": str(out_path),
                    "row_count": table.num_rows,
                    "source_files": [str(p) for p in by_hour[hour]],
                }
            )
        plan.rows_written = rows_written_symbol
    manifest = {
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "date": config.date,
        "raw_root": str(config.raw_root),
        "silver_root": str(config.silver_root),
        "verified_offset_min": config.metadata.verified_offset_min,
        "verified_offset_ms": config.metadata.verified_offset_ms,
        "join_safe_threshold_ms": config.join_safe_threshold_ms,
        "metadata_source_path": str(config.metadata.source_path),
        "metadata_source_kind": config.metadata.source_kind,
        "selected_run_id": config.run_id,
        "selected_metadata_path": str(config.metadata.source_path),
        "run_start_utc": config.run_window_start_utc,
        "run_end_utc": config.run_window_end_utc,
        "output_layout_version": config.silver_layout_version,
        "silver_layout": "run_scoped" if config.is_run_scoped else "legacy_date_level",
        "rows_excluded_by_window": rows_excluded_by_window,
        "normalizer_version": NORMALIZER_VERSION,
        "code_version": _git_hash(),
        "data_layer": DATA_LAYER,
        "symbols": [p.symbol for p in plans],
        "source_files": [
            {"symbol": p.symbol, "files": [str(f) for f in p.bronze_files]} for p in plans
        ],
        "outputs": written_outputs,
        "rows_written_by_symbol": {p.symbol: p.rows_written for p in plans},
        "broker_metadata": dict(config.metadata.broker_metadata),
        "clock_status": dict(config.metadata.clock_status),
    }
    manifest_path = manifest_root / "normalization_manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return NormalizationResult(
        config=config,
        symbol_plans=plans,
        manifest_path=manifest_path,
        started_at_utc=started_at_utc,
        ended_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        metadata_source_path=str(config.metadata.source_path),
        verified_offset_min=config.metadata.verified_offset_min,
        join_safe_threshold_ms=config.join_safe_threshold_ms,
        run_id=config.run_id,
        run_window_start_utc=config.run_window_start_utc,
        run_window_end_utc=config.run_window_end_utc,
        silver_layout_version=config.silver_layout_version,
        rows_excluded_by_window=rows_excluded_by_window,
    )
