from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.normalization.normalization import (
    DEFAULT_JOIN_SAFE_THRESHOLD_MS,
    NORMALIZER_VERSION,
    NormalizationConfig,
    normalize,
)
from polarix.orchestration.run_metadata import RunMetadata

BRONZE_FIELDS = [
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
BRONZE_SCHEMA = pa.schema(BRONZE_FIELDS)


def _make_bronze_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [] for name, _ in BRONZE_FIELDS}
    for r in rows:
        for name, _ in BRONZE_FIELDS:
            cols[name].append(r.get(name))
    table = pa.Table.from_pydict(cols, schema=BRONZE_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def _make_metadata(tmp_path: Path, verified_offset_min: int) -> RunMetadata:
    src = tmp_path / "fake_summary.json"
    src.write_text("{}", encoding="utf-8")
    return RunMetadata(
        source_path=src,
        source_kind="live_run_summary",
        run_id="testrun",
        verified_offset_min=verified_offset_min,
        timestamp_semantics_status="OFFSET_VERIFIED_FOR_SESSION",
        clock_status={},
        broker_metadata={
            "account_company": "TestBroker",
            "account_server": "Test-Server 1",
            "account_login_hash": "deadbeef",
        },
        run_started_at_utc=None,
        run_ended_at_utc=None,
        final_decision="PASS",
        symbols=("SPX500", "NDX100"),
    )


def _seed_bronze(raw_root: Path, symbol: str, date: str, hour: int, rows: list[dict]) -> Path:
    part_dir = raw_root / f"symbol={symbol}" / f"date={date}" / f"hour={hour:02d}"
    out = part_dir / "part-00000001-aaaa.parquet"
    _make_bronze_file(out, rows)
    return out


def _row(
    *,
    symbol: str = "SPX500",
    time_msc_raw: int,
    recv_time_utc_ms: int,
    bid: float = 100.0,
    ask: float = 100.5,
    bid_scaled: int = 10000,
    ask_scaled: int = 10050,
    last: float = 0.0,
    last_scaled: int = 0,
    volume: int = 0,
    flags: int = 0,
    spread_points: int = 5,
    suppressed_count: int = 0,
    monotonic_ns: int = 0,
    suppressed_reason: str | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "time_msc_raw": time_msc_raw,
        "recv_time_utc_ms": recv_time_utc_ms,
        "monotonic_ns": monotonic_ns,
        "bid": bid,
        "ask": ask,
        "last": last,
        "bid_scaled": bid_scaled,
        "ask_scaled": ask_scaled,
        "last_scaled": last_scaled,
        "volume": volume,
        "flags": flags,
        "spread_points": spread_points,
        "suppressed_count": suppressed_count,
        "first_suppressed_time_ms": None,
        "last_suppressed_time_ms": None,
        "suppressed_reason": suppressed_reason,
    }


def test_time_msc_utc_ms_uses_dynamic_offset(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=120)
    offset_ms = 120 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000010)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["time_msc_utc_ms"].to_list() == [1000000000000]
    assert df["verified_offset_min"].to_list() == [120]
    assert df["verified_offset_ms"].to_list() == [offset_ms]


def test_dynamic_offset_120(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=120)
    offset_ms = 120 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=2000000000000 + offset_ms, recv_time_utc_ms=2000000000005)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["residual_ms"].to_list() == [5]


def test_residual_ms_computed_correctly(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000020),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000999),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=999999999900),
    ]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["residual_ms"].to_list() == [20, 999, -100]


def test_mid_and_spread_price(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000, recv_time_utc_ms=1000000000000, bid=100.0, ask=101.0)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["mid"].to_list() == [100.5]
    assert df["spread_price"].to_list() == [1.0]


def test_is_join_safe_only_when_residual_at_or_below_50ms(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000000),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000050),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000051),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000999),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=999999999700),
    ]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["is_join_safe"].to_list() == [True, True, False, False, False]


def test_residual_999_not_fatal_but_not_join_safe(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000999)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["is_join_safe"][0] is False
    assert df["is_latency_outlier"][0] is False
    assert df["is_bid_ask_valid"][0] is True
    assert len(df) == 1


def test_residual_above_1000_is_latency_outlier(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000001001),
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=999999999700),
    ]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert df["is_latency_outlier"].to_list() == [True, True]


def test_preserves_unsafe_rows(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [
        _row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000000),
        _row(
            time_msc_raw=1000000000000 + offset_ms,
            recv_time_utc_ms=1000000001500,
            bid=100.0,
            ask=99.0,
            bid_scaled=10000,
            ask_scaled=9900,
            spread_points=-1,
        ),
    ]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    df = pl.read_parquet(silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet")
    assert len(df) == 2
    bad = df.row(1, named=True)
    assert bad["is_bid_ask_valid"] is False
    assert bad["is_latency_outlier"] is True
    assert bad["is_spread_valid"] is False


def test_silver_readable_by_duckdb_and_polars(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000010)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    silver_glob = silver_root / "symbol=SPX500" / "date=2026-05-18" / "*.parquet"
    df = pl.read_parquet(silver_glob)
    assert df.height == 1
    import duckdb

    con = duckdb.connect(":memory:")
    glob_str = str(silver_glob).replace("\\", "/")
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_str}')").fetchone()[0]
    con.close()
    assert cnt == 1


def test_does_not_mutate_bronze(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000 + offset_ms, recv_time_utc_ms=1000000000010)]
    bronze_path = _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    before_bytes = bronze_path.read_bytes()
    before_mtime = bronze_path.stat().st_mtime
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    after_bytes = bronze_path.read_bytes()
    after_mtime = bronze_path.stat().st_mtime
    assert before_bytes == after_bytes
    assert before_mtime == after_mtime


def test_normalization_manifest_contents(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000, recv_time_utc_ms=1000000000000)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    result = normalize(cfg)
    manifest_path = result.manifest_path
    assert manifest_path is not None and manifest_path.exists()
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["verified_offset_min"] == 60
    assert doc["join_safe_threshold_ms"] == DEFAULT_JOIN_SAFE_THRESHOLD_MS
    assert doc["metadata_source_path"] == str(md.source_path)
    assert doc["normalizer_version"] == NORMALIZER_VERSION
    assert "SPX500" in doc["rows_written_by_symbol"]


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000, recv_time_utc_ms=1000000000000)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=True
    )
    normalize(cfg)
    cfg2 = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, force=False
    )
    with pytest.raises(FileExistsError):
        normalize(cfg2)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    raw_root = tmp_path / "raw"
    silver_root = tmp_path / "silver"
    rows = [_row(time_msc_raw=1000000000000, recv_time_utc_ms=1000000000000)]
    _seed_bronze(raw_root, "SPX500", "2026-05-18", 5, rows)
    _seed_bronze(raw_root, "NDX100", "2026-05-18", 5, rows)
    cfg = NormalizationConfig(
        raw_root=raw_root, silver_root=silver_root, date="2026-05-18", metadata=md, dry_run=True
    )
    result = normalize(cfg)
    assert result.manifest_path is None
    assert not list(silver_root.rglob("*.parquet"))
