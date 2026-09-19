from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.ingestion.cme_databento_schema import DATASET, SCHEMA, VENDOR
from polarix.ingestion.cme_reference_ingest import (
    REFERENCE_TRADES_SUBDIR,
    IngestConfig,
    ingest,
)


def _make_mbp1_fixture(
    path: Path,
    *,
    include_bbo: bool = True,
    with_instrument_id: bool = False,
    rows: list[dict] | None = None,
) -> Path:
    if rows is None:
        rows = [
            {
                "ts_event": 1700000000000000000,
                "ts_recv": 1700000000000000010,
                "raw_symbol": "ESM6",
                "instrument_id": 1,
                "action": "T",
                "side": "B",
                "price": 5000.0,
                "size": 5,
                "bid_px_00": 4999.75,
                "ask_px_00": 5000.25,
                "bid_sz_00": 10,
                "ask_sz_00": 7,
                "bid_ct_00": 1,
                "ask_ct_00": 1,
            },
            {
                "ts_event": 1700000000000000100,
                "ts_recv": 1700000000000000110,
                "raw_symbol": "ESM6",
                "instrument_id": 1,
                "action": "T",
                "side": "A",
                "price": 5000.25,
                "size": 2,
                "bid_px_00": 4999.75,
                "ask_px_00": 5000.25,
                "bid_sz_00": 9,
                "ask_sz_00": 6,
                "bid_ct_00": 1,
                "ask_ct_00": 1,
            },
            {
                "ts_event": 1700000000000000200,
                "ts_recv": 1700000000000000210,
                "raw_symbol": "ESM6",
                "instrument_id": 1,
                "action": "T",
                "side": "N",
                "price": 5000.0,
                "size": 1,
                "bid_px_00": 4999.75,
                "ask_px_00": 5000.25,
                "bid_sz_00": 9,
                "ask_sz_00": 6,
                "bid_ct_00": 1,
                "ask_ct_00": 1,
            },
            {
                "ts_event": 1700000000000000300,
                "ts_recv": 1700000000000000310,
                "raw_symbol": "ESM6",
                "instrument_id": 1,
                "action": "C",
                "side": "B",
                "price": 5000.0,
                "size": 0,
                "bid_px_00": 4999.75,
                "ask_px_00": 5000.25,
                "bid_sz_00": 9,
                "ask_sz_00": 6,
                "bid_ct_00": 1,
                "ask_ct_00": 1,
            },
            {
                "ts_event": 1700000000000000400,
                "ts_recv": 1700000000000000410,
                "raw_symbol": "NQM6",
                "instrument_id": 2,
                "action": "T",
                "side": "B",
                "price": 18000.0,
                "size": 3,
                "bid_px_00": 17999.5,
                "ask_px_00": 18000.0,
                "bid_sz_00": 5,
                "ask_sz_00": 4,
                "bid_ct_00": 1,
                "ask_ct_00": 1,
            },
        ]
    fields = [
        ("ts_event", pa.int64()),
        ("ts_recv", pa.int64()),
        ("raw_symbol", pa.string()),
        ("instrument_id", pa.int64()),
        ("action", pa.string()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("size", pa.int64()),
    ]
    if include_bbo:
        fields += [
            ("bid_px_00", pa.float64()),
            ("ask_px_00", pa.float64()),
            ("bid_sz_00", pa.int64()),
            ("ask_sz_00", pa.int64()),
            ("bid_ct_00", pa.int64()),
            ("ask_ct_00", pa.int64()),
        ]
    schema = pa.schema(fields)
    cols: dict[str, list] = {name: [] for name, _ in fields}
    for r in rows:
        for name, _ in fields:
            cols[name].append(r.get(name))
        if not with_instrument_id:
            pass
    table = pa.Table.from_pydict(cols, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def _make_config(tmp_path: Path, *, force: bool = True, dry_run: bool = False) -> IngestConfig:
    return IngestConfig(
        input_root=tmp_path / "raw" / "cme_sample",
        output_root=tmp_path / "normalized" / "cme_reference",
        reports_root=tmp_path / "reports",
        date="2026-05-18",
        force=force,
        dry_run=dry_run,
    )


def test_ingest_reads_synthetic_fixture(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    result = ingest(cfg)
    assert not result.missing_sample
    assert result.schema_error is None
    assert result.row_counts_by_symbol == {"ES": 3, "NQ": 1}


def test_ingest_filters_to_action_T(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    df = pl.read_parquet(es_files[0])
    assert set(df["action"].to_list()) == {"T"}


def test_ingest_preserves_ts_event_ts_recv(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    df = pl.read_parquet(es_files[0])
    assert df["ts_event"].to_list() == [
        1700000000000000000,
        1700000000000000100,
        1700000000000000200,
    ]
    assert df["ts_recv"].to_list() == [
        1700000000000000010,
        1700000000000000110,
        1700000000000000210,
    ]


def test_ingest_stores_timestamps_as_int64_ns(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    table = pq.ParquetFile(es_files[0]).read()
    assert table.schema.field("ts_event_ns").type == pa.int64()
    assert table.schema.field("event_time_utc_ns").type == pa.int64()
    assert table.schema.field("receive_time_utc_ns").type == pa.int64()


def test_ingest_accepts_utc_timestamp_columns(tmp_path: Path) -> None:
    fpath = tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "ts_tz.parquet"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    dts = [
        _dt.datetime(2026, 5, 18, 12, 0, 0, tzinfo=_dt.timezone.utc),
        _dt.datetime(2026, 5, 18, 12, 0, 0, 100000, tzinfo=_dt.timezone.utc),
    ]
    schema = pa.schema(
        [
            ("ts_event", pa.timestamp("ns", tz="UTC")),
            ("ts_recv", pa.timestamp("ns", tz="UTC")),
            ("raw_symbol", pa.string()),
            ("action", pa.string()),
            ("side", pa.string()),
            ("price", pa.float64()),
            ("size", pa.int64()),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "ts_event": dts,
            "ts_recv": dts,
            "raw_symbol": ["ESM6", "ESM6"],
            "action": ["T", "T"],
            "side": ["B", "A"],
            "price": [5000.0, 5000.25],
            "size": [1, 2],
        },
        schema=schema,
    )
    pq.write_table(table, fpath, compression="zstd")
    cfg = _make_config(tmp_path)
    result = ingest(cfg)
    assert result.row_counts_by_symbol["ES"] == 2


def test_ingest_computes_mid_spread_when_bbo_available(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    df = pl.read_parquet(es_files[0])
    assert df["is_bbo_available"].to_list() == [True, True, True]
    assert df["mid"].to_list() == [5000.0, 5000.0, 5000.0]
    assert df["spread"].to_list() == [0.5, 0.5, 0.5]


def test_ingest_marks_bbo_missing_when_columns_absent(tmp_path: Path) -> None:
    _make_mbp1_fixture(
        tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet", include_bbo=False
    )
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    df = pl.read_parquet(es_files[0])
    assert df["is_bbo_available"].to_list() == [False, False, False]
    assert df["bid_px_00"].null_count() == 3
    assert df["mid"].null_count() == 3


def test_silver_readable_by_polars_and_duckdb(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    assert es_files
    df = pl.read_parquet(es_files[0])
    assert df.height == 3
    import duckdb

    con = duckdb.connect(":memory:")
    glob_str = str(es_files[0]).replace("\\", "/")
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_str}')").fetchone()[0]
    con.close()
    assert cnt == 3


def test_ingest_does_not_mutate_input(tmp_path: Path) -> None:
    fpath = tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet"
    _make_mbp1_fixture(fpath)
    before_bytes = fpath.read_bytes()
    before_mtime = fpath.stat().st_mtime
    cfg = _make_config(tmp_path)
    ingest(cfg)
    assert fpath.read_bytes() == before_bytes
    assert fpath.stat().st_mtime == before_mtime


def test_ingest_fails_clearly_when_sample_missing(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    result = ingest(cfg)
    assert result.missing_sample is True
    assert result.row_counts_by_symbol == {}
    assert result.manifest_path is None


def test_ingest_aggressor_sign_in_output(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    ingest(cfg)
    es_files = list(
        (cfg.output_root / REFERENCE_TRADES_SUBDIR / "symbol=ES" / "date=2026-05-18").glob(
            "*.parquet"
        )
    )
    df = pl.read_parquet(es_files[0]).sort("ts_event")
    sides = df["side_raw"].to_list()
    aggressors = df["aggressor_side"].to_list()
    signed = df["signed_size"].to_list()
    assert sides == ["B", "A", "N"]
    assert aggressors == ["BUY", "SELL", "UNKNOWN"]
    assert signed[0] == 5.0
    assert signed[1] == -2.0
    assert signed[2] is None


def test_manifest_contents(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path)
    result = ingest(cfg)
    assert result.manifest_path is not None and result.manifest_path.exists()
    doc = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert doc["vendor"] == VENDOR
    assert doc["dataset"] == DATASET
    assert doc["schema"] == SCHEMA
    assert doc["row_counts_by_symbol"]["ES"] == 3
    assert doc["row_counts_by_symbol"]["NQ"] == 1
    assert doc["binding"]["is_bbo_available"] is True


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path, force=True)
    ingest(cfg)
    cfg2 = _make_config(tmp_path, force=False)
    with pytest.raises(FileExistsError):
        ingest(cfg2)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    _make_mbp1_fixture(tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "sample.parquet")
    cfg = _make_config(tmp_path, dry_run=True)
    result = ingest(cfg)
    assert result.manifest_path is None
    assert not list(cfg.output_root.rglob("part-*.parquet"))


def test_ingest_uses_instrument_id_to_symbol_map(tmp_path: Path) -> None:
    fpath = tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "by_id.parquet"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("ts_event", pa.int64()),
            ("ts_recv", pa.int64()),
            ("instrument_id", pa.int64()),
            ("action", pa.string()),
            ("side", pa.string()),
            ("price", pa.float64()),
            ("size", pa.int64()),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "ts_event": [1, 2],
            "ts_recv": [10, 20],
            "instrument_id": [1, 2],
            "action": ["T", "T"],
            "side": ["B", "A"],
            "price": [5000.0, 18000.0],
            "size": [1, 1],
        },
        schema=schema,
    )
    pq.write_table(table, fpath, compression="zstd")
    cfg = IngestConfig(
        input_root=fpath.parent.parent,
        output_root=tmp_path / "out",
        reports_root=tmp_path / "rep",
        date="2026-05-18",
        instrument_id_to_symbol={1: "ES", 2: "NQ"},
        force=True,
    )
    result = ingest(cfg)
    assert result.row_counts_by_symbol == {"ES": 1, "NQ": 1}
