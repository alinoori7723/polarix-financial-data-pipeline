from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.ingestion.cme_reference_ingest import IngestConfig, ingest
from polarix.quality.cme_reference_quality import (
    CmeQualityConfig,
    build_quality_report,
    write_reports,
)


def _make_fixture(path: Path, *, rows: list[dict], include_bbo: bool = True) -> Path:
    fields = [
        ("ts_event", pa.int64()),
        ("ts_recv", pa.int64()),
        ("raw_symbol", pa.string()),
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
        ]
    schema = pa.schema(fields)
    cols: dict[str, list] = {name: [] for name, _ in fields}
    for r in rows:
        for name, _ in fields:
            cols[name].append(r.get(name))
    table = pa.Table.from_pydict(cols, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def _row(*, ts_event, ts_recv, symbol, action, side, price, size, bbo: bool = True) -> dict:
    d = {
        "ts_event": ts_event,
        "ts_recv": ts_recv,
        "raw_symbol": symbol,
        "action": action,
        "side": side,
        "price": price,
        "size": size,
    }
    if bbo:
        d.update(
            {"bid_px_00": price - 0.25, "ask_px_00": price + 0.25, "bid_sz_00": 5, "ask_sz_00": 5}
        )
    return d


def _setup_es_nq_clean(tmp_path: Path) -> CmeQualityConfig:
    _make_fixture(
        tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "ok.parquet",
        rows=[
            _row(
                ts_event=1700000000000000000,
                ts_recv=1700000000000000010,
                symbol="ESM6",
                action="T",
                side="B",
                price=5000.0,
                size=5,
            ),
            _row(
                ts_event=1700000000000000100,
                ts_recv=1700000000000000110,
                symbol="ESM6",
                action="T",
                side="A",
                price=5000.25,
                size=2,
            ),
            _row(
                ts_event=1700000000000000200,
                ts_recv=1700000000000000210,
                symbol="NQM6",
                action="T",
                side="B",
                price=18000.0,
                size=3,
            ),
        ],
    )
    cfg_ingest = IngestConfig(
        input_root=tmp_path / "raw" / "cme_sample",
        output_root=tmp_path / "normalized" / "cme_reference",
        reports_root=tmp_path / "reports",
        date="2026-05-18",
        force=True,
    )
    ingest(cfg_ingest)
    return CmeQualityConfig(
        date="2026-05-18",
        input_root=tmp_path / "raw" / "cme_sample",
        output_root=tmp_path / "normalized" / "cme_reference",
        reports_root=tmp_path / "reports",
    )


def test_quality_writes_json_and_txt(tmp_path: Path) -> None:
    qcfg = _setup_es_nq_clean(tmp_path)
    rep = build_quality_report(qcfg)
    json_path, txt_path = write_reports(qcfg, rep)
    assert json_path.exists() and txt_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["decision"] == rep["decision"]
    assert "Polarix CME Reference Quality Report" in txt_path.read_text(encoding="utf-8")


def test_quality_pass_on_valid_es_nq(tmp_path: Path) -> None:
    qcfg = _setup_es_nq_clean(tmp_path)
    rep = build_quality_report(qcfg)
    assert rep["decision"] == "PASS"
    assert set(rep["symbols_found"]) == {"ES", "NQ"}


def test_quality_partial_when_only_one_symbol(tmp_path: Path) -> None:
    _make_fixture(
        tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "es_only.parquet",
        rows=[
            _row(ts_event=1, ts_recv=10, symbol="ESM6", action="T", side="B", price=5000.0, size=1),
            _row(
                ts_event=2, ts_recv=20, symbol="ESM6", action="T", side="A", price=5000.25, size=1
            ),
        ],
    )
    ingest(
        IngestConfig(
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
            date="2026-05-18",
            force=True,
        )
    )
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
        )
    )
    assert rep["decision"] == "PARTIAL"
    assert rep["symbols_found"] == ["ES"]


def test_quality_partial_when_bbo_missing(tmp_path: Path) -> None:
    _make_fixture(
        tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "no_bbo.parquet",
        include_bbo=False,
        rows=[
            _row(
                ts_event=1,
                ts_recv=10,
                symbol="ESM6",
                action="T",
                side="B",
                price=5000.0,
                size=1,
                bbo=False,
            ),
            _row(
                ts_event=2,
                ts_recv=20,
                symbol="NQM6",
                action="T",
                side="B",
                price=18000.0,
                size=1,
                bbo=False,
            ),
        ],
    )
    ingest(
        IngestConfig(
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
            date="2026-05-18",
            force=True,
        )
    )
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
        )
    )
    assert rep["decision"] == "PARTIAL"
    assert any(("BBO missing" in w for w in rep["warnings"]))


def test_quality_fail_when_no_trade_rows(tmp_path: Path) -> None:
    _make_fixture(
        tmp_path / "raw" / "cme_sample" / "date=2026-05-18" / "no_trade.parquet",
        rows=[
            _row(ts_event=1, ts_recv=10, symbol="ESM6", action="C", side="B", price=5000.0, size=1)
        ],
    )
    ingest(
        IngestConfig(
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
            date="2026-05-18",
            force=True,
        )
    )
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
        )
    )
    assert rep["decision"] == "FAIL"
    assert any(("no trade rows" in w for w in rep["fatal_warnings"]))


def test_quality_fail_when_required_columns_missing(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "cme_sample" / "bad.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("ts_event", pa.int64()),
            ("ts_recv", pa.int64()),
            ("raw_symbol", pa.string()),
            ("action", pa.string()),
            ("price", pa.float64()),
            ("size", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "ts_event": [1],
                "ts_recv": [10],
                "raw_symbol": ["ESM6"],
                "action": ["T"],
                "price": [5000.0],
                "size": [1],
            },
            schema=schema,
        ),
        path,
    )
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=tmp_path / "raw" / "cme_sample",
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
        )
    )
    assert rep["decision"] == "FAIL"
    assert any(("schema rejection" in w for w in rep["fatal_warnings"]))


def test_quality_fail_when_sample_missing(tmp_path: Path) -> None:
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=tmp_path / "absent",
            output_root=tmp_path / "out",
            reports_root=tmp_path / "rep",
        )
    )
    assert rep["decision"] == "FAIL"
    assert rep["error"] == "MISSING_SAMPLE_DATA"
