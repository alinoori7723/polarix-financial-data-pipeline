from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.normalization.normalization import (
    DEFAULT_JOIN_SAFE_THRESHOLD_MS,
    NormalizationConfig,
    normalize,
)
from polarix.orchestration.run_metadata import RunMetadata
from polarix.quality.telemetry_quality import (
    QualityConfig,
    QualityThresholds,
    build_quality_report,
    write_reports,
)

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


def _row(
    *,
    symbol: str,
    time_msc_raw: int,
    recv_time_utc_ms: int,
    bid: float = 100.0,
    ask: float = 100.5,
    bid_scaled: int = 10000,
    ask_scaled: int = 10050,
    spread_points: int = 5,
    suppressed_count: int = 0,
) -> dict:
    return {
        "symbol": symbol,
        "time_msc_raw": time_msc_raw,
        "recv_time_utc_ms": recv_time_utc_ms,
        "monotonic_ns": 0,
        "bid": bid,
        "ask": ask,
        "last": 0.0,
        "bid_scaled": bid_scaled,
        "ask_scaled": ask_scaled,
        "last_scaled": 0,
        "volume": 0,
        "flags": 0,
        "spread_points": spread_points,
        "suppressed_count": suppressed_count,
        "first_suppressed_time_ms": None,
        "last_suppressed_time_ms": None,
        "suppressed_reason": None,
    }


def _make_metadata(tmp_path: Path, verified_offset_min: int) -> RunMetadata:
    src = tmp_path / "fake_meta.json"
    src.write_text("{}", encoding="utf-8")
    return RunMetadata(
        source_path=src,
        source_kind="live_run_summary",
        run_id="testrun",
        verified_offset_min=verified_offset_min,
        timestamp_semantics_status="OFFSET_VERIFIED_FOR_SESSION",
        clock_status={"status": "CALIBRATION_COARSE_OK"},
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
    part_dir.mkdir(parents=True, exist_ok=True)
    out = part_dir / "part-00000001-aaaa.parquet"
    cols = {n: [] for n, _ in BRONZE_FIELDS}
    for r in rows:
        for n, _ in BRONZE_FIELDS:
            cols[n].append(r.get(n))
    table = pa.Table.from_pydict(cols, schema=BRONZE_SCHEMA)
    pq.write_table(table, out, compression="zstd")
    return out


def _make_quality_config(
    tmp_path: Path,
    md: RunMetadata,
    *,
    ndx_spread_spike_price: float = 50.0,
    join_safe_low_pct: float = 50.0,
    latency_outlier_high_pct: float = 5.0,
) -> QualityConfig:
    return QualityConfig(
        date="2026-05-18",
        raw_root=tmp_path / "raw",
        silver_root=tmp_path / "silver",
        reports_root=tmp_path / "reports",
        metadata=md,
        join_safe_threshold_ms=DEFAULT_JOIN_SAFE_THRESHOLD_MS,
        thresholds=QualityThresholds(
            join_safe_low_pct=join_safe_low_pct,
            latency_outlier_high_pct=latency_outlier_high_pct,
            ndx_spread_spike_price=ndx_spread_spike_price,
        ),
    )


def _setup_clean_dataset(tmp_path: Path, *, n_rows_per_symbol: int = 20) -> RunMetadata:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    base = 1000000000000
    rows_spx = [
        _row(
            symbol="SPX500",
            time_msc_raw=base + offset_ms + i,
            recv_time_utc_ms=base + i + 10,
            bid=100.0 + i * 0.01,
            ask=100.5 + i * 0.01,
        )
        for i in range(n_rows_per_symbol)
    ]
    rows_ndx = [
        _row(
            symbol="NDX100",
            time_msc_raw=base + offset_ms + i,
            recv_time_utc_ms=base + i + 10,
            bid=20000.0 + i * 0.1,
            ask=20001.0 + i * 0.1,
            bid_scaled=2000000,
            ask_scaled=2000100,
        )
        for i in range(n_rows_per_symbol)
    ]
    _seed_bronze(tmp_path / "raw", "SPX500", "2026-05-18", 5, rows_spx)
    _seed_bronze(tmp_path / "raw", "NDX100", "2026-05-18", 5, rows_ndx)
    cfg = NormalizationConfig(
        raw_root=tmp_path / "raw",
        silver_root=tmp_path / "silver",
        date="2026-05-18",
        metadata=md,
        force=True,
    )
    normalize(cfg)
    return md


def test_report_includes_row_counts_and_summaries(tmp_path: Path) -> None:
    md = _setup_clean_dataset(tmp_path, n_rows_per_symbol=20)
    qcfg = _make_quality_config(tmp_path, md)
    report = build_quality_report(qcfg)
    spx = report["per_symbol"]["SPX500"]
    assert spx["total_rows"] == 20
    assert 5 in spx["row_count_by_hour"]
    assert spx["row_count_by_hour"][5] == 20
    assert spx["spread_summary"]["min"] is not None
    assert spx["residual_summary"]["p99"] is not None
    assert spx["join_safe_pct"] == 100.0
    assert spx["bid_ask_violation_count"] == 0
    assert spx["scaled_bid_ask_violation_count"] == 0


def test_decision_pass_on_clean_dataset(tmp_path: Path) -> None:
    md = _setup_clean_dataset(tmp_path, n_rows_per_symbol=20)
    qcfg = _make_quality_config(tmp_path, md)
    report = build_quality_report(qcfg)
    assert report["decision"] == "PASS"


def test_decision_fail_when_no_silver(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    qcfg = _make_quality_config(tmp_path, md)
    report = build_quality_report(qcfg)
    assert report["decision"] == "FAIL"


def test_decision_fail_on_bid_ask_violation(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    base = 1000000000000
    rows_spx = [
        _row(
            symbol="SPX500",
            time_msc_raw=base + offset_ms,
            recv_time_utc_ms=base + 10,
            bid=100.0,
            ask=99.0,
        )
    ]
    rows_ndx = [
        _row(
            symbol="NDX100",
            time_msc_raw=base + offset_ms,
            recv_time_utc_ms=base + 10,
            bid=20000.0,
            ask=20001.0,
            bid_scaled=2000000,
            ask_scaled=2000100,
        )
    ]
    _seed_bronze(tmp_path / "raw", "SPX500", "2026-05-18", 5, rows_spx)
    _seed_bronze(tmp_path / "raw", "NDX100", "2026-05-18", 5, rows_ndx)
    normalize(
        NormalizationConfig(
            raw_root=tmp_path / "raw",
            silver_root=tmp_path / "silver",
            date="2026-05-18",
            metadata=md,
            force=True,
        )
    )
    qcfg = _make_quality_config(tmp_path, md)
    report = build_quality_report(qcfg)
    assert report["decision"] == "FAIL"
    assert any(("ask < bid" in w for w in report["fatal_warnings"]))


def test_decision_partial_on_low_join_safe(tmp_path: Path) -> None:
    md = _make_metadata(tmp_path, verified_offset_min=60)
    offset_ms = 60 * 60 * 1000
    base = 1000000000000
    rows_spx = [
        _row(symbol="SPX500", time_msc_raw=base + offset_ms, recv_time_utc_ms=base + 900)
        for _ in range(20)
    ]
    rows_ndx = [
        _row(
            symbol="NDX100",
            time_msc_raw=base + offset_ms,
            recv_time_utc_ms=base + 900,
            bid=20000.0,
            ask=20001.0,
            bid_scaled=2000000,
            ask_scaled=2000100,
        )
        for _ in range(20)
    ]
    _seed_bronze(tmp_path / "raw", "SPX500", "2026-05-18", 5, rows_spx)
    _seed_bronze(tmp_path / "raw", "NDX100", "2026-05-18", 5, rows_ndx)
    normalize(
        NormalizationConfig(
            raw_root=tmp_path / "raw",
            silver_root=tmp_path / "silver",
            date="2026-05-18",
            metadata=md,
            force=True,
        )
    )
    qcfg = _make_quality_config(tmp_path, md, join_safe_low_pct=90.0)
    report = build_quality_report(qcfg)
    assert report["decision"] == "PARTIAL"
    assert any(("join-safe percentage" in w for w in report["warnings"]))


def test_text_and_json_reports_written(tmp_path: Path) -> None:
    md = _setup_clean_dataset(tmp_path, n_rows_per_symbol=5)
    qcfg = _make_quality_config(tmp_path, md)
    report = build_quality_report(qcfg)
    json_path, txt_path = write_reports(qcfg, report)
    assert json_path.exists() and txt_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["decision"] == report["decision"]
    txt = txt_path.read_text(encoding="utf-8")
    assert "Polarix Telemetry Quality Report" in txt
    assert "SPX500" in txt
    assert "NDX100" in txt


def test_no_trading_functions_in_new_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forbidden = (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "trade_request",
        "TradeRequest",
        "MqlTradeRequest",
    )
    new_files = [
        repo_root / "src" / "polarix" / "orchestration" / "run_metadata.py",
        repo_root / "src" / "polarix" / "normalization" / "normalization.py",
        repo_root / "src" / "polarix" / "quality" / "telemetry_quality.py",
        repo_root / "scripts" / "normalize_telemetry.py",
        repo_root / "scripts" / "telemetry_quality_report.py",
    ]
    for path in new_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: forbidden token {token} appears"


def test_no_alpha_model_cme_references_in_new_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_phrases = ("alpha_engine", "CVD", "OFI", "cme_ingest", "model_training", "meta_model")
    new_files = [
        repo_root / "src" / "polarix" / "orchestration" / "run_metadata.py",
        repo_root / "src" / "polarix" / "normalization" / "normalization.py",
        repo_root / "src" / "polarix" / "quality" / "telemetry_quality.py",
        repo_root / "scripts" / "normalize_telemetry.py",
        repo_root / "scripts" / "telemetry_quality_report.py",
    ]
    for path in new_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_phrases:
            assert token not in text, f"{path}: forbidden token {token} appears"
