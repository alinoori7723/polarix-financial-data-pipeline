from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.alignment.alignment_quality import (
    MS_TO_NS,
    REASON_MT5_NOT_JOIN_SAFE,
    REASON_NO_MT5_ROWS_FOR_SYMBOL,
    REASON_NO_OVERLAP_WINDOW,
    REASON_OUTSIDE_TOLERANCE,
    AlignmentQualityConfig,
    AlignmentQualityThresholds,
    _align_one_event,
    build_alignment_quality_report,
    parse_symbol_map,
    render_text_report,
    write_reports,
)

CME_REF_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("vendor", pa.string()),
        ("dataset", pa.string()),
        ("schema", pa.string()),
        ("symbol", pa.string()),
        ("event_time_utc_ns", pa.int64()),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("aggressor_side", pa.string()),
        ("signed_size", pa.float64()),
        ("is_reference_trade_valid", pa.bool_()),
    ]
)
MT5_SILVER_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("time_msc_utc_ms", pa.int64()),
        ("is_join_safe", pa.bool_()),
        ("mid", pa.float64()),
        ("spread_price", pa.float64()),
        ("spread_points", pa.int64()),
    ]
)


def _write_cme(path: Path, symbol: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f.name: [] for f in CME_REF_SCHEMA}
    for r in rows:
        cols["source"].append("databento")
        cols["vendor"].append("Databento")
        cols["dataset"].append("GLBX.MDP3")
        cols["schema"].append("mbp-1")
        cols["symbol"].append(symbol)
        cols["event_time_utc_ns"].append(r["event_ns"])
        cols["price"].append(r["price"])
        cols["size"].append(r["size"])
        cols["aggressor_side"].append(r.get("aggressor_side", "BUY"))
        cols["signed_size"].append(r.get("signed_size"))
        cols["is_reference_trade_valid"].append(r.get("valid", True))
    pq.write_table(pa.Table.from_pydict(cols, schema=CME_REF_SCHEMA), path, compression="zstd")


def _write_mt5(path: Path, symbol: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f.name: [] for f in MT5_SILVER_SCHEMA}
    for r in rows:
        cols["symbol"].append(symbol)
        cols["time_msc_utc_ms"].append(r["ts_ms"])
        cols["is_join_safe"].append(r.get("safe", True))
        cols["mid"].append(r.get("mid", 100.0))
        cols["spread_price"].append(r.get("spread", 0.25))
        cols["spread_points"].append(r.get("spread_points", 1))
    pq.write_table(pa.Table.from_pydict(cols, schema=MT5_SILVER_SCHEMA), path, compression="zstd")


def _seed_datasets(
    tmp_path: Path,
    date: str,
    cme_rows_by_symbol: dict[str, list[dict]],
    mt5_rows_by_symbol: dict[str, list[dict]],
) -> tuple[Path, Path]:
    cme_root = tmp_path / "cme" / "reference_trades"
    mt5_root = tmp_path / "mt5"
    for sym, rows in cme_rows_by_symbol.items():
        _write_cme(cme_root / f"symbol={sym}" / f"date={date}" / "part-0001.parquet", sym, rows)
    for sym, rows in mt5_rows_by_symbol.items():
        _write_mt5(mt5_root / f"symbol={sym}" / f"date={date}" / "part-0001.parquet", sym, rows)
    return (cme_root, mt5_root)


def _make_config(
    tmp_path: Path, cme_root: Path, mt5_root: Path, date: str = "2026-05-18", **overrides
) -> AlignmentQualityConfig:
    kw = dict(
        date=date,
        cme_root=cme_root,
        mt5_root=mt5_root,
        reports_root=tmp_path / "reports",
        symbol_map={"ES": "SPX500", "NQ": "NDX100"},
        alignment_tolerance_ms=50,
        diagnostic_tolerances_ms=(50, 100, 250, 500, 1000),
        sample_unmatched_limit=100,
        thresholds=AlignmentQualityThresholds(),
        write_unmatched_sample=True,
    )
    kw.update(overrides)
    return AlignmentQualityConfig(**kw)


def test_align_one_event_picks_latest_at_or_before_event() -> None:
    mt5_ms = [1000, 1010, 1020]
    safe = [True, True, True]
    idx, delta, reason = _align_one_event(1020 * MS_TO_NS, mt5_ms, safe, 50)
    assert idx == 2
    assert delta == 0.0
    assert reason == ""


def test_align_one_event_never_uses_future_quote() -> None:
    mt5_ms = [1000, 1050]
    safe = [True, True]
    idx, delta, reason = _align_one_event(1010 * MS_TO_NS, mt5_ms, safe, 50)
    assert idx == 0
    assert delta == 10.0


def test_align_one_event_rejects_outside_tolerance() -> None:
    mt5_ms = [1000]
    safe = [True]
    idx, delta, reason = _align_one_event(1060 * MS_TO_NS, mt5_ms, safe, 50)
    assert idx is None
    assert reason == REASON_OUTSIDE_TOLERANCE


def test_align_one_event_skips_unsafe_and_finds_earlier_safe() -> None:
    mt5_ms = [1000, 1010, 1020]
    safe = [True, False, False]
    idx, delta, reason = _align_one_event(1020 * MS_TO_NS, mt5_ms, safe, 50)
    assert idx == 0
    assert reason == ""


def test_align_one_event_unsafe_when_all_recent_unsafe() -> None:
    mt5_ms = [990, 1000, 1010]
    safe = [False, False, False]
    idx, delta, reason = _align_one_event(1020 * MS_TO_NS, mt5_ms, safe, 50)
    assert idx is None
    assert reason == REASON_MT5_NOT_JOIN_SAFE


def test_align_one_event_no_mt5_rows() -> None:
    idx, _, reason = _align_one_event(1000 * MS_TO_NS, [], [], 50)
    assert idx is None
    assert reason == REASON_NO_MT5_ROWS_FOR_SYMBOL


def test_match_rate_and_unmatched_volume_ratio_basic(tmp_path: Path) -> None:
    cme_rows = [
        {"event_ns": 1000000000, "price": 5000.0, "size": 1},
        {"event_ns": 1010000000, "price": 5000.0, "size": 5},
        {"event_ns": 1061000000, "price": 5000.0, "size": 10},
    ]
    mt5_rows = [
        {"ts_ms": 999, "safe": True, "mid": 100.0, "spread": 0.25},
        {"ts_ms": 1010, "safe": True, "mid": 100.0, "spread": 0.25},
    ]
    cme_root, mt5_root = _seed_datasets(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": mt5_rows, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, sample = build_alignment_quality_report(cfg)
    es = report["per_symbol"]["ES"]
    assert es["cme_trades_total"] == 3
    assert es["aligned_count"] == 2
    assert es["unmatched_count"] == 1
    assert es["alignment_match_rate"] == pytest.approx(2 / 3)
    assert es["aligned_volume"] == 6
    assert es["unmatched_volume"] == 10
    assert es["unmatched_volume_ratio"] == pytest.approx(10 / 16)


def test_high_volume_unmatched_dominates_ratio(tmp_path: Path) -> None:
    cme_rows = [
        {"event_ns": (1000 + i) * MS_TO_NS, "price": 5000.0, "size": 1} for i in range(10)
    ] + [{"event_ns": 1060 * MS_TO_NS, "price": 5000.0, "size": 1000}]
    mt5_rows = [{"ts_ms": 1000 + i, "safe": True, "mid": 100.0, "spread": 0.25} for i in range(10)]
    cme_root, mt5_root = _seed_datasets(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": mt5_rows, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    es = report["per_symbol"]["ES"]
    assert es["cme_trades_total"] == 11
    assert es["aligned_count"] == 10
    assert es["alignment_match_rate"] == pytest.approx(10 / 11)
    assert es["unmatched_volume"] == 1000
    assert es["cme_volume_total"] == 10 + 1000
    assert es["unmatched_volume_ratio"] == pytest.approx(1000 / 1010)


def test_no_overlap_window_reports_no_overlap(tmp_path: Path) -> None:
    cme_rows = [{"event_ns": 1000000000, "price": 5000.0, "size": 1}]
    mt5_rows = [{"ts_ms": 5000000, "safe": True}]
    cme_root, mt5_root = _seed_datasets(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": mt5_rows, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "FAIL"
    assert report["decision_reason"] == "MISSING_OVERLAP_DATA"
    es = report["per_symbol"]["ES"]
    assert es["unmatched_count_by_reason"].get(REASON_NO_OVERLAP_WINDOW) == 1


def test_missing_cme_reference(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [], "NQ": []},
        {"SPX500": [{"ts_ms": 1000, "safe": True}], "NDX100": []},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    for d in (cme_root / "symbol=ES" / "date=2026-05-18").iterdir():
        d.unlink()
    (cme_root / "symbol=ES" / "date=2026-05-18").rmdir()
    for d in (cme_root / "symbol=NQ" / "date=2026-05-18").iterdir():
        d.unlink()
    (cme_root / "symbol=NQ" / "date=2026-05-18").rmdir()
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "FAIL"
    assert report["decision_reason"] == "MISSING_CME_REFERENCE"


def test_missing_mt5_silver(tmp_path: Path) -> None:
    cme_rows = [{"event_ns": 1000000000, "price": 5000.0, "size": 1}]
    cme_root, mt5_root = _seed_datasets(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": [], "NDX100": []}
    )
    for s in ("SPX500", "NDX100"):
        d = mt5_root / f"symbol={s}" / "date=2026-05-18"
        if d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "FAIL"
    assert report["decision_reason"] == "MISSING_MT5_SILVER"


def test_pass_decision_when_metrics_meet_thresholds(tmp_path: Path) -> None:
    cme_rows_es = [
        {"event_ns": (1000 + i) * MS_TO_NS, "price": 5000.0, "size": 1} for i in range(5)
    ]
    cme_rows_nq = [
        {"event_ns": (1000 + i) * MS_TO_NS, "price": 18000.0, "size": 1} for i in range(5)
    ]
    mt5_rows = [{"ts_ms": 1000 + i, "safe": True, "mid": 100.0, "spread": 0.25} for i in range(5)]
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows_es, "NQ": cme_rows_nq},
        {"SPX500": mt5_rows, "NDX100": mt5_rows},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "PASS"


def test_partial_when_match_rate_below_threshold(tmp_path: Path) -> None:
    cme_rows = [
        {"event_ns": 1000 * MS_TO_NS, "price": 5000.0, "size": 1},
        {"event_ns": 2000 * MS_TO_NS, "price": 5000.0, "size": 1},
        {"event_ns": 3000 * MS_TO_NS, "price": 5000.0, "size": 1},
    ]
    mt5_rows = [{"ts_ms": 1000, "safe": True}]
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows, "NQ": cme_rows},
        {"SPX500": mt5_rows, "NDX100": mt5_rows},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "PARTIAL"
    assert any(("match rate" in w for w in report["warnings"]))


def test_partial_when_unmatched_volume_ratio_high(tmp_path: Path) -> None:
    cme_rows = [
        {"event_ns": (1000 + i) * MS_TO_NS, "price": 5000.0, "size": 1} for i in range(10)
    ] + [{"event_ns": 3000 * MS_TO_NS, "price": 5000.0, "size": 20}]
    mt5_rows = [{"ts_ms": 1000 + i, "safe": True} for i in range(10)]
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows, "NQ": cme_rows},
        {"SPX500": mt5_rows, "NDX100": mt5_rows},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "PARTIAL"
    assert any(("unmatched_volume_ratio" in w for w in report["warnings"]))


def test_fail_when_required_columns_missing(tmp_path: Path) -> None:
    cme_rows = [{"event_ns": 1000000000, "price": 5000.0, "size": 1}]
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows, "NQ": []},
        {"SPX500": [{"ts_ms": 1000, "safe": True}], "NDX100": []},
    )
    bad_path = mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet"
    schema = pa.schema(
        [
            ("symbol", pa.string()),
            ("time_msc_utc_ms", pa.int64()),
            ("mid", pa.float64()),
            ("spread_price", pa.float64()),
            ("spread_points", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "symbol": ["SPX500"],
                "time_msc_utc_ms": [1000],
                "mid": [100.0],
                "spread_price": [0.25],
                "spread_points": [1],
            },
            schema=schema,
        ),
        bad_path,
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    assert report["quality_decision"] == "FAIL"
    assert report["decision_reason"] == "MISSING_REQUIRED_COLUMNS"


def test_diagnostic_tolerance_table_monotonic(tmp_path: Path) -> None:
    cme_rows = (
        [{"event_ns": (1000 + i) * MS_TO_NS, "price": 5000.0, "size": 1} for i in range(3)]
        + [{"event_ns": 1100 * MS_TO_NS, "price": 5000.0, "size": 1}]
        + [{"event_ns": 1300 * MS_TO_NS, "price": 5000.0, "size": 1}]
        + [{"event_ns": 2000 * MS_TO_NS, "price": 5000.0, "size": 1}]
    )
    mt5_rows = [{"ts_ms": 1000 + i, "safe": True} for i in range(3)]
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows, "NQ": cme_rows},
        {"SPX500": mt5_rows, "NDX100": mt5_rows},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    es = report["per_symbol"]["ES"]
    diag = es["diagnostic_rates_by_tolerance_ms"]
    rates = [diag[t]["match_rate"] for t in sorted(diag, key=int)]
    for prev, nxt in zip(rates, rates[1:]):
        assert nxt >= prev
    assert rates[0] < rates[-1]


def test_unmatched_sample_records_high_size_first(tmp_path: Path) -> None:
    cme_rows = [
        {"event_ns": 3000 * MS_TO_NS, "price": 5000.0, "size": 100},
        {"event_ns": 4000 * MS_TO_NS, "price": 5000.0, "size": 5},
    ]
    mt5_rows = [{"ts_ms": 1000, "safe": True}]
    cme_root, mt5_root = _seed_datasets(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": mt5_rows, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, sample = build_alignment_quality_report(cfg)
    sizes = [r["cme_size"] for r in sample]
    assert 100 in sizes


def test_parse_symbol_map_default() -> None:
    m = parse_symbol_map("ES=SPX500,NQ=NDX100")
    assert m == {"ES": "SPX500", "NQ": "NDX100"}


@pytest.mark.parametrize("bad", ["", "ESSPX500", "=SPX500", "ES=", ","])
def test_parse_symbol_map_rejects_bad_inputs(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_symbol_map(bad)


def test_parse_symbol_map_ignores_blank_pairs() -> None:
    m = parse_symbol_map("ES=SPX500, ,NQ=NDX100")
    assert m == {"ES": "SPX500", "NQ": "NDX100"}


def test_render_text_report_includes_decision_and_window(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 1000000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": 999, "safe": True}], "NDX100": []},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, _ = build_alignment_quality_report(cfg)
    txt = render_text_report(report)
    assert "Decision               :" in txt
    assert "SYMBOL: ES -> SPX500" in txt
    assert "NEXT:" in txt


def test_write_reports_writes_json_txt_and_sample(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 5000000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": 999, "safe": True}], "NDX100": []},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root)
    report, sample = build_alignment_quality_report(cfg)
    json_path, txt_path, sample_path = write_reports(cfg, report, sample)
    assert json_path.exists() and txt_path.exists()
    if sample:
        assert sample_path is not None and sample_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["quality_decision"] == report["quality_decision"]


PHASE_2B_FILES = (
    "src/polarix/alignment/alignment_quality.py",
    "scripts/alignment_quality_report.py",
    "scripts/check_databento_overlap_window.py",
)


def _phase_2b_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2B_FILES}


def test_no_trading_functions_in_phase_2b() -> None:
    forbidden = (
        "order_send",
        "order_check",
        "order_calc_margin",
        "order_calc_profit",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for path, text in _phase_2b_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden token {token!r}"


def test_no_mt5_order_send_in_phase_2b() -> None:
    for path, text in _phase_2b_texts().items():
        assert "mt5.order_send" not in text, f"{path}: mt5.order_send present"


def test_no_model_training_imports_in_phase_2b() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "model.fit(",
        "model_training",
        "train_test_split",
    )
    for path, text in _phase_2b_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_no_cvd_aggregation_in_phase_2b() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running")
    for path, text in _phase_2b_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"
    import ast

    for path, text in _phase_2b_texts().items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value.lower() != "cvd", f"{path}:{node.lineno}: 'cvd' literal"


def test_no_synthetic_overlap_real_report_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2b_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"
