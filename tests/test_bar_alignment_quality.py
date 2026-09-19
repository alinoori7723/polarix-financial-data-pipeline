from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.alignment.bar_alignment_quality import (
    REASON_LOW_JOIN_SAFE_TICK_RATIO,
    REASON_NO_CME_DATA,
    REASON_NO_MT5_DATA,
    BarAlignmentConfig,
    BarAlignmentThresholds,
    build_bar_alignment_quality_report,
    join_and_classify_bars,
    parse_symbol_map,
    write_reports,
)
from polarix.features.bar_aggregation import (
    aggregate_cme_bars,
    aggregate_mt5_bars,
    parse_bucket_sizes,
)

SEC = 1000000000
CME_SCHEMA = pa.schema(
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
MT5_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("time_msc_utc_ms", pa.int64()),
        ("is_join_safe", pa.bool_()),
        ("mid", pa.float64()),
        ("spread_price", pa.float64()),
        ("spread_points", pa.int64()),
        ("residual_ms", pa.int64()),
        ("is_latency_outlier", pa.bool_()),
    ]
)


def _write_cme(path: Path, symbol: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f.name: [] for f in CME_SCHEMA}
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
    pq.write_table(pa.Table.from_pydict(cols, schema=CME_SCHEMA), path, compression="zstd")


def _write_mt5(path: Path, symbol: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f.name: [] for f in MT5_SCHEMA}
    for r in rows:
        cols["symbol"].append(symbol)
        cols["time_msc_utc_ms"].append(r["ts_ms"])
        cols["is_join_safe"].append(r.get("safe", True))
        cols["mid"].append(r.get("mid", 100.0))
        cols["spread_price"].append(r.get("spread", 0.25))
        cols["spread_points"].append(r.get("spread_points", 1))
        cols["residual_ms"].append(r.get("residual_ms", 5))
        cols["is_latency_outlier"].append(r.get("outlier", False))
    pq.write_table(pa.Table.from_pydict(cols, schema=MT5_SCHEMA), path, compression="zstd")


def _seed(
    tmp_path: Path,
    date: str,
    cme_rows_by_symbol: dict[str, list[dict]],
    mt5_rows_by_symbol: dict[str, list[dict]],
) -> tuple[Path, Path]:
    cme_root = tmp_path / "cme" / "reference_trades"
    mt5_root = tmp_path / "mt5"
    for s, rows in cme_rows_by_symbol.items():
        _write_cme(cme_root / f"symbol={s}" / f"date={date}" / "part-0001.parquet", s, rows)
    for s, rows in mt5_rows_by_symbol.items():
        _write_mt5(mt5_root / f"symbol={s}" / f"date={date}" / "part-0001.parquet", s, rows)
    return (cme_root, mt5_root)


def _make_config(
    tmp_path: Path,
    cme_root: Path,
    mt5_root: Path,
    date: str = "2026-05-18",
    bucket_sizes: str = "1s,5s,15s,60s",
    **overrides,
) -> BarAlignmentConfig:
    kw = dict(
        date=date,
        cme_root=cme_root,
        mt5_root=mt5_root,
        reports_root=tmp_path / "reports",
        symbol_map={"ES": "SPX500", "NQ": "NDX100"},
        bucket_sizes=parse_bucket_sizes(bucket_sizes),
        thresholds=BarAlignmentThresholds(),
        write_buckets=False,
    )
    kw.update(overrides)
    return BarAlignmentConfig(**kw)


def test_join_aligns_cme_and_mt5_by_bucket_start() -> None:
    cme = aggregate_cme_bars(
        pl.DataFrame(
            {
                "symbol": ["ES"],
                "event_time_utc_ns": [SEC // 2],
                "price": [100.0],
                "size": [1],
                "aggressor_side": ["BUY"],
                "is_reference_trade_valid": [True],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    mt5 = aggregate_mt5_bars(
        pl.DataFrame(
            {
                "symbol": ["SPX500"],
                "time_msc_utc_ms": [500],
                "is_join_safe": [True],
                "mid": [100.0],
                "spread_price": [0.5],
                "spread_points": [1],
                "residual_ms": [5],
                "is_latency_outlier": [False],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    joined = join_and_classify_bars(
        cme,
        mt5,
        cme_symbol="ES",
        mt5_symbol="SPX500",
        bucket_label="1s",
        thresholds=BarAlignmentThresholds(),
    )
    assert joined.height == 1
    row = joined.row(0, named=True)
    assert row["has_cme_data"] is True and row["has_mt5_data"] is True
    assert row["is_bar_aligned"] is True
    assert row["bar_reject_reason"] is None
    assert row["cme_vwap"] == pytest.approx(100.0)
    assert row["mt5_mid_twap"] == pytest.approx(100.0)


def test_join_marks_no_cme_data() -> None:
    empty_cme = aggregate_cme_bars(
        pl.DataFrame(
            {
                "symbol": [],
                "event_time_utc_ns": [],
                "price": [],
                "size": [],
                "aggressor_side": [],
                "is_reference_trade_valid": [],
            },
            schema={
                "symbol": pl.String,
                "event_time_utc_ns": pl.Int64,
                "price": pl.Float64,
                "size": pl.Int64,
                "aggressor_side": pl.String,
                "is_reference_trade_valid": pl.Boolean,
            },
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    mt5 = aggregate_mt5_bars(
        pl.DataFrame(
            {
                "symbol": ["SPX500"],
                "time_msc_utc_ms": [500],
                "is_join_safe": [True],
                "mid": [100.0],
                "spread_price": [0.5],
                "spread_points": [1],
                "residual_ms": [5],
                "is_latency_outlier": [False],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    joined = join_and_classify_bars(
        empty_cme,
        mt5,
        cme_symbol="ES",
        mt5_symbol="SPX500",
        bucket_label="1s",
        thresholds=BarAlignmentThresholds(),
    )
    assert joined.row(0, named=True)["bar_reject_reason"] == REASON_NO_CME_DATA


def test_join_marks_no_mt5_data() -> None:
    cme = aggregate_cme_bars(
        pl.DataFrame(
            {
                "symbol": ["ES"],
                "event_time_utc_ns": [500000000],
                "price": [100.0],
                "size": [1],
                "aggressor_side": ["BUY"],
                "is_reference_trade_valid": [True],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    empty_mt5 = aggregate_mt5_bars(
        pl.DataFrame(
            {
                "symbol": [],
                "time_msc_utc_ms": [],
                "is_join_safe": [],
                "mid": [],
                "spread_price": [],
                "spread_points": [],
                "residual_ms": [],
                "is_latency_outlier": [],
            },
            schema={
                "symbol": pl.String,
                "time_msc_utc_ms": pl.Int64,
                "is_join_safe": pl.Boolean,
                "mid": pl.Float64,
                "spread_price": pl.Float64,
                "spread_points": pl.Int64,
                "residual_ms": pl.Int64,
                "is_latency_outlier": pl.Boolean,
            },
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    joined = join_and_classify_bars(
        cme,
        empty_mt5,
        cme_symbol="ES",
        mt5_symbol="SPX500",
        bucket_label="1s",
        thresholds=BarAlignmentThresholds(),
    )
    assert joined.row(0, named=True)["bar_reject_reason"] == REASON_NO_MT5_DATA


def test_join_marks_low_join_safe_tick_ratio() -> None:
    cme = aggregate_cme_bars(
        pl.DataFrame(
            {
                "symbol": ["ES"],
                "event_time_utc_ns": [500000000],
                "price": [100.0],
                "size": [1],
                "aggressor_side": ["BUY"],
                "is_reference_trade_valid": [True],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    mt5 = aggregate_mt5_bars(
        pl.DataFrame(
            {
                "symbol": ["SPX500"] * 4,
                "time_msc_utc_ms": [0, 100, 200, 300],
                "is_join_safe": [True, False, False, False],
                "mid": [100.0, 100.0, 100.0, 100.0],
                "spread_price": [0.5, 0.5, 0.5, 0.5],
                "spread_points": [1, 1, 1, 1],
                "residual_ms": [5, 5, 5, 5],
                "is_latency_outlier": [False, False, False, False],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    joined = join_and_classify_bars(
        cme,
        mt5,
        cme_symbol="ES",
        mt5_symbol="SPX500",
        bucket_label="1s",
        thresholds=BarAlignmentThresholds(),
    )
    assert joined.row(0, named=True)["bar_reject_reason"] == REASON_LOW_JOIN_SAFE_TICK_RATIO


def test_join_computes_basis_and_bps() -> None:
    cme = aggregate_cme_bars(
        pl.DataFrame(
            {
                "symbol": ["ES"],
                "event_time_utc_ns": [500000000],
                "price": [5000.0],
                "size": [1],
                "aggressor_side": ["BUY"],
                "is_reference_trade_valid": [True],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    mt5 = aggregate_mt5_bars(
        pl.DataFrame(
            {
                "symbol": ["SPX500"],
                "time_msc_utc_ms": [500],
                "is_join_safe": [True],
                "mid": [4999.5],
                "spread_price": [0.5],
                "spread_points": [1],
                "residual_ms": [5],
                "is_latency_outlier": [False],
            }
        ),
        bucket_size_ns=SEC,
        bucket_label="1s",
    )
    joined = join_and_classify_bars(
        cme,
        mt5,
        cme_symbol="ES",
        mt5_symbol="SPX500",
        bucket_label="1s",
        thresholds=BarAlignmentThresholds(),
    )
    row = joined.row(0, named=True)
    assert row["basis_cme_vwap_to_mt5_twap"] == pytest.approx(-0.5)
    assert row["abs_basis"] == pytest.approx(0.5)
    assert row["basis_bps"] == pytest.approx(-1.0)


def _matched_overlap(
    date_seconds: int, n: int, *, bucket_seconds: int
) -> tuple[list[dict], list[dict]]:
    cme_rows = []
    mt5_rows = []
    for i in range(n):
        t_s = date_seconds + i * bucket_seconds
        cme_rows.append(
            {
                "event_ns": int(t_s * 1000000000.0 + 100000),
                "price": 5000.0,
                "size": 1,
                "aggressor_side": "BUY",
            }
        )
        for k in range(5):
            mt5_rows.append(
                {"ts_ms": int(t_s * 1000 + k * (bucket_seconds * 200)), "safe": True, "mid": 4999.5}
            )
    return (cme_rows, mt5_rows)


def test_pass_when_all_symbols_align_at_5s_or_better(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=20, bucket_seconds=5)
    cme_nq, mt5_ndx = _matched_overlap(base, n=20, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="5s,15s,60s")
    res = build_bar_alignment_quality_report(cfg)
    assert res.report["quality_decision"] == "PASS"
    assert res.report["real_overlap_present"] is True


def test_partial_when_alignment_only_at_longer_buckets(tmp_path: Path) -> None:
    base = 1779081406
    cme_rows = [
        {
            "event_ns": int((base + i) * 1000000000.0 + 100),
            "price": 5000.0,
            "size": 1,
            "aggressor_side": "BUY",
        }
        for i in range(60)
        if i % 2 == 1
    ]
    mt5_rows = [
        {"ts_ms": int((base + i) * 1000 + k * 100), "safe": True, "mid": 4999.5}
        for i in range(60)
        for k in range(5)
    ]
    cme_root, mt5_root = _seed(
        tmp_path,
        "2026-05-18",
        {"ES": cme_rows, "NQ": cme_rows},
        {"SPX500": mt5_rows, "NDX100": mt5_rows},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="1s")
    res = build_bar_alignment_quality_report(cfg)
    assert res.report["quality_decision"] == "PARTIAL"


def test_fail_when_no_overlap(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed(
        tmp_path,
        "2026-05-18",
        {
            "ES": [
                {"event_ns": int(1000000000.0), "price": 5000.0, "size": 1, "aggressor_side": "BUY"}
            ],
            "NQ": [],
        },
        {"SPX500": [{"ts_ms": 60000, "safe": True}], "NDX100": []},
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="1s,5s")
    res = build_bar_alignment_quality_report(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert res.report["decision_reason"] == "MISSING_OVERLAP_DATA"


def test_cme_volume_alignment_ratio_in_metrics(tmp_path: Path) -> None:
    base = 1779081406
    cme_rows = [
        {
            "event_ns": int((base + i) * 1000000000.0 + 100),
            "price": 5000.0,
            "size": 1,
            "aggressor_side": "BUY",
        }
        for i in range(5)
    ] + [
        {
            "event_ns": int((base + 7) * 1000000000.0),
            "price": 5000.0,
            "size": 999,
            "aggressor_side": "BUY",
        }
    ]
    mt5_rows = [
        {"ts_ms": int((base + i) * 1000 + k * 100), "safe": True, "mid": 4999.5}
        for i in range(5)
        for k in range(5)
    ] + [{"ts_ms": int((base + 10) * 1000), "safe": True, "mid": 4999.5}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_rows, "NQ": []}, {"SPX500": mt5_rows, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="1s")
    res = build_bar_alignment_quality_report(cfg)
    es = res.report["per_symbol"]["ES"]["by_bucket_size"]["1s"]
    assert es["cme_volume_total"] == 1004
    assert es["cme_volume_in_aligned_buckets"] == 5
    assert es["cme_volume_alignment_ratio"] == pytest.approx(5 / 1004)


def test_write_reports_creates_json_txt_and_optional_parquet(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="5s", write_buckets=True)
    res = build_bar_alignment_quality_report(cfg)
    json_path, txt_path, buckets_path = write_reports(cfg, res)
    assert json_path.exists() and txt_path.exists()
    assert buckets_path is not None and buckets_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    for k in (
        "date",
        "cme_root",
        "mt5_root",
        "reports_root",
        "symbol_map",
        "bucket_sizes",
        "thresholds",
        "quality_decision",
        "per_symbol",
        "per_bucket_size",
        "overall",
        "warnings",
        "errors",
    ):
        assert k in parsed


def test_report_records_basis_and_spread_summaries(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="5s")
    res = build_bar_alignment_quality_report(cfg)
    es = res.report["per_symbol"]["ES"]["by_bucket_size"]["5s"]
    assert "basis_summary" in es and es["basis_summary"]["mean"] is not None
    assert "spread_price_max_summary" in es
    assert "reject_reason_counts" in es


def test_parse_symbol_map_default() -> None:
    m = parse_symbol_map("ES=SPX500,NQ=NDX100")
    assert m == {"ES": "SPX500", "NQ": "NDX100"}


@pytest.mark.parametrize("bad", ["", "ESSPX500", "=SPX500", "ES=", ","])
def test_parse_symbol_map_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_symbol_map(bad)


PHASE_2C_FILES = (
    "src/polarix/features/bar_aggregation.py",
    "src/polarix/alignment/bar_alignment_quality.py",
    "scripts/bar_alignment_quality_report.py",
)


def _phase_2c_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2C_FILES}


def test_phase_2c_no_trading_functions() -> None:
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
    for path, text in _phase_2c_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2c_no_mt5_sdk_imports() -> None:
    forbidden = (
        "import MetaTrader5",
        "from MetaTrader5",
        "import mt5_readonly",
        "import polarix.ingestion.mt5_readonly",
        "from polarix.ingestion.mt5_readonly",
        "mt5.order_send",
        "mt5.copy_ticks_from",
    )
    for path, text in _phase_2c_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2c_no_model_training_imports() -> None:
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
    for path, text in _phase_2c_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2c_no_cvd_cumulative_aggregation() -> None:
    import ast

    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running")
    for path, text in _phase_2c_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value.lower() != "cvd", f"{path}:{node.lineno}: 'cvd' literal"


def test_phase_2c_no_synthetic_overlap_real_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2c_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2c_does_not_weaken_50ms_tick_contract() -> None:
    for path, text in _phase_2c_texts().items():
        assert "DEFAULT_ALIGNMENT_TOLERANCE_MS" not in text, (
            f"{path}: must not redefine the tick-level tolerance"
        )
    from polarix.alignment.alignment_contract import DEFAULT_ALIGNMENT_TOLERANCE_MS as TOL_CONTRACT
    from polarix.alignment.alignment_quality import DEFAULT_ALIGNMENT_TOLERANCE_MS as TOL_QUALITY

    assert TOL_CONTRACT == 50
    assert TOL_QUALITY == 50
