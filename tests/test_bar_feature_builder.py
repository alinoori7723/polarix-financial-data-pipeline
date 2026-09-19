from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.features.bar_aggregation import parse_bucket_sizes
from polarix.features.bar_feature_builder import BuilderConfig, build

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
        cols["mid"].append(r.get("mid", 5000.0))
        cols["spread_price"].append(r.get("spread", 0.25))
        cols["spread_points"].append(r.get("spread_points", 25))
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
    *,
    dry_run: bool = False,
    force: bool = True,
    bucket_sizes: str = "15s,60s",
    include_diagnostic_5s: bool = False,
) -> BuilderConfig:
    return BuilderConfig(
        date="2026-05-18",
        cme_root=cme_root,
        mt5_root=mt5_root,
        output_root=tmp_path / "out",
        reports_root=tmp_path / "reports",
        bucket_sizes=parse_bucket_sizes(bucket_sizes),
        dry_run=dry_run,
        force=force,
        include_diagnostic_5s=include_diagnostic_5s,
    )


def _matched(
    base_sec: int,
    n: int,
    *,
    bucket_seconds: int,
    agg: str = "BUY",
    cme_price: float = 5000.0,
    mt5_mid: float = 4999.5,
) -> tuple[list[dict], list[dict]]:
    cme_rows = []
    mt5_rows = []
    for i in range(n):
        t = base_sec + i * bucket_seconds
        cme_rows.append(
            {
                "event_ns": int(t * 1000000000.0 + 100000),
                "price": cme_price + i * 0.5,
                "size": 2,
                "aggressor_side": agg,
            }
        )
        for k in range(5):
            mt5_rows.append(
                {
                    "ts_ms": int(t * 1000 + k * bucket_seconds * 200),
                    "safe": True,
                    "mid": mt5_mid + i * 0.5,
                }
            )
    return (cme_rows, mt5_rows)


def test_cme_vwap_correct_in_output(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int(base * 1000000000.0 + 200),
            "price": 200.0,
            "size": 9,
            "aggressor_side": "SELL",
        },
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 200), "safe": True, "mid": 99.5}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df.height == 1
    assert df["cme_vwap"][0] == pytest.approx(190.0)


def test_primary_basis_is_close_not_vwap_twap(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int(base * 1000000000.0 + 200),
            "price": 200.0,
            "size": 9,
            "aggressor_side": "SELL",
        },
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 200), "safe": True, "mid": 195.0}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df["basis_close"][0] == pytest.approx(-5.0)
    assert df["diagnostic_basis_vwap_twap"][0] == pytest.approx(5.0)
    assert df["basis_close"][0] != df["diagnostic_basis_vwap_twap"][0]


def test_aggressor_volumes_preserved(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 1),
            "price": 100.0,
            "size": 5,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int(base * 1000000000.0 + 2),
            "price": 100.0,
            "size": 3,
            "aggressor_side": "SELL",
        },
        {
            "event_ns": int(base * 1000000000.0 + 3),
            "price": 100.0,
            "size": 2,
            "aggressor_side": "UNKNOWN",
        },
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 1), "safe": True}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    row = df.row(0, named=True)
    assert row["cme_buy_volume"] == 5
    assert row["cme_sell_volume"] == 3
    assert row["cme_unknown_aggressor_volume"] == 2
    assert row["cme_buy_volume_ratio"] == pytest.approx(0.5)
    assert row["cme_sell_volume_ratio"] == pytest.approx(0.3)
    assert row["cme_unknown_aggressor_volume_ratio"] == pytest.approx(0.2)
    assert row["cme_signed_volume_ratio"] == pytest.approx(0.2)


def test_cme_max_single_trade_volume_and_p99(tmp_path: Path) -> None:
    base = 1779081406
    sizes = [1, 1, 2, 5, 100]
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + i),
            "price": 100.0,
            "size": s,
            "aggressor_side": "BUY",
        }
        for i, s in enumerate(sizes)
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 1), "safe": True}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df["cme_max_single_trade_volume"][0] == 100
    assert df["cme_trade_size_p99"][0] >= 60.0
    assert df["cme_trade_size_mean"][0] == pytest.approx(sum(sizes) / len(sizes))


def test_cme_high_low_range_bps(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 1),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int(base * 1000000000.0 + 2),
            "price": 101.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int(base * 1000000000.0 + 3),
            "price": 99.0,
            "size": 1,
            "aggressor_side": "SELL",
        },
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 1), "safe": True}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df["cme_high_low_range_bps"][0] == pytest.approx((101 - 99) / 99 * 10000)


def test_no_max_1s_columns_in_output(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=3, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    for forbidden in ("cme_max_1s_volume_share", "cme_max_1s_signed_volume_share"):
        assert forbidden not in df.columns


def test_mt5_metadata_state_columns_present(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=3, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    for col in (
        "mt5_mid_open",
        "mt5_mid_close",
        "mt5_mid_high",
        "mt5_mid_low",
        "mt5_mid_twap",
        "mt5_mid_mean",
    ):
        assert col in df.columns


def test_mt5_mid_return_close_to_close(tmp_path: Path) -> None:
    base = 1779081406
    cme = []
    mt5 = []
    for i, mid in enumerate([100.0, 101.0, 102.0]):
        t = base + i * 15
        cme.append(
            {
                "event_ns": int(t * 1000000000.0 + 100),
                "price": 50.0,
                "size": 1,
                "aggressor_side": "BUY",
            }
        )
        mt5.append({"ts_ms": int(t * 1000 + 100), "safe": True, "mid": mid})
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    ).sort("bucket_start_utc_ns")
    rets = df["mt5_mid_return_close_to_close"].to_list()
    assert rets[0] is None
    assert rets[1] == pytest.approx(0.01)
    assert rets[2] == pytest.approx(1.0 / 101.0)


def test_basis_close_equals_mt5_mid_close_minus_cme_close_price(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 5000.0,
            "size": 1,
            "aggressor_side": "BUY",
        }
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 100), "safe": True, "mid": 4999.5}]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    row = df.row(0, named=True)
    assert row["basis_close"] == pytest.approx(-0.5)
    assert row["basis_close_bps"] == pytest.approx(-0.5 / 5000.0 * 10000)


def test_basis_change_as_lag_within_pair_bucket(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int((base + 15) * 1000000000.0 + 100),
            "price": 110.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
    ]
    mt5 = [
        {"ts_ms": int(base * 1000 + 100), "safe": True, "mid": 95.0},
        {"ts_ms": int((base + 15) * 1000 + 100), "safe": True, "mid": 99.0},
    ]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    ).sort("bucket_start_utc_ns")
    assert df["basis_close"].to_list() == [pytest.approx(-5.0), pytest.approx(-11.0)]
    assert df["basis_change"].to_list()[1] == pytest.approx(-6.0)
    assert df["basis_change"].to_list()[0] is None


def test_return_diff_close_to_close_consistent(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
        {
            "event_ns": int((base + 15) * 1000000000.0 + 100),
            "price": 110.0,
            "size": 1,
            "aggressor_side": "BUY",
        },
    ]
    mt5 = [
        {"ts_ms": int(base * 1000 + 100), "safe": True, "mid": 100.0},
        {"ts_ms": int((base + 15) * 1000 + 100), "safe": True, "mid": 105.0},
    ]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    ).sort("bucket_start_utc_ns")
    diff = df["return_diff_close_to_close"].to_list()
    assert diff[0] is None
    assert diff[1] == pytest.approx(0.05 - 0.1)


def test_feature_timestamp_equals_bucket_end_utc_ns(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=2, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df.filter(pl.col("feature_timestamp_utc_ns") != pl.col("bucket_end_utc_ns")).height == 0
    assert "decision_lag_ms" not in df.columns


def test_manifest_contents(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=3, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s,60s")
    res = build(cfg)
    assert res.manifest_path and res.manifest_path.exists()
    manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    contract = manifest["feature_contract"]
    for col in ("cme_close_price", "mt5_mid_close", "mt5_mid_twap", "cme_vwap"):
        assert col in contract["non_feature_columns"]
        assert col not in contract["model_feature_candidate_columns"]
    assert "diagnostic_basis_vwap_twap" in contract["diagnostic_feature_columns"]
    assert "diagnostic_basis_vwap_twap" not in contract["model_feature_candidate_columns"]
    assert "decision_lag_ms" in contract["forbidden_columns"]


def test_quality_flag_ok_when_all_gates_pass(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=2, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert all((df["feature_quality_flag"].to_list()[i] == "OK" for i in range(df.height)))
    assert all(df["is_model_eligible_candidate"].to_list())


def test_quality_flag_diagnostic_only_for_5s(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=2, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s", include_diagnostic_5s=True)
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=5s" / "*.parquet"
    )
    flags = set(df["feature_quality_flag"].to_list())
    assert "DIAGNOSTIC_ONLY" in flags
    assert all((not e for e in df["is_model_eligible_candidate"].to_list()))


def test_quality_flag_low_join_safe(tmp_path: Path) -> None:
    base = 1779081406
    cme = [
        {
            "event_ns": int(base * 1000000000.0 + 100),
            "price": 100.0,
            "size": 1,
            "aggressor_side": "BUY",
        }
    ]
    mt5 = [{"ts_ms": int(base * 1000 + 100 + i), "safe": False, "mid": 100.0} for i in range(5)]
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s")
    build(cfg)
    df = pl.read_parquet(
        cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s" / "*.parquet"
    )
    assert df["feature_quality_flag"][0] == "LOW_JOIN_SAFE_RATIO"
    assert not df["is_model_eligible_candidate"][0]


def test_missing_input_returns_missing_input_data(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, tmp_path / "absent_cme", tmp_path / "absent_mt5")
    res = build(cfg)
    assert res.missing_input is True
    assert res.error == "MISSING_INPUT_DATA"


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=2, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s", force=True)
    build(cfg)
    cfg2 = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s", force=False)
    with pytest.raises(FileExistsError):
        build(cfg2)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    base = 1779081406
    cme, mt5 = _matched(base, n=2, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme, "NQ": []}, {"SPX500": mt5, "NDX100": []}
    )
    cfg = _make_config(tmp_path, cme_root, mt5_root, bucket_sizes="15s", dry_run=True)
    res = build(cfg)
    assert res.manifest_path is None
    assert not list(cfg.output_root.rglob("part-*.parquet"))
