from __future__ import annotations

import polars as pl
import pytest

from polarix.features.bar_aggregation import (
    aggregate_cme_bars,
    aggregate_mt5_bars,
    parse_bucket_size_ns,
    parse_bucket_sizes,
)

SEC = 1000000000


def test_parse_bucket_size_ns_seconds() -> None:
    assert parse_bucket_size_ns("1s") == SEC
    assert parse_bucket_size_ns("5s") == 5 * SEC
    assert parse_bucket_size_ns("60s") == 60 * SEC


def test_parse_bucket_size_ns_ms_and_min() -> None:
    assert parse_bucket_size_ns("250ms") == 250 * 1000000
    assert parse_bucket_size_ns("1m") == 60 * SEC


@pytest.mark.parametrize("bad", ["", "s", "0s", "-1s", "1x", "abc"])
def test_parse_bucket_size_ns_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_bucket_size_ns(bad)


def test_parse_bucket_sizes_default_ordering_and_dedup() -> None:
    sizes = parse_bucket_sizes("1s, 5s, 15s, 60s")
    assert [s[0] for s in sizes] == ["1s", "5s", "15s", "60s"]
    sizes2 = parse_bucket_sizes("1s,1s,5s")
    assert [s[0] for s in sizes2] == ["1s", "5s"]


def test_parse_bucket_sizes_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_bucket_sizes(",,")


def _cme_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "event_time_utc_ns": pl.Int64,
            "price": pl.Float64,
            "size": pl.Int64,
            "aggressor_side": pl.String,
            "is_reference_trade_valid": pl.Boolean,
        },
    )


def test_cme_vwap_uses_volume_weighted_average() -> None:
    df = _cme_df(
        [
            {
                "symbol": "ES",
                "event_time_utc_ns": 0,
                "price": 100.0,
                "size": 1,
                "aggressor_side": "BUY",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": 500000000,
                "price": 200.0,
                "size": 9,
                "aggressor_side": "SELL",
                "is_reference_trade_valid": True,
            },
        ]
    )
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars.height == 1
    assert bars["vwap"][0] == pytest.approx(190.0)
    assert bars["vwap"][0] != pytest.approx(150.0)


def test_cme_buy_sell_volumes_separated() -> None:
    df = _cme_df(
        [
            {
                "symbol": "ES",
                "event_time_utc_ns": 0,
                "price": 100.0,
                "size": 5,
                "aggressor_side": "BUY",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": 100000,
                "price": 100.0,
                "size": 3,
                "aggressor_side": "SELL",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": 200000,
                "price": 100.0,
                "size": 2,
                "aggressor_side": "UNKNOWN",
                "is_reference_trade_valid": True,
            },
        ]
    )
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    row = bars.row(0, named=True)
    assert row["buy_volume"] == 5
    assert row["sell_volume"] == 3
    assert row["unknown_aggressor_volume"] == 2
    assert row["net_signed_volume"] == 2
    assert row["total_volume"] == 10
    assert row["signed_volume_ratio"] == pytest.approx(0.2)


def test_cme_unknown_aggressor_volume_accounted() -> None:
    df = _cme_df(
        [
            {
                "symbol": "ES",
                "event_time_utc_ns": 0,
                "price": 100.0,
                "size": 7,
                "aggressor_side": "UNKNOWN",
                "is_reference_trade_valid": True,
            }
        ]
    )
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    row = bars.row(0, named=True)
    assert row["unknown_aggressor_volume"] == 7
    assert row["buy_volume"] == 0
    assert row["sell_volume"] == 0
    assert row["net_signed_volume"] == 0
    assert row["signed_volume_ratio"] == pytest.approx(0.0)


def test_cme_empty_input_returns_empty() -> None:
    df = _cme_df([])
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars.height == 0


def test_cme_skips_invalid_rows() -> None:
    df = _cme_df(
        [
            {
                "symbol": "ES",
                "event_time_utc_ns": 0,
                "price": 100.0,
                "size": 5,
                "aggressor_side": "BUY",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": 100,
                "price": 0.0,
                "size": 0,
                "aggressor_side": "UNKNOWN",
                "is_reference_trade_valid": False,
            },
        ]
    )
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars["total_volume"][0] == 5
    assert bars["trade_count"][0] == 1


def test_cme_buckets_split_correctly_across_seconds() -> None:
    df = _cme_df(
        [
            {
                "symbol": "ES",
                "event_time_utc_ns": 0,
                "price": 100.0,
                "size": 1,
                "aggressor_side": "BUY",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": SEC - 1,
                "price": 101.0,
                "size": 2,
                "aggressor_side": "BUY",
                "is_reference_trade_valid": True,
            },
            {
                "symbol": "ES",
                "event_time_utc_ns": SEC,
                "price": 200.0,
                "size": 3,
                "aggressor_side": "SELL",
                "is_reference_trade_valid": True,
            },
        ]
    )
    bars = aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars.height == 2
    assert bars["trade_count"].to_list() == [2, 1]
    assert bars["total_volume"].to_list() == [3, 3]


def test_cme_required_columns_missing_raises() -> None:
    df = pl.DataFrame({"symbol": ["ES"], "price": [100.0]})
    with pytest.raises(ValueError):
        aggregate_cme_bars(df, bucket_size_ns=SEC, bucket_label="1s")


def _mt5_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
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
    )


def test_mt5_mid_twap_multi_tick_interval_weighted() -> None:
    df = _mt5_df(
        [
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 0,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 100,
                "is_join_safe": True,
                "mid": 2.0,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 900,
                "is_join_safe": True,
                "mid": 3.0,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
        ]
    )
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars.height == 1
    assert bars["mid_twap"][0] == pytest.approx(2.0)


def test_mt5_mid_twap_distinguishes_from_mean() -> None:
    df = _mt5_df(
        [
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 0,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 900,
                "is_join_safe": True,
                "mid": 10.0,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
        ]
    )
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars["mid_twap"][0] == pytest.approx(1.9)
    assert bars["mid_mean"][0] == pytest.approx(5.5)


def test_mt5_mid_twap_single_tick() -> None:
    df = _mt5_df(
        [
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 500,
                "is_join_safe": True,
                "mid": 7.5,
                "spread_price": 0.5,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            }
        ]
    )
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars["mid_twap"][0] == pytest.approx(7.5)


def test_mt5_spread_summaries() -> None:
    df = _mt5_df(
        [
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 0,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.1,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 100,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.5,
                "spread_points": 5,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 200,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 2.0,
                "spread_points": 20,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
        ]
    )
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    row = bars.row(0, named=True)
    assert row["spread_price_min"] == pytest.approx(0.1)
    assert row["spread_price_max"] == pytest.approx(2.0)
    assert row["spread_price_mean"] == pytest.approx((0.1 + 0.5 + 2.0) / 3)


def test_mt5_join_safe_tick_ratio() -> None:
    df = _mt5_df(
        [
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 0,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.1,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 100,
                "is_join_safe": False,
                "mid": 1.0,
                "spread_price": 0.5,
                "spread_points": 5,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 200,
                "is_join_safe": True,
                "mid": 1.0,
                "spread_price": 0.1,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
            {
                "symbol": "SPX500",
                "time_msc_utc_ms": 300,
                "is_join_safe": False,
                "mid": 1.0,
                "spread_price": 0.1,
                "spread_points": 1,
                "residual_ms": 0,
                "is_latency_outlier": False,
            },
        ]
    )
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    row = bars.row(0, named=True)
    assert row["join_safe_tick_count"] == 2
    assert row["tick_count"] == 4
    assert row["join_safe_tick_ratio"] == pytest.approx(0.5)


def test_mt5_empty_input_returns_empty() -> None:
    df = _mt5_df([])
    bars = aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
    assert bars.height == 0


def test_mt5_required_columns_missing_raises() -> None:
    df = pl.DataFrame({"symbol": ["SPX500"], "mid": [1.0]})
    with pytest.raises(ValueError):
        aggregate_mt5_bars(df, bucket_size_ns=SEC, bucket_label="1s")
