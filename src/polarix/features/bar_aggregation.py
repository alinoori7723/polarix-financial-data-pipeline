from __future__ import annotations

from typing import Mapping

import polars as pl

_BUCKET_SUFFIX_NS: Mapping[str, int] = {"ms": 1000000, "s": 1000000000, "m": 60 * 1000000000}


def parse_bucket_size_ns(spec: str) -> int:
    s = spec.strip().lower()
    if not s:
        raise ValueError("empty bucket size")
    for suf in sorted(_BUCKET_SUFFIX_NS, key=lambda x: -len(x)):
        if s.endswith(suf):
            num = s[: -len(suf)].strip()
            if not num:
                raise ValueError(f"bucket size missing number: {spec!r}")
            try:
                value = int(num)
            except ValueError as exc:
                raise ValueError(f"bucket size must be integer-prefixed: {spec!r}") from exc
            if value <= 0:
                raise ValueError(f"bucket size must be > 0: {spec!r}")
            return value * _BUCKET_SUFFIX_NS[suf]
    raise ValueError(f"unknown bucket-size suffix: {spec!r}")


def parse_bucket_sizes(spec: str) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        out.append((item, parse_bucket_size_ns(item)))
    if not out:
        raise ValueError("bucket-sizes must contain at least one entry")
    seen: set[int] = set()
    deduped: list[tuple[str, int]] = []
    for label, ns in out:
        if ns in seen:
            continue
        seen.add(ns)
        deduped.append((label, ns))
    return tuple(deduped)


CME_BAR_COLUMNS = (
    "symbol",
    "bucket_size",
    "bucket_start_utc_ns",
    "bucket_end_utc_ns",
    "trade_count",
    "total_volume",
    "buy_volume",
    "sell_volume",
    "unknown_aggressor_volume",
    "buy_trade_count",
    "sell_trade_count",
    "unknown_aggressor_count",
    "net_signed_volume",
    "signed_volume_ratio",
    "vwap",
    "first_trade_price",
    "last_trade_price",
    "high_trade_price",
    "low_trade_price",
    "has_cme_data",
)
_CME_BAR_DTYPES: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "bucket_size": pl.String,
    "bucket_start_utc_ns": pl.Int64,
    "bucket_end_utc_ns": pl.Int64,
    "trade_count": pl.Int64,
    "total_volume": pl.Int64,
    "buy_volume": pl.Int64,
    "sell_volume": pl.Int64,
    "unknown_aggressor_volume": pl.Int64,
    "buy_trade_count": pl.Int64,
    "sell_trade_count": pl.Int64,
    "unknown_aggressor_count": pl.Int64,
    "net_signed_volume": pl.Int64,
    "signed_volume_ratio": pl.Float64,
    "vwap": pl.Float64,
    "first_trade_price": pl.Float64,
    "last_trade_price": pl.Float64,
    "high_trade_price": pl.Float64,
    "low_trade_price": pl.Float64,
    "has_cme_data": pl.Boolean,
}


def aggregate_cme_bars(
    reference_trades: pl.DataFrame, *, bucket_size_ns: int, bucket_label: str
) -> pl.DataFrame:
    required = (
        "symbol",
        "event_time_utc_ns",
        "price",
        "size",
        "aggressor_side",
        "is_reference_trade_valid",
    )
    missing = [c for c in required if c not in reference_trades.columns]
    if missing:
        raise ValueError(f"reference_trades missing required columns: {missing}")
    if reference_trades.height == 0:
        return pl.DataFrame(schema=_CME_BAR_DTYPES)
    df = (
        reference_trades.filter(pl.col("is_reference_trade_valid"))
        .with_columns(
            (pl.col("event_time_utc_ns") // bucket_size_ns * bucket_size_ns).alias(
                "bucket_start_utc_ns"
            )
        )
        .sort(["symbol", "event_time_utc_ns"])
    )
    is_buy = pl.col("aggressor_side") == "BUY"
    is_sell = pl.col("aggressor_side") == "SELL"
    is_unk = pl.col("aggressor_side") == "UNKNOWN"
    bars = df.group_by(["symbol", "bucket_start_utc_ns"], maintain_order=True).agg(
        pl.len().alias("trade_count"),
        pl.col("size").sum().alias("total_volume"),
        pl.when(is_buy).then(pl.col("size")).otherwise(0).sum().alias("buy_volume"),
        pl.when(is_sell).then(pl.col("size")).otherwise(0).sum().alias("sell_volume"),
        pl.when(is_unk).then(pl.col("size")).otherwise(0).sum().alias("unknown_aggressor_volume"),
        pl.when(is_buy).then(1).otherwise(0).sum().alias("buy_trade_count"),
        pl.when(is_sell).then(1).otherwise(0).sum().alias("sell_trade_count"),
        pl.when(is_unk).then(1).otherwise(0).sum().alias("unknown_aggressor_count"),
        (pl.col("price") * pl.col("size")).sum().alias("_pv_sum"),
        pl.col("price").first().alias("first_trade_price"),
        pl.col("price").last().alias("last_trade_price"),
        pl.col("price").max().alias("high_trade_price"),
        pl.col("price").min().alias("low_trade_price"),
    )
    bars = bars.with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("net_signed_volume"),
        pl.when(pl.col("total_volume") > 0)
        .then((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("total_volume"))
        .otherwise(None)
        .alias("signed_volume_ratio"),
        pl.when(pl.col("total_volume") > 0)
        .then(pl.col("_pv_sum") / pl.col("total_volume"))
        .otherwise(None)
        .alias("vwap"),
        pl.lit(bucket_label).alias("bucket_size"),
        (pl.col("bucket_start_utc_ns") + bucket_size_ns).alias("bucket_end_utc_ns"),
        pl.lit(True).alias("has_cme_data"),
    ).drop("_pv_sum")
    return bars.select(list(CME_BAR_COLUMNS)).sort(["symbol", "bucket_start_utc_ns"])


MT5_BAR_COLUMNS = (
    "symbol",
    "bucket_size",
    "bucket_start_utc_ns",
    "bucket_end_utc_ns",
    "tick_count",
    "join_safe_tick_count",
    "join_safe_tick_ratio",
    "mid_first",
    "mid_last",
    "mid_min",
    "mid_max",
    "mid_mean",
    "mid_twap",
    "spread_price_min",
    "spread_price_mean",
    "spread_price_p50",
    "spread_price_p95",
    "spread_price_max",
    "spread_points_min",
    "spread_points_mean",
    "spread_points_p50",
    "spread_points_p95",
    "spread_points_max",
    "residual_ms_p50",
    "residual_ms_p95",
    "residual_ms_max",
    "latency_outlier_count",
    "has_mt5_data",
)
_MT5_BAR_DTYPES: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "bucket_size": pl.String,
    "bucket_start_utc_ns": pl.Int64,
    "bucket_end_utc_ns": pl.Int64,
    "tick_count": pl.Int64,
    "join_safe_tick_count": pl.Int64,
    "join_safe_tick_ratio": pl.Float64,
    "mid_first": pl.Float64,
    "mid_last": pl.Float64,
    "mid_min": pl.Float64,
    "mid_max": pl.Float64,
    "mid_mean": pl.Float64,
    "mid_twap": pl.Float64,
    "spread_price_min": pl.Float64,
    "spread_price_mean": pl.Float64,
    "spread_price_p50": pl.Float64,
    "spread_price_p95": pl.Float64,
    "spread_price_max": pl.Float64,
    "spread_points_min": pl.Float64,
    "spread_points_mean": pl.Float64,
    "spread_points_p50": pl.Float64,
    "spread_points_p95": pl.Float64,
    "spread_points_max": pl.Int64,
    "residual_ms_p50": pl.Float64,
    "residual_ms_p95": pl.Float64,
    "residual_ms_max": pl.Int64,
    "latency_outlier_count": pl.Int64,
    "has_mt5_data": pl.Boolean,
}


def aggregate_mt5_bars(
    silver: pl.DataFrame, *, bucket_size_ns: int, bucket_label: str
) -> pl.DataFrame:
    required = (
        "symbol",
        "time_msc_utc_ms",
        "is_join_safe",
        "mid",
        "spread_price",
        "spread_points",
        "residual_ms",
        "is_latency_outlier",
    )
    missing = [c for c in required if c not in silver.columns]
    if missing:
        raise ValueError(f"silver missing required columns: {missing}")
    if silver.height == 0:
        return pl.DataFrame(schema=_MT5_BAR_DTYPES)
    df = silver.with_columns(
        (pl.col("time_msc_utc_ms").cast(pl.Int64) * 1000000).alias("time_msc_utc_ns")
    )
    df = df.with_columns(
        (pl.col("time_msc_utc_ns") // bucket_size_ns * bucket_size_ns).alias("bucket_start_utc_ns")
    ).sort(["symbol", "bucket_start_utc_ns", "time_msc_utc_ns"])
    next_t = pl.col("time_msc_utc_ns").shift(-1).over(["symbol", "bucket_start_utc_ns"])
    bucket_end = pl.col("bucket_start_utc_ns") + bucket_size_ns
    weight_ns = (pl.coalesce([next_t, bucket_end]) - pl.col("time_msc_utc_ns")).clip(lower_bound=0)
    df = df.with_columns(weight_ns.alias("_weight_ns"))
    bars = df.group_by(["symbol", "bucket_start_utc_ns"], maintain_order=True).agg(
        pl.len().alias("tick_count"),
        pl.col("is_join_safe").cast(pl.Int64).sum().alias("join_safe_tick_count"),
        pl.col("mid").first().alias("mid_first"),
        pl.col("mid").last().alias("mid_last"),
        pl.col("mid").min().alias("mid_min"),
        pl.col("mid").max().alias("mid_max"),
        pl.col("mid").mean().alias("mid_mean"),
        (pl.col("mid") * pl.col("_weight_ns")).sum().alias("_mid_w_sum"),
        pl.col("_weight_ns").sum().alias("_w_sum"),
        pl.col("spread_price").min().alias("spread_price_min"),
        pl.col("spread_price").mean().alias("spread_price_mean"),
        pl.col("spread_price").quantile(0.5).alias("spread_price_p50"),
        pl.col("spread_price").quantile(0.95).alias("spread_price_p95"),
        pl.col("spread_price").max().alias("spread_price_max"),
        pl.col("spread_points").min().alias("spread_points_min"),
        pl.col("spread_points").mean().alias("spread_points_mean"),
        pl.col("spread_points").quantile(0.5).alias("spread_points_p50"),
        pl.col("spread_points").quantile(0.95).alias("spread_points_p95"),
        pl.col("spread_points").max().alias("spread_points_max"),
        pl.col("residual_ms").quantile(0.5).alias("residual_ms_p50"),
        pl.col("residual_ms").quantile(0.95).alias("residual_ms_p95"),
        pl.col("residual_ms").max().alias("residual_ms_max"),
        pl.col("is_latency_outlier").cast(pl.Int64).sum().alias("latency_outlier_count"),
    )
    bars = bars.with_columns(
        pl.when(pl.col("_w_sum") > 0)
        .then(pl.col("_mid_w_sum") / pl.col("_w_sum"))
        .otherwise(pl.col("mid_mean"))
        .alias("mid_twap"),
        pl.when(pl.col("tick_count") > 0)
        .then(pl.col("join_safe_tick_count") / pl.col("tick_count"))
        .otherwise(None)
        .alias("join_safe_tick_ratio"),
        pl.lit(bucket_label).alias("bucket_size"),
        (pl.col("bucket_start_utc_ns") + bucket_size_ns).alias("bucket_end_utc_ns"),
        pl.lit(True).alias("has_mt5_data"),
    ).drop(["_mid_w_sum", "_w_sum"])
    return bars.select(list(MT5_BAR_COLUMNS)).sort(["symbol", "bucket_start_utc_ns"])
