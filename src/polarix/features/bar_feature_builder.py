from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import secrets
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import polars as pl

from polarix.features.bar_aggregation import (
    aggregate_cme_bars,
    aggregate_mt5_bars,
    parse_bucket_size_ns,
    parse_bucket_sizes,
)
from polarix.features.bar_feature_contract import (
    FEATURE_QUALITY_DIAGNOSTIC_ONLY,
    FEATURE_QUALITY_INVALID_PRICE,
    FEATURE_QUALITY_LOW_JOIN_SAFE_RATIO,
    FEATURE_QUALITY_MISSING_CME,
    FEATURE_QUALITY_MISSING_MT5,
    FEATURE_QUALITY_OK,
    FEATURE_QUALITY_SPREAD_EXTREME,
    FEATURE_QUALITY_ZERO_VOLUME,
    FeatureContract,
    assert_contract_invariants,
)

BUILDER_VERSION = "0.2.0"
ZSCORE_WINDOW = 20
DATA_LAYER = "gold_candidate_features"
DEFAULT_SYMBOL_MAP: Mapping[str, str] = {"ES": "SPX500", "NQ": "NDX100"}
DEFAULT_BUCKET_SIZES = ("15s", "60s")
DIAGNOSTIC_BUCKET_SIZE = "5s"
DEFAULT_MIN_JOIN_SAFE_TICK_RATIO = 0.5
DEFAULT_MIN_VOLUME_ALIGNMENT_RATIO = 0.8
CME_REQUIRED_COLS = (
    "symbol",
    "event_time_utc_ns",
    "price",
    "size",
    "aggressor_side",
    "is_reference_trade_valid",
)
MT5_REQUIRED_COLS = (
    "symbol",
    "time_msc_utc_ms",
    "is_join_safe",
    "mid",
    "spread_price",
    "spread_points",
    "residual_ms",
    "is_latency_outlier",
)


class BuilderError(RuntimeError):
    pass


@dataclass
class BuilderConfig:
    date: str
    cme_root: Path
    mt5_root: Path
    output_root: Path
    reports_root: Path
    symbol_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))
    bucket_sizes: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: parse_bucket_sizes(",".join(DEFAULT_BUCKET_SIZES))
    )
    include_diagnostic_5s: bool = False
    min_join_safe_tick_ratio: float = DEFAULT_MIN_JOIN_SAFE_TICK_RATIO
    min_volume_alignment_ratio: float = DEFAULT_MIN_VOLUME_ALIGNMENT_RATIO
    max_spread_price_threshold: Optional[float] = None
    dry_run: bool = False
    force: bool = False

    def __post_init__(self) -> None:
        self.cme_root = Path(self.cme_root).resolve()
        self.mt5_root = Path(self.mt5_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        if self.include_diagnostic_5s:
            labels = {label for label, _ in self.bucket_sizes}
            if DIAGNOSTIC_BUCKET_SIZE not in labels:
                self.bucket_sizes = (
                    (DIAGNOSTIC_BUCKET_SIZE, parse_bucket_size_ns(DIAGNOSTIC_BUCKET_SIZE)),
                ) + tuple(self.bucket_sizes)


def _glob_parts(root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def _read_polars(files: list[Path], required_cols: tuple[str, ...]) -> Optional[pl.DataFrame]:
    if not files:
        return None
    frames = []
    for f in files:
        df = pl.read_parquet(f)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise BuilderError(f"{f}: missing required columns {missing}")
        frames.append(df)
    return pl.concat(frames, how="vertical_relaxed")


def _git_hash() -> Optional[str]:
    try:
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


def _cme_size_summaries(cme_df: pl.DataFrame, bucket_size_ns: int) -> pl.DataFrame:
    if cme_df.height == 0:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "bucket_start_utc_ns": pl.Int64,
                "cme_max_single_trade_volume": pl.Int64,
                "cme_trade_size_p99": pl.Float64,
                "cme_trade_size_mean": pl.Float64,
                "cme_trade_size_std": pl.Float64,
            }
        )
    df = cme_df.filter(pl.col("is_reference_trade_valid")).with_columns(
        (pl.col("event_time_utc_ns") // bucket_size_ns * bucket_size_ns).alias(
            "bucket_start_utc_ns"
        )
    )
    return df.group_by(["symbol", "bucket_start_utc_ns"], maintain_order=True).agg(
        pl.col("size").max().alias("cme_max_single_trade_volume"),
        pl.col("size").cast(pl.Float64).quantile(0.99).alias("cme_trade_size_p99"),
        pl.col("size").cast(pl.Float64).mean().alias("cme_trade_size_mean"),
        pl.col("size").cast(pl.Float64).std().alias("cme_trade_size_std"),
    )


def _zscore(series: pl.Expr) -> pl.Expr:
    mean = series.rolling_mean(window_size=ZSCORE_WINDOW, min_samples=2)
    std = series.rolling_std(window_size=ZSCORE_WINDOW, min_samples=2)
    return pl.when(std.is_not_null() & (std > 0)).then((series - mean) / std).otherwise(None)


def _build_features_one_pair(
    cme_df: Optional[pl.DataFrame],
    mt5_df: Optional[pl.DataFrame],
    *,
    cme_symbol: str,
    mt5_symbol: str,
    bucket_label: str,
    bucket_size_ns: int,
    config: BuilderConfig,
    overlap_ns: Optional[tuple[int, int]],
) -> pl.DataFrame:
    is_diag = bucket_label == DIAGNOSTIC_BUCKET_SIZE
    bucket_size_seconds = bucket_size_ns / 1000000000
    if cme_df is None or cme_df.height == 0:
        cme_bars = aggregate_cme_bars(
            cme_df
            if cme_df is not None
            else pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "event_time_utc_ns": pl.Int64,
                    "price": pl.Float64,
                    "size": pl.Int64,
                    "aggressor_side": pl.String,
                    "is_reference_trade_valid": pl.Boolean,
                }
            ),
            bucket_size_ns=bucket_size_ns,
            bucket_label=bucket_label,
        )
        cme_size_stats = _cme_size_summaries(
            pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "event_time_utc_ns": pl.Int64,
                    "price": pl.Float64,
                    "size": pl.Int64,
                    "aggressor_side": pl.String,
                    "is_reference_trade_valid": pl.Boolean,
                }
            ),
            bucket_size_ns=bucket_size_ns,
        )
    else:
        cme_bars = aggregate_cme_bars(
            cme_df, bucket_size_ns=bucket_size_ns, bucket_label=bucket_label
        )
        cme_size_stats = _cme_size_summaries(cme_df, bucket_size_ns=bucket_size_ns)
    if mt5_df is None or mt5_df.height == 0:
        mt5_bars = aggregate_mt5_bars(
            mt5_df
            if mt5_df is not None
            else pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "time_msc_utc_ms": pl.Int64,
                    "is_join_safe": pl.Boolean,
                    "mid": pl.Float64,
                    "spread_price": pl.Float64,
                    "spread_points": pl.Int64,
                    "residual_ms": pl.Int64,
                    "is_latency_outlier": pl.Boolean,
                }
            ),
            bucket_size_ns=bucket_size_ns,
            bucket_label=bucket_label,
        )
    else:
        mt5_bars = aggregate_mt5_bars(
            mt5_df, bucket_size_ns=bucket_size_ns, bucket_label=bucket_label
        )
    cme_bars_renamed = cme_bars.rename(
        {
            "first_trade_price": "cme_open_price",
            "last_trade_price": "cme_close_price",
            "high_trade_price": "cme_high_price",
            "low_trade_price": "cme_low_price",
            "vwap": "cme_vwap",
            "trade_count": "cme_trade_count",
            "total_volume": "cme_total_volume",
            "buy_volume": "cme_buy_volume",
            "sell_volume": "cme_sell_volume",
            "unknown_aggressor_volume": "cme_unknown_aggressor_volume",
            "net_signed_volume": "cme_net_signed_volume",
            "signed_volume_ratio": "cme_signed_volume_ratio",
        }
    ).drop(
        [
            "buy_trade_count",
            "sell_trade_count",
            "unknown_aggressor_count",
            "bucket_size",
            "has_cme_data",
            "bucket_end_utc_ns",
            "symbol",
        ]
    )
    mt5_bars_renamed = mt5_bars.rename(
        {
            "tick_count": "mt5_tick_count",
            "join_safe_tick_count": "mt5_join_safe_tick_count",
            "join_safe_tick_ratio": "mt5_join_safe_tick_ratio",
            "mid_first": "mt5_mid_open",
            "mid_last": "mt5_mid_close",
            "mid_min": "mt5_mid_low",
            "mid_max": "mt5_mid_high",
            "mid_mean": "mt5_mid_mean",
            "mid_twap": "mt5_mid_twap",
            "spread_price_min": "mt5_spread_price_min",
            "spread_price_mean": "mt5_spread_price_mean",
            "spread_price_p50": "mt5_spread_price_p50",
            "spread_price_p95": "mt5_spread_price_p95",
            "spread_price_max": "mt5_spread_price_max",
            "spread_points_min": "mt5_spread_points_min",
            "spread_points_mean": "mt5_spread_points_mean",
            "spread_points_p50": "mt5_spread_points_p50",
            "spread_points_p95": "mt5_spread_points_p95",
            "spread_points_max": "mt5_spread_points_max",
            "residual_ms_p50": "mt5_residual_ms_p50",
            "residual_ms_p95": "mt5_residual_ms_p95",
            "residual_ms_max": "mt5_residual_ms_max",
        }
    ).drop(
        [
            "bucket_size",
            "has_mt5_data",
            "bucket_end_utc_ns",
            "symbol",
            "latency_outlier_count",
            "mt5_spread_price_min",
            "mt5_spread_points_min",
        ],
        strict=False,
    )
    joined = cme_bars_renamed.join(
        mt5_bars_renamed, on="bucket_start_utc_ns", how="full", coalesce=True
    ).join(cme_size_stats.drop("symbol"), on="bucket_start_utc_ns", how="left")
    if overlap_ns is not None:
        lo, hi = overlap_ns
        joined = joined.filter(pl.col("bucket_start_utc_ns") + bucket_size_ns > lo).filter(
            pl.col("bucket_start_utc_ns") <= hi
        )
    joined = joined.with_columns(
        pl.col("cme_trade_count").is_not_null().alias("has_cme_data"),
        pl.col("mt5_tick_count").is_not_null().alias("has_mt5_data"),
        pl.lit(cme_symbol).alias("cme_symbol"),
        pl.lit(mt5_symbol).alias("mt5_symbol"),
        pl.lit(f"{cme_symbol}_{mt5_symbol}").alias("symbol_pair"),
        pl.lit(bucket_label).alias("bucket_size"),
        (pl.col("bucket_start_utc_ns") + bucket_size_ns).alias("bucket_end_utc_ns"),
        pl.lit(config.date).alias("date"),
        pl.lit(DATA_LAYER).alias("data_layer"),
        pl.lit(BUILDER_VERSION).alias("builder_version"),
    )
    joined = joined.with_columns(pl.col("bucket_end_utc_ns").alias("feature_timestamp_utc_ns"))
    cme_total_filled = pl.col("cme_total_volume").fill_null(0)
    mt5_tick_filled = pl.col("mt5_tick_count").fill_null(0)
    aligned_pre = (
        pl.col("has_cme_data")
        & pl.col("has_mt5_data")
        & (cme_total_filled > 0)
        & (mt5_tick_filled > 0)
    )
    joined = joined.with_columns(
        pl.when(aligned_pre).then(1.0).otherwise(0.0).alias("cme_volume_alignment_ratio")
    )
    has_cme = pl.col("has_cme_data")
    has_mt5 = pl.col("has_mt5_data")
    cme_total_filled = pl.col("cme_total_volume").fill_null(0)
    mt5_tick_filled = pl.col("mt5_tick_count").fill_null(0)
    zero_volume = cme_total_filled <= 0
    low_jstr = pl.col("mt5_join_safe_tick_ratio").fill_null(0.0) < config.min_join_safe_tick_ratio
    invalid_price = (
        pl.col("cme_close_price").is_null()
        | pl.col("mt5_mid_close").is_null()
        | (pl.col("cme_close_price") <= 0)
    )
    spread_max_col = pl.col("mt5_spread_price_max")
    spread_extreme = (
        pl.lit(False)
        if config.max_spread_price_threshold is None
        else spread_max_col.fill_null(config.max_spread_price_threshold + 1)
        > config.max_spread_price_threshold
    )
    bar_reject_reason = (
        pl.when(~has_cme)
        .then(pl.lit("no_cme_data"))
        .when(~has_mt5)
        .then(pl.lit("no_mt5_data"))
        .when(zero_volume)
        .then(pl.lit("zero_cme_volume"))
        .when(mt5_tick_filled <= 0)
        .then(pl.lit("zero_mt5_ticks"))
        .when(low_jstr)
        .then(pl.lit("low_join_safe_tick_ratio"))
        .when(invalid_price)
        .then(pl.lit("invalid_price"))
        .when(spread_extreme)
        .then(pl.lit("spread_too_wide"))
        .otherwise(None)
    )
    joined = joined.with_columns(bar_reject_reason.alias("bar_reject_reason"))
    joined = joined.with_columns(pl.col("bar_reject_reason").is_null().alias("is_bar_aligned"))
    feature_quality_flag = (
        pl.when(~has_cme)
        .then(pl.lit(FEATURE_QUALITY_MISSING_CME))
        .when(~has_mt5)
        .then(pl.lit(FEATURE_QUALITY_MISSING_MT5))
        .when(zero_volume)
        .then(pl.lit(FEATURE_QUALITY_ZERO_VOLUME))
        .when(invalid_price)
        .then(pl.lit(FEATURE_QUALITY_INVALID_PRICE))
        .when(low_jstr)
        .then(pl.lit(FEATURE_QUALITY_LOW_JOIN_SAFE_RATIO))
        .when(spread_extreme)
        .then(pl.lit(FEATURE_QUALITY_SPREAD_EXTREME))
        .when(pl.lit(is_diag))
        .then(pl.lit(FEATURE_QUALITY_DIAGNOSTIC_ONLY))
        .otherwise(pl.lit(FEATURE_QUALITY_OK))
    )
    joined = joined.with_columns(feature_quality_flag.alias("feature_quality_flag"))
    joined = joined.with_columns(
        ((pl.col("feature_quality_flag") == FEATURE_QUALITY_OK) & pl.lit(not is_diag)).alias(
            "is_model_eligible_candidate"
        )
    )
    joined = joined.sort(["symbol_pair", "bucket_size", "bucket_start_utc_ns"])
    joined = joined.with_columns(
        (
            (
                pl.col("cme_close_price")
                - pl.col("cme_close_price").shift(1).over(["symbol_pair", "bucket_size"])
            )
            / pl.col("cme_close_price").shift(1).over(["symbol_pair", "bucket_size"])
        ).alias("cme_return_close_to_close"),
        (
            (pl.col("cme_vwap") - pl.col("cme_vwap").shift(1).over(["symbol_pair", "bucket_size"]))
            / pl.col("cme_vwap").shift(1).over(["symbol_pair", "bucket_size"])
        ).alias("cme_vwap_return"),
        pl.when(pl.col("cme_low_price") > 0)
        .then(
            (pl.col("cme_high_price") - pl.col("cme_low_price")) / pl.col("cme_low_price") * 10000
        )
        .otherwise(None)
        .alias("cme_high_low_range_bps"),
        pl.when(pl.col("cme_total_volume") > 0)
        .then(pl.col("cme_buy_volume") / pl.col("cme_total_volume"))
        .otherwise(None)
        .alias("cme_buy_volume_ratio"),
        pl.when(pl.col("cme_total_volume") > 0)
        .then(pl.col("cme_sell_volume") / pl.col("cme_total_volume"))
        .otherwise(None)
        .alias("cme_sell_volume_ratio"),
        pl.when(pl.col("cme_total_volume") > 0)
        .then(pl.col("cme_unknown_aggressor_volume") / pl.col("cme_total_volume"))
        .otherwise(None)
        .alias("cme_unknown_aggressor_volume_ratio"),
        pl.when(pl.col("cme_trade_count") > 0)
        .then(
            pl.col("cme_total_volume").cast(pl.Float64) / pl.col("cme_trade_count").cast(pl.Float64)
        )
        .otherwise(None)
        .alias("cme_volume_per_trade"),
        (pl.col("cme_trade_count").cast(pl.Float64) / bucket_size_seconds).alias(
            "cme_trade_intensity"
        ),
    )
    joined = joined.with_columns(
        (
            (
                pl.col("mt5_mid_close")
                - pl.col("mt5_mid_close").shift(1).over(["symbol_pair", "bucket_size"])
            )
            / pl.col("mt5_mid_close").shift(1).over(["symbol_pair", "bucket_size"])
        ).alias("mt5_mid_return_close_to_close"),
        pl.when(pl.col("mt5_mid_low") > 0)
        .then((pl.col("mt5_mid_high") - pl.col("mt5_mid_low")) / pl.col("mt5_mid_low") * 10000)
        .otherwise(None)
        .alias("mt5_mid_range_bps"),
    )
    joined = joined.with_columns(
        (pl.col("mt5_mid_close") - pl.col("cme_close_price")).alias("basis_close"),
        pl.when(pl.col("cme_close_price") > 0)
        .then(
            (pl.col("mt5_mid_close") - pl.col("cme_close_price"))
            / pl.col("cme_close_price")
            * 10000
        )
        .otherwise(None)
        .alias("basis_close_bps"),
        (pl.col("mt5_mid_twap") - pl.col("cme_vwap")).alias("diagnostic_basis_vwap_twap"),
        pl.when(pl.col("cme_vwap") > 0)
        .then((pl.col("mt5_mid_twap") - pl.col("cme_vwap")) / pl.col("cme_vwap") * 10000)
        .otherwise(None)
        .alias("diagnostic_basis_vwap_twap_bps"),
    )
    joined = joined.with_columns(
        (
            pl.col("basis_close")
            - pl.col("basis_close").shift(1).over(["symbol_pair", "bucket_size"])
        ).alias("basis_change"),
        (
            pl.col("basis_close_bps")
            - pl.col("basis_close_bps").shift(1).over(["symbol_pair", "bucket_size"])
        ).alias("basis_change_bps"),
        (pl.col("mt5_mid_return_close_to_close") - pl.col("cme_return_close_to_close")).alias(
            "return_diff_close_to_close"
        ),
    )
    joined = joined.with_columns(
        _zscore(pl.col("cme_total_volume").cast(pl.Float64))
        .over(["symbol_pair", "bucket_size"])
        .alias("cme_volume_zscore_by_symbol_bucket"),
        _zscore(pl.col("cme_trade_count").cast(pl.Float64))
        .over(["symbol_pair", "bucket_size"])
        .alias("cme_trade_count_zscore_by_symbol_bucket"),
        _zscore(pl.col("cme_high_low_range_bps"))
        .over(["symbol_pair", "bucket_size"])
        .alias("cme_range_zscore_by_symbol_bucket"),
        _zscore(pl.col("mt5_spread_price_p95"))
        .over(["symbol_pair", "bucket_size"])
        .alias("mt5_spread_zscore_by_symbol_bucket"),
        _zscore(pl.col("mt5_tick_count").cast(pl.Float64))
        .over(["symbol_pair", "bucket_size"])
        .alias("mt5_tick_count_zscore_by_symbol_bucket"),
    )
    row_ids = [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"polarix:{BUILDER_VERSION}:{config.date}:{cme_symbol}:{mt5_symbol}:{bucket_label}:{start}",
            )
        )
        for start in joined["bucket_start_utc_ns"]
    ]
    joined = joined.with_columns(pl.Series("feature_row_id", row_ids, dtype=pl.String))
    contract = FeatureContract(builder_version=BUILDER_VERSION)
    assert_contract_invariants(contract)
    all_cols = list(contract.all_columns())
    for col in all_cols:
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).alias(col))
    return joined.select(all_cols)


@dataclass
class BuildResult:
    config: BuilderConfig
    written_paths: list[Path]
    manifest_path: Optional[Path]
    rows_by_pair_bucket: dict[str, dict[str, int]]
    missing_input: bool
    error: Optional[str]
    started_at_utc: str
    ended_at_utc: str
    feature_rows: dict[str, dict[str, pl.DataFrame]]


def build(config: BuilderConfig) -> BuildResult:
    started_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    written_paths: list[Path] = []
    rows_by_pair_bucket: dict[str, dict[str, int]] = {}
    feature_rows_out: dict[str, dict[str, pl.DataFrame]] = {}
    any_pair_built = False
    any_missing_input = True
    for cme_symbol, mt5_symbol in config.symbol_map.items():
        pair_label = f"{cme_symbol}_{mt5_symbol}"
        feature_rows_out[pair_label] = {}
        rows_by_pair_bucket[pair_label] = {}
        cme_files = _glob_parts(config.cme_root, cme_symbol, config.date)
        mt5_files = _glob_parts(config.mt5_root, mt5_symbol, config.date)
        if not cme_files or not mt5_files:
            continue
        any_missing_input = False
        cme_df = _read_polars(cme_files, CME_REQUIRED_COLS)
        mt5_df = _read_polars(mt5_files, MT5_REQUIRED_COLS)
        assert cme_df is not None and mt5_df is not None
        cme_df = cme_df.filter(pl.col("symbol") == cme_symbol)
        mt5_df = mt5_df.filter(pl.col("symbol") == mt5_symbol)
        cme_min = cme_df["event_time_utc_ns"].min()
        cme_max = cme_df["event_time_utc_ns"].max()
        mt5_min_ns = (mt5_df["time_msc_utc_ms"].min() or 0) * 1000000
        mt5_max_ns = (mt5_df["time_msc_utc_ms"].max() or 0) * 1000000
        overlap_ns: Optional[tuple[int, int]] = None
        if cme_min is not None and cme_max is not None and mt5_min_ns and mt5_max_ns:
            lo = max(int(cme_min), int(mt5_min_ns))
            hi = min(int(cme_max), int(mt5_max_ns))
            if lo <= hi:
                overlap_ns = (lo, hi)
        for bucket_label, bucket_ns in config.bucket_sizes:
            rows = _build_features_one_pair(
                cme_df,
                mt5_df,
                cme_symbol=cme_symbol,
                mt5_symbol=mt5_symbol,
                bucket_label=bucket_label,
                bucket_size_ns=bucket_ns,
                config=config,
                overlap_ns=overlap_ns,
            )
            feature_rows_out[pair_label][bucket_label] = rows
            rows_by_pair_bucket[pair_label][bucket_label] = rows.height
            any_pair_built = any_pair_built or rows.height > 0
    if any_missing_input and (not any_pair_built):
        ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return BuildResult(
            config=config,
            written_paths=[],
            manifest_path=None,
            rows_by_pair_bucket=rows_by_pair_bucket,
            missing_input=True,
            error="MISSING_INPUT_DATA",
            started_at_utc=started_at_utc,
            ended_at_utc=ended,
            feature_rows={},
        )
    if config.dry_run:
        ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        return BuildResult(
            config=config,
            written_paths=[],
            manifest_path=None,
            rows_by_pair_bucket=rows_by_pair_bucket,
            missing_input=False,
            error=None,
            started_at_utc=started_at_utc,
            ended_at_utc=ended,
            feature_rows=feature_rows_out,
        )
    for pair_label, by_bucket in feature_rows_out.items():
        for bucket_label, df in by_bucket.items():
            out_dir = (
                config.output_root
                / f"symbol_pair={pair_label}"
                / f"date={config.date}"
                / f"bucket={bucket_label}"
            )
            if out_dir.exists():
                existing_parts = [
                    p
                    for p in out_dir.iterdir()
                    if p.name.startswith("part-") and p.suffix == ".parquet"
                ]
                if existing_parts and (not config.force):
                    raise FileExistsError(
                        f"Gold feature partition already exists at {out_dir}; pass --force"
                    )
                if config.force:
                    for p in existing_parts:
                        p.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)
            if df.height == 0:
                continue
            name = f"part-00000001-{secrets.token_hex(4)}.parquet"
            out_path = out_dir / name
            tmp_path = out_path.with_suffix(".parquet.tmp")
            df.write_parquet(tmp_path, compression="zstd")
            os.replace(tmp_path, out_path)
            written_paths.append(out_path)
    contract = FeatureContract(builder_version=BUILDER_VERSION)
    manifest_dir = config.output_root / f"date={config.date}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "date": config.date,
        "builder_version": BUILDER_VERSION,
        "code_version": _git_hash(),
        "data_layer": DATA_LAYER,
        "cme_root": str(config.cme_root),
        "mt5_root": str(config.mt5_root),
        "zscore_window_bars": ZSCORE_WINDOW,
        "feature_availability": "bucket_end_utc_ns",
        "output_root": str(config.output_root),
        "symbol_map": dict(config.symbol_map),
        "bucket_sizes": [b[0] for b in config.bucket_sizes],
        "include_diagnostic_5s": config.include_diagnostic_5s,
        "min_join_safe_tick_ratio": config.min_join_safe_tick_ratio,
        "min_volume_alignment_ratio": config.min_volume_alignment_ratio,
        "max_spread_price_threshold": config.max_spread_price_threshold,
        "rows_by_pair_bucket": rows_by_pair_bucket,
        "written_paths": [str(p) for p in written_paths],
        "feature_contract": contract.to_dict(),
    }
    manifest_path = manifest_dir / "bar_features_manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    return BuildResult(
        config=config,
        written_paths=written_paths,
        manifest_path=manifest_path,
        rows_by_pair_bucket=rows_by_pair_bucket,
        missing_input=False,
        error=None,
        started_at_utc=started_at_utc,
        ended_at_utc=ended,
        feature_rows=feature_rows_out,
    )


def parse_symbol_map(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"symbol-map entry {item!r} must contain '='")
        k, v = item.split("=", 1)
        k, v = (k.strip(), v.strip())
        if not k or not v:
            raise ValueError(f"symbol-map entry {item!r} has empty key or value")
        out[k] = v
    if not out:
        raise ValueError("symbol-map must contain at least one entry")
    return out
