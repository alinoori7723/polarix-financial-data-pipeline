from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import polars as pl

from polarix.features.bar_aggregation import (
    aggregate_cme_bars,
    aggregate_mt5_bars,
    parse_bucket_sizes,
)

DEFAULT_SYMBOL_MAP: Mapping[str, str] = {"ES": "SPX500", "NQ": "NDX100"}
DEFAULT_BUCKET_SIZES = ("1s", "5s", "15s", "60s")
DEFAULT_MIN_JOIN_SAFE_TICK_RATIO = 0.5
DEFAULT_PASS_BAR_ALIGNMENT_RATE = 0.8
DEFAULT_PASS_CME_VOLUME_ALIGNMENT_RATIO = 0.8
REASON_NO_CME_DATA = "no_cme_data"
REASON_NO_MT5_DATA = "no_mt5_data"
REASON_ZERO_CME_VOLUME = "zero_cme_volume"
REASON_ZERO_MT5_TICKS = "zero_mt5_ticks"
REASON_LOW_JOIN_SAFE_TICK_RATIO = "low_join_safe_tick_ratio"
REASON_SPREAD_TOO_WIDE = "spread_too_wide"
REASON_INVALID_PRICE = "invalid_price"
REASON_MISSING_REQUIRED_COLUMNS = "missing_required_columns"
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
ALIGNED_BAR_COLUMNS = (
    "cme_symbol",
    "mt5_symbol",
    "bucket_size",
    "bucket_start_utc_ns",
    "bucket_end_utc_ns",
    "has_cme_data",
    "has_mt5_data",
    "cme_trade_count",
    "cme_total_volume",
    "cme_buy_volume",
    "cme_sell_volume",
    "cme_unknown_aggressor_volume",
    "cme_net_signed_volume",
    "cme_signed_volume_ratio",
    "cme_vwap",
    "cme_first_trade_price",
    "cme_last_trade_price",
    "cme_high_trade_price",
    "cme_low_trade_price",
    "mt5_tick_count",
    "mt5_join_safe_tick_count",
    "mt5_join_safe_tick_ratio",
    "mt5_mid_twap",
    "mt5_mid_mean",
    "mt5_mid_first",
    "mt5_mid_last",
    "mt5_mid_min",
    "mt5_mid_max",
    "mt5_spread_price_mean",
    "mt5_spread_price_p50",
    "mt5_spread_price_p95",
    "mt5_spread_price_max",
    "mt5_spread_points_mean",
    "mt5_spread_points_p50",
    "mt5_spread_points_p95",
    "mt5_spread_points_max",
    "mt5_residual_ms_p50",
    "mt5_residual_ms_p95",
    "mt5_residual_ms_max",
    "basis_cme_vwap_to_mt5_twap",
    "abs_basis",
    "basis_bps",
    "is_bar_aligned",
    "bar_reject_reason",
)


class BarAlignmentError(RuntimeError):
    pass


@dataclass
class BarAlignmentThresholds:
    min_join_safe_tick_ratio: float = DEFAULT_MIN_JOIN_SAFE_TICK_RATIO
    max_spread_price_threshold: Optional[float] = None
    pass_bar_alignment_rate: float = DEFAULT_PASS_BAR_ALIGNMENT_RATE
    pass_cme_volume_alignment_ratio: float = DEFAULT_PASS_CME_VOLUME_ALIGNMENT_RATIO


@dataclass
class BarAlignmentConfig:
    date: str
    cme_root: Path
    mt5_root: Path
    reports_root: Path
    symbol_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))
    bucket_sizes: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: parse_bucket_sizes(",".join(DEFAULT_BUCKET_SIZES))
    )
    thresholds: BarAlignmentThresholds = field(default_factory=BarAlignmentThresholds)
    write_buckets: bool = False
    pass_buckets_min_size: str = "5s"

    def __post_init__(self) -> None:
        self.cme_root = Path(self.cme_root).resolve()
        self.mt5_root = Path(self.mt5_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()


def _glob_parts(root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def _read_polars(files: list[Path], required_cols: Iterable[str]) -> Optional[pl.DataFrame]:
    if not files:
        return None
    frames = []
    for f in files:
        df = pl.read_parquet(f)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise BarAlignmentError(f"{f}: missing required columns {missing}")
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


def _percentile(values: Iterable[float], pct: float) -> Optional[float]:
    arr = sorted(
        (
            float(v)
            for v in values
            if v is not None and (not (isinstance(v, float) and math.isnan(v)))
        )
    )
    if not arr:
        return None
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] * (c - k) + arr[c] * (k - f)


def _summary(values: Iterable[float]) -> dict[str, Optional[float]]:
    arr = [
        float(v) for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))
    ]
    if not arr:
        return {k: None for k in ("min", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": float(min(arr)),
        "mean": float(sum(arr) / len(arr)),
        "p50": _percentile(arr, 0.5),
        "p95": _percentile(arr, 0.95),
        "p99": _percentile(arr, 0.99),
        "max": float(max(arr)),
    }


def join_and_classify_bars(
    cme_bars: pl.DataFrame,
    mt5_bars: pl.DataFrame,
    *,
    cme_symbol: str,
    mt5_symbol: str,
    bucket_label: str,
    thresholds: BarAlignmentThresholds,
) -> pl.DataFrame:
    cme = cme_bars.rename(
        {
            "trade_count": "cme_trade_count",
            "total_volume": "cme_total_volume",
            "buy_volume": "cme_buy_volume",
            "sell_volume": "cme_sell_volume",
            "unknown_aggressor_volume": "cme_unknown_aggressor_volume",
            "net_signed_volume": "cme_net_signed_volume",
            "signed_volume_ratio": "cme_signed_volume_ratio",
            "vwap": "cme_vwap",
            "first_trade_price": "cme_first_trade_price",
            "last_trade_price": "cme_last_trade_price",
            "high_trade_price": "cme_high_trade_price",
            "low_trade_price": "cme_low_trade_price",
        }
    ).drop(
        [
            "buy_trade_count",
            "sell_trade_count",
            "unknown_aggressor_count",
            "bucket_end_utc_ns",
            "bucket_size",
            "symbol",
        ]
    )
    mt5 = mt5_bars.rename(
        {
            "tick_count": "mt5_tick_count",
            "join_safe_tick_count": "mt5_join_safe_tick_count",
            "join_safe_tick_ratio": "mt5_join_safe_tick_ratio",
            "mid_twap": "mt5_mid_twap",
            "mid_mean": "mt5_mid_mean",
            "mid_first": "mt5_mid_first",
            "mid_last": "mt5_mid_last",
            "mid_min": "mt5_mid_min",
            "mid_max": "mt5_mid_max",
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
            "bucket_end_utc_ns",
            "bucket_size",
            "symbol",
            "latency_outlier_count",
            "mt5_spread_price_min",
            "mt5_spread_points_min",
        ],
        strict=False,
    )
    joined = cme.join(mt5, on="bucket_start_utc_ns", how="full", coalesce=True)
    joined = joined.with_columns(
        pl.col("cme_trade_count").is_not_null().alias("has_cme_data"),
        pl.col("mt5_tick_count").is_not_null().alias("has_mt5_data"),
    )
    bucket_size_ns = joined.select(pl.col("bucket_start_utc_ns").min()).item() or 0
    if cme_bars.height > 0:
        bucket_size_ns = int(cme_bars["bucket_end_utc_ns"][0]) - int(
            cme_bars["bucket_start_utc_ns"][0]
        )
    elif mt5_bars.height > 0:
        bucket_size_ns = int(mt5_bars["bucket_end_utc_ns"][0]) - int(
            mt5_bars["bucket_start_utc_ns"][0]
        )
    else:
        bucket_size_ns = 0
    joined = joined.with_columns(
        pl.lit(cme_symbol).alias("cme_symbol"),
        pl.lit(mt5_symbol).alias("mt5_symbol"),
        pl.lit(bucket_label).alias("bucket_size"),
        (pl.col("bucket_start_utc_ns") + bucket_size_ns).alias("bucket_end_utc_ns"),
        (pl.col("mt5_mid_twap") - pl.col("cme_vwap")).alias("basis_cme_vwap_to_mt5_twap"),
    )
    joined = joined.with_columns(
        pl.col("basis_cme_vwap_to_mt5_twap").abs().alias("abs_basis"),
        pl.when(pl.col("cme_vwap") > 0)
        .then(pl.col("basis_cme_vwap_to_mt5_twap") / pl.col("cme_vwap") * 10000)
        .otherwise(None)
        .alias("basis_bps"),
    )
    has_cme = pl.col("has_cme_data")
    has_mt5 = pl.col("has_mt5_data")
    zero_cme_vol = pl.col("cme_total_volume").fill_null(0) == 0
    zero_mt5_ticks = pl.col("mt5_tick_count").fill_null(0) == 0
    low_jstr = (
        pl.col("mt5_join_safe_tick_ratio").fill_null(0.0) < thresholds.min_join_safe_tick_ratio
    )
    invalid_price = (
        pl.col("cme_vwap").is_null() | pl.col("mt5_mid_twap").is_null() | (pl.col("cme_vwap") <= 0)
    )
    reject = (
        pl.when(~has_cme)
        .then(pl.lit(REASON_NO_CME_DATA))
        .when(~has_mt5)
        .then(pl.lit(REASON_NO_MT5_DATA))
        .when(zero_cme_vol)
        .then(pl.lit(REASON_ZERO_CME_VOLUME))
        .when(zero_mt5_ticks)
        .then(pl.lit(REASON_ZERO_MT5_TICKS))
        .when(low_jstr)
        .then(pl.lit(REASON_LOW_JOIN_SAFE_TICK_RATIO))
        .when(invalid_price)
        .then(pl.lit(REASON_INVALID_PRICE))
    )
    if thresholds.max_spread_price_threshold is not None:
        thr = thresholds.max_spread_price_threshold
        too_wide = pl.col("mt5_spread_price_max").fill_null(thr + 1) > thr
        reject = reject.when(too_wide).then(pl.lit(REASON_SPREAD_TOO_WIDE))
    reject = reject.otherwise(None).alias("bar_reject_reason")
    joined = joined.with_columns(reject)
    joined = joined.with_columns(pl.col("bar_reject_reason").is_null().alias("is_bar_aligned"))
    for col in ALIGNED_BAR_COLUMNS:
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).alias(col))
    return joined.select(list(ALIGNED_BAR_COLUMNS)).sort("bucket_start_utc_ns")


def _per_bucket_metrics(bars: pl.DataFrame) -> dict:
    total_buckets = bars.height
    buckets_with_cme = int(bars["has_cme_data"].cast(pl.Int64).sum() or 0)
    buckets_with_mt5 = int(bars["has_mt5_data"].cast(pl.Int64).sum() or 0)
    aligned_bucket_count = int(bars["is_bar_aligned"].cast(pl.Int64).sum() or 0)
    cme_volume_total = int(bars["cme_total_volume"].fill_null(0).sum() or 0)
    cme_volume_in_aligned = int(
        bars.filter(pl.col("is_bar_aligned"))["cme_total_volume"].fill_null(0).sum() or 0
    )
    jstr_values = bars["mt5_join_safe_tick_ratio"].to_list()
    jstr_clean = [v for v in jstr_values if v is not None]
    jstr_mean = sum(jstr_clean) / len(jstr_clean) if jstr_clean else None
    bar_alignment_rate = aligned_bucket_count / total_buckets if total_buckets else None
    cme_volume_alignment_ratio = (
        cme_volume_in_aligned / cme_volume_total if cme_volume_total else None
    )
    reject_counts: dict[str, int] = {}
    reject_volume: dict[str, int] = {}
    for reason in (
        REASON_NO_CME_DATA,
        REASON_NO_MT5_DATA,
        REASON_ZERO_CME_VOLUME,
        REASON_ZERO_MT5_TICKS,
        REASON_LOW_JOIN_SAFE_TICK_RATIO,
        REASON_SPREAD_TOO_WIDE,
        REASON_INVALID_PRICE,
    ):
        sub = bars.filter(pl.col("bar_reject_reason") == reason)
        reject_counts[reason] = sub.height
        reject_volume[reason] = int(sub["cme_total_volume"].fill_null(0).sum() or 0)
    aligned = bars.filter(pl.col("is_bar_aligned"))
    basis_summary = _summary(aligned["basis_cme_vwap_to_mt5_twap"].to_list())
    abs_basis_summary = _summary(aligned["abs_basis"].to_list())
    basis_bps_summary = _summary(aligned["basis_bps"].to_list())
    spread_max_summary = _summary(aligned["mt5_spread_price_max"].to_list())
    spread_p95_summary = _summary(aligned["mt5_spread_price_p95"].to_list())
    residual_p95_summary = _summary(aligned["mt5_residual_ms_p95"].to_list())
    return {
        "total_buckets": total_buckets,
        "buckets_with_cme_data": buckets_with_cme,
        "buckets_with_mt5_data": buckets_with_mt5,
        "aligned_bucket_count": aligned_bucket_count,
        "bar_alignment_rate": bar_alignment_rate,
        "cme_volume_total": cme_volume_total,
        "cme_volume_in_aligned_buckets": cme_volume_in_aligned,
        "cme_volume_alignment_ratio": cme_volume_alignment_ratio,
        "mt5_tick_count_total": int(bars["mt5_tick_count"].fill_null(0).sum() or 0),
        "mt5_join_safe_tick_ratio_mean": jstr_mean,
        "basis_summary": basis_summary,
        "abs_basis_summary": abs_basis_summary,
        "basis_bps_summary": basis_bps_summary,
        "spread_price_max_summary": spread_max_summary,
        "spread_price_p95_summary": spread_p95_summary,
        "residual_ms_p95_summary": residual_p95_summary,
        "reject_reason_counts": reject_counts,
        "reject_reason_cme_volume": reject_volume,
    }


@dataclass
class BarAlignmentResult:
    report: dict
    aligned_bars: dict[str, dict[str, pl.DataFrame]]


def build_bar_alignment_quality_report(config: BarAlignmentConfig) -> BarAlignmentResult:
    warnings: list[str] = []
    errors: list[str] = []
    per_symbol: dict[str, dict] = {}
    per_bucket_size: dict[str, dict] = {b[0]: {} for b in config.bucket_sizes}
    aligned_bars_out: dict[str, dict[str, pl.DataFrame]] = {}
    cme_present_any = False
    mt5_present_any = False
    real_overlap_present = False
    for cme_symbol, mt5_symbol in config.symbol_map.items():
        cme_files = _glob_parts(config.cme_root, cme_symbol, config.date)
        mt5_files = _glob_parts(config.mt5_root, mt5_symbol, config.date)
        cme_present = bool(cme_files)
        mt5_present = bool(mt5_files)
        cme_present_any = cme_present_any or cme_present
        mt5_present_any = mt5_present_any or mt5_present
        sym_entry: dict = {
            "cme_symbol": cme_symbol,
            "mt5_symbol": mt5_symbol,
            "cme_present": cme_present,
            "mt5_present": mt5_present,
        }
        aligned_bars_out[cme_symbol] = {}
        if not cme_present or not mt5_present:
            sym_entry["status"] = (
                "MISSING_CME_REFERENCE" if not cme_present else "MISSING_MT5_SILVER"
            )
            per_symbol[cme_symbol] = sym_entry
            continue
        try:
            cme_df = _read_polars(cme_files, CME_REQUIRED_COLS)
            mt5_df = _read_polars(mt5_files, MT5_REQUIRED_COLS)
        except BarAlignmentError as exc:
            errors.append(str(exc))
            sym_entry["status"] = "MISSING_REQUIRED_COLUMNS"
            sym_entry["error"] = str(exc)
            per_symbol[cme_symbol] = sym_entry
            continue
        assert cme_df is not None and mt5_df is not None
        cme_df = cme_df.filter(pl.col("symbol") == cme_symbol)
        mt5_df = mt5_df.filter(pl.col("symbol") == mt5_symbol)
        cme_min = cme_df["event_time_utc_ns"].min()
        cme_max = cme_df["event_time_utc_ns"].max()
        mt5_min_ns = (mt5_df["time_msc_utc_ms"].min() or 0) * 1000000
        mt5_max_ns = (mt5_df["time_msc_utc_ms"].max() or 0) * 1000000
        ov_lo = max(cme_min or 0, mt5_min_ns)
        ov_hi = min(cme_max or 0, mt5_max_ns)
        has_overlap = (
            cme_min is not None and mt5_min_ns and (cme_max is not None) and (ov_lo <= ov_hi)
        )
        real_overlap_present = real_overlap_present or has_overlap
        sym_entry["status"] = "OK" if has_overlap else "NO_OVERLAP_WINDOW"
        sym_entry["cme_window_ns"] = [cme_min, cme_max]
        sym_entry["mt5_window_ns"] = [mt5_min_ns, mt5_max_ns]
        sym_entry["overlap_window_ns"] = [ov_lo, ov_hi] if has_overlap else None
        sym_entry["by_bucket_size"] = {}
        if not has_overlap:
            per_symbol[cme_symbol] = sym_entry
            continue
        for bucket_label, bucket_ns in config.bucket_sizes:
            cme_bars = aggregate_cme_bars(
                cme_df, bucket_size_ns=bucket_ns, bucket_label=bucket_label
            )
            mt5_bars = aggregate_mt5_bars(
                mt5_df, bucket_size_ns=bucket_ns, bucket_label=bucket_label
            )
            aligned = join_and_classify_bars(
                cme_bars,
                mt5_bars,
                cme_symbol=cme_symbol,
                mt5_symbol=mt5_symbol,
                bucket_label=bucket_label,
                thresholds=config.thresholds,
            )
            aligned = aligned.filter(pl.col("bucket_start_utc_ns") + bucket_ns > ov_lo).filter(
                pl.col("bucket_start_utc_ns") <= ov_hi
            )
            aligned_bars_out[cme_symbol][bucket_label] = aligned
            metrics = _per_bucket_metrics(aligned)
            sym_entry["by_bucket_size"][bucket_label] = metrics
            per_bucket_size[bucket_label].setdefault(cme_symbol, metrics)
        per_symbol[cme_symbol] = sym_entry
    decision: str
    decision_reason: Optional[str] = None
    pass_buckets = [
        (label, ns)
        for label, ns in config.bucket_sizes
        if ns >= _bucket_ns_for_label(config.pass_buckets_min_size)
    ]
    if not cme_present_any:
        decision = "FAIL"
        decision_reason = "MISSING_CME_REFERENCE"
    elif not mt5_present_any:
        decision = "FAIL"
        decision_reason = "MISSING_MT5_SILVER"
    elif errors:
        decision = "FAIL"
        decision_reason = "MISSING_REQUIRED_COLUMNS"
    elif not real_overlap_present:
        decision = "FAIL"
        decision_reason = "MISSING_OVERLAP_DATA"
    else:
        ok_symbols = [
            s for s in config.symbol_map if (per_symbol.get(s) or {}).get("status") == "OK"
        ]
        pass_found = False
        for label, _ns in pass_buckets:
            symbols_ok_here = 0
            for s in ok_symbols:
                metrics = (per_symbol[s].get("by_bucket_size") or {}).get(label)
                if not metrics:
                    continue
                bar_rate = metrics.get("bar_alignment_rate") or 0.0
                vol_ratio = metrics.get("cme_volume_alignment_ratio") or 0.0
                if (
                    bar_rate >= config.thresholds.pass_bar_alignment_rate
                    and vol_ratio >= config.thresholds.pass_cme_volume_alignment_ratio
                ):
                    symbols_ok_here += 1
            if symbols_ok_here == len(config.symbol_map):
                pass_found = True
                break
        if pass_found:
            decision = "PASS"
        else:
            decision = "PARTIAL"
            warnings.append(
                f"No bucket size satisfies both bar_alignment_rate>={config.thresholds.pass_bar_alignment_rate} AND cme_volume_alignment_ratio>={config.thresholds.pass_cme_volume_alignment_ratio} for ALL symbols at the same time."
            )
    overall: dict = {
        "total_buckets": 0,
        "aligned_bucket_count": 0,
        "cme_volume_total": 0,
        "cme_volume_in_aligned_buckets": 0,
    }
    for s, e in per_symbol.items():
        for label, metrics in (e.get("by_bucket_size") or {}).items():
            for k in (
                "total_buckets",
                "aligned_bucket_count",
                "cme_volume_total",
                "cme_volume_in_aligned_buckets",
            ):
                overall[k] += int(metrics.get(k) or 0)
    overall["bar_alignment_rate"] = (
        overall["aligned_bucket_count"] / overall["total_buckets"]
        if overall["total_buckets"]
        else None
    )
    overall["cme_volume_alignment_ratio"] = (
        overall["cme_volume_in_aligned_buckets"] / overall["cme_volume_total"]
        if overall["cme_volume_total"]
        else None
    )
    report = {
        "date": config.date,
        "cme_root": str(config.cme_root),
        "mt5_root": str(config.mt5_root),
        "reports_root": str(config.reports_root),
        "symbol_map": dict(config.symbol_map),
        "bucket_sizes": [b[0] for b in config.bucket_sizes],
        "thresholds": {
            "min_join_safe_tick_ratio": config.thresholds.min_join_safe_tick_ratio,
            "max_spread_price_threshold": config.thresholds.max_spread_price_threshold,
            "pass_bar_alignment_rate": config.thresholds.pass_bar_alignment_rate,
            "pass_cme_volume_alignment_ratio": config.thresholds.pass_cme_volume_alignment_ratio,
            "pass_buckets_min_size": config.pass_buckets_min_size,
        },
        "quality_decision": decision,
        "decision_reason": decision_reason,
        "errors": errors,
        "warnings": warnings,
        "per_symbol": per_symbol,
        "per_bucket_size": per_bucket_size,
        "overall": overall,
        "real_overlap_present": real_overlap_present,
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "code_version": _git_hash(),
    }
    return BarAlignmentResult(report=report, aligned_bars=aligned_bars_out)


def _bucket_ns_for_label(label: str) -> int:
    from polarix.features.bar_aggregation import parse_bucket_size_ns

    return parse_bucket_size_ns(label)


def _ns_to_iso(ns: Optional[int]) -> Optional[str]:
    if ns is None:
        return None
    return _dt.datetime.fromtimestamp(int(ns) / 1000000000.0, tz=_dt.timezone.utc).isoformat()


def render_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Polarix Bar-Alignment Feasibility Report  --  date={report['date']}")
    lines.append("=" * 80)
    lines.append(f"Decision                : {report['quality_decision']}")
    if report.get("decision_reason"):
        lines.append(f"Decision reason         : {report['decision_reason']}")
    lines.append(f"Real overlap present    : {report['real_overlap_present']}")
    lines.append(f"Bucket sizes            : {', '.join(report['bucket_sizes'])}")
    lines.append(f"Symbol map              : {report['symbol_map']}")
    thr = report["thresholds"]
    lines.append(
        f"Thresholds              : min_join_safe={thr['min_join_safe_tick_ratio']}, pass_bar_rate>={thr['pass_bar_alignment_rate']}, pass_vol_ratio>={thr['pass_cme_volume_alignment_ratio']}, pass_min_bucket={thr['pass_buckets_min_size']}, max_spread={thr['max_spread_price_threshold']}"
    )
    lines.append("")
    for sym, e in report["per_symbol"].items():
        lines.append("-" * 80)
        lines.append(f"SYMBOL: {sym} -> {e.get('mt5_symbol')}")
        lines.append(f"  status                  : {e.get('status')}")
        if e.get("status") not in ("OK",):
            if e.get("error"):
                lines.append(f"  error                   : {e['error']}")
            continue
        cwin = e.get("cme_window_ns") or [None, None]
        mwin = e.get("mt5_window_ns") or [None, None]
        owin = e.get("overlap_window_ns") or [None, None]
        lines.append(f"  cme_window              : {_ns_to_iso(cwin[0])} -> {_ns_to_iso(cwin[1])}")
        lines.append(f"  mt5_window              : {_ns_to_iso(mwin[0])} -> {_ns_to_iso(mwin[1])}")
        lines.append(f"  overlap_window          : {_ns_to_iso(owin[0])} -> {_ns_to_iso(owin[1])}")
        for label, metrics in (e.get("by_bucket_size") or {}).items():
            lines.append(f"  bucket {label}:")
            mr = metrics.get("bar_alignment_rate")
            vr = metrics.get("cme_volume_alignment_ratio")
            jr = metrics.get("mt5_join_safe_tick_ratio_mean")
            lines.append(
                "    total_buckets={tb} aligned={ab} bar_alignment_rate={mr}".format(
                    tb=metrics.get("total_buckets"),
                    ab=metrics.get("aligned_bucket_count"),
                    mr="-" if mr is None else f"{mr:.4f}",
                )
            )
            lines.append(
                "    cme_volume_total={cv} aligned_vol={av} cme_volume_alignment_ratio={vr}".format(
                    cv=metrics.get("cme_volume_total"),
                    av=metrics.get("cme_volume_in_aligned_buckets"),
                    vr="-" if vr is None else f"{vr:.4f}",
                )
            )
            lines.append(
                f"    mt5_join_safe_tick_ratio_mean={('-' if jr is None else f'{jr:.4f}')}"
            )
            bsum = metrics.get("basis_summary") or {}
            ssum = metrics.get("spread_price_max_summary") or {}
            bbps = metrics.get("basis_bps_summary") or {}
            lines.append(
                f"    basis             : min={bsum.get('min')} mean={bsum.get('mean')} p50={bsum.get('p50')} p95={bsum.get('p95')} p99={bsum.get('p99')} max={bsum.get('max')}"
            )
            lines.append(
                f"    basis_bps         : min={bbps.get('min')} mean={bbps.get('mean')} p50={bbps.get('p50')} p95={bbps.get('p95')} p99={bbps.get('p99')} max={bbps.get('max')}"
            )
            lines.append(
                f"    mt5_spread_max   : min={ssum.get('min')} mean={ssum.get('mean')} p50={ssum.get('p50')} p95={ssum.get('p95')} p99={ssum.get('p99')} max={ssum.get('max')}"
            )
            rc = metrics.get("reject_reason_counts") or {}
            rv = metrics.get("reject_reason_cme_volume") or {}
            non_zero = {k: rc[k] for k in rc if rc[k]}
            lines.append(f"    reject_reasons    : {non_zero}")
            non_zero_v = {k: rv[k] for k in rv if rv[k]}
            lines.append(f"    reject_vol        : {non_zero_v}")
    lines.append("-" * 80)
    overall = report.get("overall") or {}
    lines.append("OVERALL")
    lines.append(f"  total_buckets       : {overall.get('total_buckets')}")
    lines.append(f"  aligned_bucket_count: {overall.get('aligned_bucket_count')}")
    bar_rate = overall.get("bar_alignment_rate")
    vol_ratio = overall.get("cme_volume_alignment_ratio")
    lines.append(f"  bar_alignment_rate  : {('-' if bar_rate is None else f'{bar_rate:.4f}')}")
    lines.append(f"  cme_volume_alignment: {('-' if vol_ratio is None else f'{vol_ratio:.4f}')}")
    if report.get("warnings"):
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report.get("errors"):
        lines.append("Errors:")
        for w in report["errors"]:
            lines.append(f"  - {w}")
    lines.append("-" * 80)
    if report["quality_decision"] == "FAIL":
        lines.append(
            f"NEXT: {report.get('decision_reason')} -- restore the missing dataset or verify CME/MT5 overlap before re-running."
        )
    elif report["quality_decision"] == "PARTIAL":
        lines.append(
            "NEXT: bar-level alignment is diagnostically usable for the bucket sizes shown above but does not yet satisfy PASS thresholds. Inspect per-bucket metrics and consider whether 5m/15m horizons (out of Phase 2C scope) would be a more honest acceptance horizon."
        )
    else:
        lines.append(
            "NEXT: bar-alignment feasibility ACCEPTED at the indicated bucket size. This authorizes the future design of bar-level reference features. It does NOT authorize trading, model training, or CVD aggregation."
        )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_reports(
    config: BarAlignmentConfig, result: BarAlignmentResult
) -> tuple[Path, Path, Optional[Path]]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"bar_alignment_quality_{config.date}.json"
    txt_path = config.reports_root / f"bar_alignment_quality_{config.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(result.report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(result.report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    buckets_path: Optional[Path] = None
    if config.write_buckets:
        rows: list[pl.DataFrame] = []
        for sym, by_bucket in result.aligned_bars.items():
            for _label, df in by_bucket.items():
                if df.height > 0:
                    rows.append(df)
        if rows:
            combined = pl.concat(rows, how="vertical_relaxed")
            buckets_path = config.reports_root / f"bar_alignment_buckets_{config.date}.parquet"
            tmp = buckets_path.with_suffix(".parquet.tmp")
            combined.write_parquet(tmp, compression="zstd")
            os.replace(tmp, buckets_path)
    return (json_path, txt_path, buckets_path)


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
