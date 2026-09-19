from __future__ import annotations

import math
from typing import Optional, Sequence

import polars as pl

DEFAULT_NEAR_CONSTANT_DISTINCT_THRESHOLD = 2
DEFAULT_HEAVY_TAIL_THRESHOLD = 5.0
DEFAULT_IQR_OUTLIER_K = 1.5
DEFAULT_SMALL_SAMPLE_MIN_ROWS = 5000
MISSINGNESS_COLUMNS = (
    "symbol_pair",
    "bucket_size",
    "feature_column",
    "n_rows",
    "n_non_null",
    "null_count",
    "null_rate",
    "finite_count",
    "finite_rate",
    "inf_count",
    "nan_count",
    "zero_count",
    "zero_rate",
    "distinct_count",
    "near_constant_flag",
    "all_null_flag",
    "all_zero_flag",
)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    try:
        return dtype.is_numeric()
    except Exception:
        return dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )


def _summarize_missing_one_group(
    df: pl.DataFrame, column: str, *, near_constant_distinct_threshold: int
) -> dict:
    n_rows = df.height
    null_count = int(df[column].null_count())
    n_non_null = n_rows - null_count
    if _is_numeric_dtype(df[column].dtype):
        s = df[column]
        finite = s.is_finite()
        if isinstance(finite, pl.Series):
            finite_count = int(finite.fill_null(False).cast(pl.Int64).sum() or 0)
        else:
            finite_count = 0
        inf_count = int(s.is_infinite().fill_null(False).cast(pl.Int64).sum() or 0)
        nan_count = int(s.is_nan().fill_null(False).cast(pl.Int64).sum() or 0)
        zero_count = int(df.filter(pl.col(column) == 0).height)
    else:
        finite_count = n_non_null
        inf_count = 0
        nan_count = 0
        zero_count = 0
    null_rate = null_count / n_rows if n_rows else None
    finite_rate = finite_count / n_rows if n_rows else None
    zero_rate = zero_count / n_rows if n_rows else None
    distinct_count = int(df[column].n_unique())
    near_constant_flag = distinct_count <= near_constant_distinct_threshold
    all_null_flag = null_count == n_rows if n_rows else False
    all_zero_flag = n_non_null > 0 and zero_count == n_non_null
    return {
        "n_rows": n_rows,
        "n_non_null": n_non_null,
        "null_count": null_count,
        "null_rate": null_rate,
        "finite_count": finite_count,
        "finite_rate": finite_rate,
        "inf_count": inf_count,
        "nan_count": nan_count,
        "zero_count": zero_count,
        "zero_rate": zero_rate,
        "distinct_count": distinct_count,
        "near_constant_flag": near_constant_flag,
        "all_null_flag": all_null_flag,
        "all_zero_flag": all_zero_flag,
    }


def compute_missingness_summary(
    df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_keys: Sequence[str] = ("symbol_pair", "bucket_size"),
    near_constant_distinct_threshold: int = DEFAULT_NEAR_CONSTANT_DISTINCT_THRESHOLD,
) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(
            schema={
                c: pl.String if c in ("symbol_pair", "bucket_size", "feature_column") else pl.Int64
                for c in MISSINGNESS_COLUMNS
            }
        )
    rows: list[dict] = []
    group_cols = list(group_keys)
    if not all((c in df.columns for c in group_cols)):
        group_cols = []
    if group_cols:
        groups = df.partition_by(group_cols, as_dict=True)
    else:
        groups = {("__all__",): df}
    for key, sub in groups.items():
        if isinstance(key, tuple):
            key_values = list(key)
        else:
            key_values = [key]
        identity = {
            gc: kv if not isinstance(kv, tuple) else kv[0] for gc, kv in zip(group_cols, key_values)
        }
        for col in feature_columns:
            if col not in sub.columns:
                continue
            stats = _summarize_missing_one_group(
                sub, col, near_constant_distinct_threshold=near_constant_distinct_threshold
            )
            row = {
                "symbol_pair": identity.get("symbol_pair"),
                "bucket_size": identity.get("bucket_size"),
                "feature_column": col,
                **stats,
            }
            rows.append(row)
    return pl.DataFrame(
        rows,
        schema={
            "symbol_pair": pl.String,
            "bucket_size": pl.String,
            "feature_column": pl.String,
            "n_rows": pl.Int64,
            "n_non_null": pl.Int64,
            "null_count": pl.Int64,
            "null_rate": pl.Float64,
            "finite_count": pl.Int64,
            "finite_rate": pl.Float64,
            "inf_count": pl.Int64,
            "nan_count": pl.Int64,
            "zero_count": pl.Int64,
            "zero_rate": pl.Float64,
            "distinct_count": pl.Int64,
            "near_constant_flag": pl.Boolean,
            "all_null_flag": pl.Boolean,
            "all_zero_flag": pl.Boolean,
        },
    )


DISTRIBUTION_COLUMNS = (
    "symbol_pair",
    "bucket_size",
    "feature_column",
    "count",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "max",
    "iqr",
    "robust_scale",
    "skewness",
    "kurtosis",
    "heavy_tail_score",
    "heavy_tail_flag",
    "outlier_rate_iqr",
    "sample_size_warning",
)


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
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


def _skew_and_kurt(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    arr = [
        float(v) for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))
    ]
    n = len(arr)
    if n < 4:
        return (None, None)
    mean = sum(arr) / n
    m2 = sum(((x - mean) ** 2 for x in arr)) / n
    if m2 <= 0:
        return (None, None)
    m3 = sum(((x - mean) ** 3 for x in arr)) / n
    m4 = sum(((x - mean) ** 4 for x in arr)) / n
    skewness = m3 / m2**1.5
    kurtosis = m4 / m2**2 - 3.0
    return (skewness, kurtosis)


def _summarize_distribution_one_group(
    df: pl.DataFrame,
    column: str,
    *,
    heavy_tail_threshold: float,
    iqr_k: float,
    small_sample_min_rows: int,
) -> dict:
    if not _is_numeric_dtype(df[column].dtype):
        return {
            k: None
            for k in (
                "count",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99",
                "max",
                "iqr",
                "robust_scale",
                "skewness",
                "kurtosis",
                "heavy_tail_score",
                "outlier_rate_iqr",
            )
        } | {"heavy_tail_flag": False, "sample_size_warning": True}
    series = df[column].drop_nulls()
    series = series.filter(series.is_finite())
    n = series.len()
    if n == 0:
        return {
            k: None
            for k in (
                "count",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99",
                "max",
                "iqr",
                "robust_scale",
                "skewness",
                "kurtosis",
                "heavy_tail_score",
                "outlier_rate_iqr",
            )
        } | {"heavy_tail_flag": False, "sample_size_warning": True}
    values = series.to_list()

    def _to_float(x):
        return float(x) if x is not None else None

    mean = _to_float(series.mean())
    std = _to_float(series.std())
    mn = _to_float(series.min())
    mx = _to_float(series.max())
    p01 = _percentile(values, 0.01)
    p05 = _percentile(values, 0.05)
    p25 = _percentile(values, 0.25)
    p50 = _percentile(values, 0.5)
    p75 = _percentile(values, 0.75)
    p95 = _percentile(values, 0.95)
    p99 = _percentile(values, 0.99)
    iqr = p75 - p25 if p75 is not None and p25 is not None else None
    robust_scale = iqr
    skewness, kurtosis = _skew_and_kurt(values)
    eps = 1e-12
    if iqr is not None and abs(iqr) > eps and (p99 is not None) and (p50 is not None):
        heavy_tail_score = abs(p99 - p50) / abs(iqr)
    else:
        heavy_tail_score = None
    heavy_tail_flag = heavy_tail_score is not None and heavy_tail_score >= heavy_tail_threshold
    if iqr is not None and iqr > 0:
        lo = p25 - iqr_k * iqr
        hi = p75 + iqr_k * iqr
        outlier_count = sum((1 for v in values if v < lo or v > hi))
        outlier_rate_iqr = outlier_count / n
    else:
        outlier_rate_iqr = None
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": mn,
        "p01": p01,
        "p05": p05,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "iqr": iqr,
        "robust_scale": robust_scale,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "heavy_tail_score": heavy_tail_score,
        "heavy_tail_flag": heavy_tail_flag,
        "outlier_rate_iqr": outlier_rate_iqr,
        "sample_size_warning": n < small_sample_min_rows,
    }


def compute_distribution_summary(
    df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_keys: Sequence[str] = ("symbol_pair", "bucket_size"),
    heavy_tail_threshold: float = DEFAULT_HEAVY_TAIL_THRESHOLD,
    iqr_k: float = DEFAULT_IQR_OUTLIER_K,
    small_sample_min_rows: int = DEFAULT_SMALL_SAMPLE_MIN_ROWS,
) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(schema={c: pl.Float64 for c in DISTRIBUTION_COLUMNS})
    rows: list[dict] = []
    group_cols = list(group_keys)
    if not all((c in df.columns for c in group_cols)):
        group_cols = []
    groups = df.partition_by(group_cols, as_dict=True) if group_cols else {("__all__",): df}
    for key, sub in groups.items():
        if isinstance(key, tuple):
            key_values = list(key)
        else:
            key_values = [key]
        identity = {
            gc: kv if not isinstance(kv, tuple) else kv[0] for gc, kv in zip(group_cols, key_values)
        }
        for col in feature_columns:
            if col not in sub.columns:
                continue
            stats = _summarize_distribution_one_group(
                sub,
                col,
                heavy_tail_threshold=heavy_tail_threshold,
                iqr_k=iqr_k,
                small_sample_min_rows=small_sample_min_rows,
            )
            rows.append(
                {
                    "symbol_pair": identity.get("symbol_pair"),
                    "bucket_size": identity.get("bucket_size"),
                    "feature_column": col,
                    **stats,
                }
            )
    schema: dict[str, pl.DataType] = {
        "symbol_pair": pl.String,
        "bucket_size": pl.String,
        "feature_column": pl.String,
        "count": pl.Int64,
        "mean": pl.Float64,
        "std": pl.Float64,
        "min": pl.Float64,
        "p01": pl.Float64,
        "p05": pl.Float64,
        "p25": pl.Float64,
        "p50": pl.Float64,
        "p75": pl.Float64,
        "p95": pl.Float64,
        "p99": pl.Float64,
        "max": pl.Float64,
        "iqr": pl.Float64,
        "robust_scale": pl.Float64,
        "skewness": pl.Float64,
        "kurtosis": pl.Float64,
        "heavy_tail_score": pl.Float64,
        "heavy_tail_flag": pl.Boolean,
        "outlier_rate_iqr": pl.Float64,
        "sample_size_warning": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


STABILITY_COLUMNS = (
    "symbol_pair",
    "bucket_size",
    "feature_column",
    "first_half_count",
    "second_half_count",
    "first_half_mean",
    "second_half_mean",
    "first_half_std",
    "second_half_std",
    "abs_mean_shift",
    "relative_mean_shift",
)


def compute_intra_sample_stability(
    df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_keys: Sequence[str] = ("symbol_pair", "bucket_size"),
    order_by: str = "bucket_start_utc_ns",
) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(schema={c: pl.Float64 for c in STABILITY_COLUMNS})
    rows: list[dict] = []
    group_cols = list(group_keys)
    if not all((c in df.columns for c in group_cols)):
        group_cols = []
    groups = df.partition_by(group_cols, as_dict=True) if group_cols else {("__all__",): df}
    for key, sub in groups.items():
        if isinstance(key, tuple):
            key_values = list(key)
        else:
            key_values = [key]
        identity = {
            gc: kv if not isinstance(kv, tuple) else kv[0] for gc, kv in zip(group_cols, key_values)
        }
        if order_by in sub.columns:
            sub = sub.sort(order_by)
        mid = sub.height // 2
        first = sub.head(mid) if mid > 0 else sub.head(0)
        second = sub.tail(sub.height - mid)
        for col in feature_columns:
            if col not in sub.columns:
                continue
            if not _is_numeric_dtype(sub[col].dtype):
                continue

            def _to_float(x):
                return float(x) if x is not None else None

            f_mean = _to_float(first[col].drop_nulls().mean()) if first.height else None
            s_mean = _to_float(second[col].drop_nulls().mean()) if second.height else None
            f_std = _to_float(first[col].drop_nulls().std()) if first.height else None
            s_std = _to_float(second[col].drop_nulls().std()) if second.height else None
            abs_shift = abs(s_mean - f_mean) if f_mean is not None and s_mean is not None else None
            rel_shift = None
            if f_mean is not None and s_mean is not None and (f_mean != 0):
                rel_shift = (s_mean - f_mean) / abs(f_mean)
            rows.append(
                {
                    "symbol_pair": identity.get("symbol_pair"),
                    "bucket_size": identity.get("bucket_size"),
                    "feature_column": col,
                    "first_half_count": first[col].drop_nulls().len(),
                    "second_half_count": second[col].drop_nulls().len(),
                    "first_half_mean": f_mean,
                    "second_half_mean": s_mean,
                    "first_half_std": f_std,
                    "second_half_std": s_std,
                    "abs_mean_shift": abs_shift,
                    "relative_mean_shift": rel_shift,
                }
            )
    schema: dict[str, pl.DataType] = {
        "symbol_pair": pl.String,
        "bucket_size": pl.String,
        "feature_column": pl.String,
        "first_half_count": pl.Int64,
        "second_half_count": pl.Int64,
        "first_half_mean": pl.Float64,
        "second_half_mean": pl.Float64,
        "first_half_std": pl.Float64,
        "second_half_std": pl.Float64,
        "abs_mean_shift": pl.Float64,
        "relative_mean_shift": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)
