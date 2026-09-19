from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence

import polars as pl

DEFAULT_PVALUE_ALPHA = 0.05
DEFAULT_SMALL_SAMPLE_MIN_ROWS = 5000
try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except Exception:
    _scipy_stats = None
    _SCIPY_AVAILABLE = False
CORRELATION_COLUMNS = (
    "symbol_pair",
    "bucket_size",
    "feature_a",
    "feature_b",
    "n_obs",
    "correlation",
    "abs_correlation",
    "p_value",
    "pvalue_available",
    "pvalue_alpha",
    "is_statistically_significant",
    "is_small_sample_warning",
    "pair_bucket_scope",
    "pair_symbol_scope",
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


def pearson_t_pvalue(r: float, n: int) -> tuple[Optional[float], bool]:
    if n is None or n < 3:
        return (None, True)
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return (None, True)
    eps = 1e-12
    if abs(r) >= 1.0 - eps:
        return (0.0, True)
    if not _SCIPY_AVAILABLE:
        return (None, False)
    denom = 1.0 - r * r
    if denom <= 0:
        return (0.0, True)
    t = r * math.sqrt((n - 2) / denom)
    p = 2 * float(_scipy_stats.t.sf(abs(t), df=n - 2))
    return (p, True)


def _pearson_two_columns(df: pl.DataFrame, a: str, b: str) -> tuple[Optional[float], int]:
    if a not in df.columns or b not in df.columns:
        return (None, 0)
    if not _is_numeric_dtype(df[a].dtype) or not _is_numeric_dtype(df[b].dtype):
        return (None, 0)
    pair = df.select([a, b]).drop_nulls()
    pair = pair.filter(pl.col(a).is_finite() & pl.col(b).is_finite())
    n = pair.height
    if n < 3:
        return (None, n)
    r = pair.select(pl.corr(a, b)).item()
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return (None, n)
    return (float(r), n)


def compute_correlation_summary(
    df: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_keys: Sequence[str] = ("symbol_pair", "bucket_size"),
    pvalue_alpha: float = DEFAULT_PVALUE_ALPHA,
    small_sample_min_rows: int = DEFAULT_SMALL_SAMPLE_MIN_ROWS,
    max_features: Optional[int] = None,
) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(
            schema={
                c: pl.String
                if c
                in (
                    "symbol_pair",
                    "bucket_size",
                    "feature_a",
                    "feature_b",
                    "pair_bucket_scope",
                    "pair_symbol_scope",
                )
                else pl.Float64
                if c in ("correlation", "abs_correlation", "p_value", "pvalue_alpha")
                else pl.Int64
                if c == "n_obs"
                else pl.Boolean
                for c in CORRELATION_COLUMNS
            }
        )
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
        cand_cols = [
            c for c in feature_columns if c in sub.columns and _is_numeric_dtype(sub[c].dtype)
        ]
        if max_features is not None:
            cand_cols = cand_cols[:max_features]
        for a, b in itertools.combinations(cand_cols, 2):
            r, n_obs = _pearson_two_columns(sub, a, b)
            p_value, pvalue_available = (
                pearson_t_pvalue(r, n_obs) if r is not None else (None, _SCIPY_AVAILABLE)
            )
            if r is None:
                abs_r = None
                is_sig = False
            else:
                abs_r = abs(r)
                is_sig = p_value is not None and p_value < pvalue_alpha
            rows.append(
                {
                    "symbol_pair": identity.get("symbol_pair"),
                    "bucket_size": identity.get("bucket_size"),
                    "feature_a": a,
                    "feature_b": b,
                    "n_obs": n_obs,
                    "correlation": r,
                    "abs_correlation": abs_r,
                    "p_value": p_value,
                    "pvalue_available": pvalue_available,
                    "pvalue_alpha": pvalue_alpha,
                    "is_statistically_significant": is_sig,
                    "is_small_sample_warning": n_obs < small_sample_min_rows,
                    "pair_bucket_scope": identity.get("bucket_size"),
                    "pair_symbol_scope": identity.get("symbol_pair"),
                }
            )
    schema = {
        "symbol_pair": pl.String,
        "bucket_size": pl.String,
        "feature_a": pl.String,
        "feature_b": pl.String,
        "n_obs": pl.Int64,
        "correlation": pl.Float64,
        "abs_correlation": pl.Float64,
        "p_value": pl.Float64,
        "pvalue_available": pl.Boolean,
        "pvalue_alpha": pl.Float64,
        "is_statistically_significant": pl.Boolean,
        "is_small_sample_warning": pl.Boolean,
        "pair_bucket_scope": pl.String,
        "pair_symbol_scope": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def scipy_available() -> bool:
    return _SCIPY_AVAILABLE
