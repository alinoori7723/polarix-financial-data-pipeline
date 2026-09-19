from __future__ import annotations

import math

import polars as pl
import pytest

from polarix.features.feature_statistics import (
    compute_distribution_summary,
    compute_intra_sample_stability,
    compute_missingness_summary,
)


def _df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_null_rate_correct() -> None:
    df = _df(
        [
            {"symbol_pair": "ES_SPX500", "bucket_size": "15s", "x": 1.0},
            {"symbol_pair": "ES_SPX500", "bucket_size": "15s", "x": None},
            {"symbol_pair": "ES_SPX500", "bucket_size": "15s", "x": 3.0},
            {"symbol_pair": "ES_SPX500", "bucket_size": "15s", "x": None},
        ]
    )
    out = compute_missingness_summary(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["null_count"] == 2
    assert row["n_non_null"] == 2
    assert row["null_rate"] == 0.5


def test_finite_rate_and_inf_nan_counts() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["ES_SPX500"] * 5,
            "bucket_size": ["15s"] * 5,
            "x": [1.0, math.inf, -math.inf, math.nan, 2.0],
        }
    )
    out = compute_missingness_summary(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["inf_count"] == 2
    assert row["nan_count"] == 1
    assert row["finite_count"] == 2
    assert row["finite_rate"] == 2 / 5


def test_detects_all_null_feature() -> None:
    df = pl.DataFrame(
        {"symbol_pair": ["ES_SPX500"] * 3, "bucket_size": ["15s"] * 3, "x": [None, None, None]},
        schema={"symbol_pair": pl.String, "bucket_size": pl.String, "x": pl.Float64},
    )
    out = compute_missingness_summary(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["all_null_flag"] is True
    assert row["null_rate"] == 1.0


def test_detects_near_constant_feature() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["ES_SPX500"] * 5,
            "bucket_size": ["15s"] * 5,
            "x": [7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )
    out = compute_missingness_summary(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["near_constant_flag"] is True
    assert row["distinct_count"] == 1


def test_groups_by_symbol_pair_and_bucket() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["A", "A", "B", "B"],
            "bucket_size": ["15s", "15s", "60s", "60s"],
            "x": [1.0, None, None, None],
        }
    )
    out = compute_missingness_summary(df, feature_columns=["x"]).sort(
        ["symbol_pair", "bucket_size"]
    )
    assert out.height == 2
    rows = {r["symbol_pair"]: r for r in out.to_dicts()}
    assert rows["A"]["null_rate"] == 0.5
    assert rows["B"]["all_null_flag"] is True


def test_quantiles_computed_correctly() -> None:
    df = pl.DataFrame(
        {"symbol_pair": ["ES"] * 5, "bucket_size": ["15s"] * 5, "x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    out = compute_distribution_summary(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["p25"] == pytest.approx(2.0)
    assert row["p50"] == pytest.approx(3.0)
    assert row["p75"] == pytest.approx(4.0)
    assert row["iqr"] == pytest.approx(2.0)


def test_heavy_tail_flag_without_mutation() -> None:
    bulk = [float(i) / 100 for i in range(95)]
    df = pl.DataFrame(
        {"symbol_pair": ["ES"] * 100, "bucket_size": ["15s"] * 100, "x": bulk + [1000.0] * 5}
    )
    snapshot = df["x"].to_list()
    out = compute_distribution_summary(df, feature_columns=["x"], heavy_tail_threshold=2.0)
    assert df["x"].to_list() == snapshot
    row = out.row(0, named=True)
    assert row["heavy_tail_score"] is not None and row["heavy_tail_score"] >= 2.0
    assert row["heavy_tail_flag"] is True
    assert row["outlier_rate_iqr"] is not None


def test_zero_iqr_does_not_claim_heavy_tail() -> None:
    df = pl.DataFrame(
        {"symbol_pair": ["ES"] * 100, "bucket_size": ["15s"] * 100, "x": [0.0] * 95 + [1000.0] * 5}
    )
    out = compute_distribution_summary(df, feature_columns=["x"], heavy_tail_threshold=2.0)
    row = out.row(0, named=True)
    assert row["heavy_tail_score"] is None
    assert row["heavy_tail_flag"] is False


def test_does_not_clip_or_winsorize() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["ES"] * 6,
            "bucket_size": ["15s"] * 6,
            "x": [-1000000000.0, -1.0, 0.0, 1.0, 2.0, 1000000000.0],
        }
    )
    before = df.clone()
    compute_distribution_summary(df, feature_columns=["x"])
    assert df.to_dicts() == before.to_dicts()


def test_sample_size_warning_set_when_below_threshold() -> None:
    df = pl.DataFrame(
        {"symbol_pair": ["ES"] * 10, "bucket_size": ["15s"] * 10, "x": list(range(10))}
    )
    out = compute_distribution_summary(df, feature_columns=["x"], small_sample_min_rows=100)
    assert out["sample_size_warning"][0] is True


def test_intra_sample_stability_splits_first_and_second_half() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["ES"] * 10,
            "bucket_size": ["15s"] * 10,
            "bucket_start_utc_ns": list(range(10)),
            "x": [0.0] * 5 + [10.0] * 5,
        }
    )
    out = compute_intra_sample_stability(df, feature_columns=["x"])
    row = out.row(0, named=True)
    assert row["first_half_count"] == 5
    assert row["second_half_count"] == 5
    assert row["first_half_mean"] == pytest.approx(0.0)
    assert row["second_half_mean"] == pytest.approx(10.0)
    assert row["abs_mean_shift"] == pytest.approx(10.0)
