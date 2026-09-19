from __future__ import annotations

import polars as pl
import pytest

from polarix.features.feature_correlation import (
    _pearson_two_columns,
    compute_correlation_summary,
    pearson_t_pvalue,
    scipy_available,
)


def test_pearson_perfect_positive() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    r, n = _pearson_two_columns(df, "a", "b")
    assert n == 5
    assert r == pytest.approx(1.0)


def test_pearson_perfect_negative() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [50.0, 40.0, 30.0, 20.0, 10.0]})
    r, n = _pearson_two_columns(df, "a", "b")
    assert r == pytest.approx(-1.0)


def test_pearson_zero_for_orthogonal() -> None:
    df = pl.DataFrame({"a": [1.0, -1.0, 1.0, -1.0], "b": [1.0, 1.0, -1.0, -1.0]})
    r, n = _pearson_two_columns(df, "a", "b")
    assert n == 4
    assert r == pytest.approx(0.0, abs=1e-10)


def test_pearson_handles_pairwise_nulls() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0], "b": [10.0, None, 30.0, 40.0, 50.0]})
    r, n = _pearson_two_columns(df, "a", "b")
    assert n == 3
    assert r == pytest.approx(1.0)


def test_pearson_returns_none_for_too_few_rows() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    r, n = _pearson_two_columns(df, "a", "b")
    assert r is None
    assert n == 2


def test_pvalue_returns_none_when_n_below_three() -> None:
    p, available = pearson_t_pvalue(0.9, 2)
    assert p is None
    assert available is True


def test_pvalue_returns_zero_at_perfect_correlation() -> None:
    p, available = pearson_t_pvalue(1.0, 100)
    assert p == 0.0
    assert available is True


def test_pvalue_available_flag_matches_scipy() -> None:
    p, available = pearson_t_pvalue(0.5, 100)
    if scipy_available():
        assert available is True
        assert p is not None
        assert 0.0 < p < 1.0
    else:
        assert available is False
        assert p is None


def _make_grouped_df() -> pl.DataFrame:
    rng = list(range(100))
    return pl.DataFrame(
        {
            "symbol_pair": ["ES_SPX500"] * 50 + ["NQ_NDX100"] * 50,
            "bucket_size": ["15s"] * 100,
            "a": [float(i) for i in rng],
            "b": [float(i * 2) for i in rng],
            "c": [float(i * 7 % 11) for i in rng],
        }
    )


def test_compute_correlation_summary_pearson_basic() -> None:
    df = _make_grouped_df()
    out = compute_correlation_summary(df, feature_columns=["a", "b", "c"], small_sample_min_rows=10)
    assert out.height == 6
    ab = out.filter((pl.col("feature_a") == "a") & (pl.col("feature_b") == "b"))
    assert ab.height == 2
    for r in ab["correlation"].to_list():
        assert r == pytest.approx(1.0)
    assert all((int(v) == 50 for v in ab["n_obs"].to_list()))


def test_compute_correlation_summary_marks_small_sample_warning() -> None:
    df = _make_grouped_df()
    out = compute_correlation_summary(df, feature_columns=["a", "b"], small_sample_min_rows=10000)
    assert all(out["is_small_sample_warning"].to_list())


def test_high_correlation_with_tiny_n_still_flagged_small_sample() -> None:
    df = pl.DataFrame(
        {
            "symbol_pair": ["ES_SPX500"] * 4,
            "bucket_size": ["15s"] * 4,
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = compute_correlation_summary(df, feature_columns=["a", "b"], small_sample_min_rows=5000)
    row = out.row(0, named=True)
    assert row["correlation"] == pytest.approx(1.0)
    assert row["is_small_sample_warning"] is True


def test_correlation_summary_is_polars_dataframe() -> None:
    df = _make_grouped_df()
    out = compute_correlation_summary(df, feature_columns=["a", "b", "c"])
    assert isinstance(out, pl.DataFrame)


def test_no_pandas_corr_in_phase_2e_production_files() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "src" / "polarix" / "features" / "feature_statistics.py",
        repo_root / "src" / "polarix" / "features" / "feature_correlation.py",
        repo_root / "src" / "polarix" / "features" / "feature_eda.py",
        repo_root / "scripts" / "feature_eda_report.py",
    ]
    import ast

    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name.split(".")[0] for n in node.names]
                else:
                    names = [(node.module or "").split(".")[0]]
                assert "pandas" not in names, f"{p}: pandas import forbidden"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "corr":
                    base = node.func.value
                    is_pl_corr = isinstance(base, ast.Name) and base.id == "pl"
                    assert is_pl_corr, (
                        f"{p}:{node.lineno}: '.corr(' call forbidden (use pl.corr at module level)"
                    )
