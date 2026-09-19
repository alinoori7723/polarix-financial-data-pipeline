from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import polars as pl

from tests.test_feature_eda import _build_gold

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "feature_eda_report.py"


def _load_main(module_name: str) -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_dry")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(features_root),
            "--reports-root",
            str(reports),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not list(reports.glob("feature_*_2026-05-18*"))


def test_cli_writes_outputs_with_force(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_write")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(features_root),
            "--reports-root",
            str(reports),
            "--force",
        ]
    )
    assert rc == 0
    assert (reports / "feature_eda_2026-05-18.json").exists()
    assert (reports / "feature_eda_2026-05-18.txt").exists()
    assert (reports / "feature_missingness_summary_2026-05-18.parquet").exists()
    assert (reports / "feature_distribution_summary_2026-05-18.parquet").exists()
    assert (reports / "feature_correlation_summary_2026-05-18.parquet").exists()


def test_cli_missing_manifest_returns_1(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_missing_manifest")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(tmp_path / "absent"),
            "--reports-root",
            str(reports),
            "--force",
        ]
    )
    assert rc == 1


def test_cli_existing_outputs_require_force(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_force_required")
    args = [
        "--date",
        "2026-05-18",
        "--features-root",
        str(features_root),
        "--reports-root",
        str(reports),
    ]
    rc1 = main(args + ["--force"])
    assert rc1 == 0
    rc2 = main(args)
    assert rc2 == 3


def test_cli_include_diagnostic_5s_keeps_5s_in_diagnostic_block(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path, include_5s=True)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_diag_5s")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(features_root),
            "--reports-root",
            str(reports),
            "--include-diagnostic-5s",
            "--force",
        ]
    )
    assert rc == 0
    parsed = json.loads((reports / "feature_eda_2026-05-18.json").read_text(encoding="utf-8"))
    assert parsed["diagnostic_block"]["rows"] > 0


def test_cli_bucket_sizes_parsed(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_bucket_parse")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(features_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "15s,60s",
            "--force",
        ]
    )
    assert rc == 0
    parsed = json.loads((reports / "feature_eda_2026-05-18.json").read_text(encoding="utf-8"))
    assert parsed["bucket_sizes"] == ["15s", "60s"]


def test_cli_bad_symbol_pairs_returns_2(tmp_path: Path) -> None:
    main = _load_main("feature_eda_cli_bad_sym")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(tmp_path / "x"),
            "--reports-root",
            str(tmp_path / "rep"),
            "--symbol-pairs",
            "",
        ]
    )
    assert rc == 2


def test_cli_bad_correlation_method_returns_2(tmp_path: Path) -> None:
    main = _load_main("feature_eda_cli_bad_method")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(tmp_path / "x"),
            "--reports-root",
            str(tmp_path / "rep"),
            "--correlation-method",
            "spearman",
        ]
    )
    assert rc == 2


def test_cli_output_parquets_readable_by_polars_and_duckdb(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    reports = tmp_path / "reports"
    main = _load_main("feature_eda_cli_readable")
    main(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(features_root),
            "--reports-root",
            str(reports),
            "--force",
        ]
    )
    for name in (
        "feature_missingness_summary_2026-05-18.parquet",
        "feature_distribution_summary_2026-05-18.parquet",
        "feature_correlation_summary_2026-05-18.parquet",
    ):
        path = reports / name
        df = pl.read_parquet(path)
        assert df.height > 0
        import duckdb

        con = duckdb.connect(":memory:")
        glob_str = str(path).replace("\\", "/")
        cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_str}')").fetchone()[0]
        con.close()
        assert cnt == df.height
