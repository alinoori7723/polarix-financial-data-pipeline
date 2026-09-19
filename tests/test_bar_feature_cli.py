from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import polars as pl

from tests.test_bar_feature_builder import _matched, _seed

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_CLI = REPO_ROOT / "scripts" / "build_bar_features.py"
QUALITY_CLI = REPO_ROOT / "scripts" / "bar_feature_quality_report.py"


def _load_main(path: Path, module_name: str) -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def _seed_real(tmp_path: Path) -> tuple[Path, Path]:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=6, bucket_seconds=15)
    cme_nq, mt5_ndx = _matched(base, n=6, bucket_seconds=15)
    return _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )


def test_build_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not list((tmp_path / "features").rglob("part-*.parquet"))
    assert not (tmp_path / "features" / "date=2026-05-18" / "bar_features_manifest.json").exists()


def test_build_cli_writes_parquet_and_manifest(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_2")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s,60s",
            "--force",
        ]
    )
    assert rc == 0
    assert list((tmp_path / "features").rglob("part-*.parquet"))
    assert (tmp_path / "features" / "date=2026-05-18" / "bar_features_manifest.json").exists()


def test_quality_cli_writes_json_and_txt(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_3")
    main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s,60s",
            "--force",
        ]
    )
    qmain = _load_main(QUALITY_CLI, "bar_feature_quality_cli")
    rc = qmain(
        [
            "--date",
            "2026-05-18",
            "--features-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "reports" / "bar_feature_quality_2026-05-18.json").exists()
    assert (tmp_path / "reports" / "bar_feature_quality_2026-05-18.txt").exists()


def test_include_diagnostic_5s_adds_5s_rows(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_4")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s",
            "--include-diagnostic-5s",
            "--force",
        ]
    )
    assert rc == 0
    out_5s = tmp_path / "features" / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=5s"
    assert out_5s.exists()
    parts = list(out_5s.glob("part-*.parquet"))
    assert parts
    df = pl.read_parquet(parts[0])
    assert "DIAGNOSTIC_ONLY" in df["feature_quality_flag"].to_list()


def test_bucket_sizes_parsed(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_5")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s,60s",
            "--force",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "features" / "date=2026-05-18" / "bar_features_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["bucket_sizes"] == ["15s", "60s"]


def test_force_overwrites(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_6")
    args = [
        "--date",
        "2026-05-18",
        "--cme-root",
        str(cme_root),
        "--mt5-root",
        str(mt5_root),
        "--output-root",
        str(tmp_path / "features"),
        "--reports-root",
        str(tmp_path / "reports"),
        "--bucket-sizes",
        "15s",
        "--force",
    ]
    rc1 = main(args)
    assert rc1 == 0
    rc2 = main(args)
    assert rc2 == 0


def test_missing_input_returns_2(tmp_path: Path) -> None:
    main = _load_main(BUILD_CLI, "build_bar_features_cli_missing")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "absent_cme"),
            "--mt5-root",
            str(tmp_path / "absent_mt5"),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "15s",
        ]
    )
    assert rc == 2


def test_bad_symbol_map_returns_2(tmp_path: Path) -> None:
    main = _load_main(BUILD_CLI, "build_bar_features_cli_bad_map")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "x"),
            "--mt5-root",
            str(tmp_path / "y"),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--symbol-map",
            "ESSPX500",
        ]
    )
    assert rc == 2


def test_bad_bucket_sizes_returns_2(tmp_path: Path) -> None:
    main = _load_main(BUILD_CLI, "build_bar_features_cli_bad_bucket")
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "x"),
            "--mt5-root",
            str(tmp_path / "y"),
            "--output-root",
            str(tmp_path / "features"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--bucket-sizes",
            "abc",
        ]
    )
    assert rc == 2


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_real(tmp_path)
    main = _load_main(BUILD_CLI, "build_bar_features_cli_no_force")
    args = [
        "--date",
        "2026-05-18",
        "--cme-root",
        str(cme_root),
        "--mt5-root",
        str(mt5_root),
        "--output-root",
        str(tmp_path / "features"),
        "--reports-root",
        str(tmp_path / "reports"),
        "--bucket-sizes",
        "15s",
    ]
    rc1 = main(args + ["--force"])
    assert rc1 == 0
    rc2 = main(args)
    assert rc2 == 3
