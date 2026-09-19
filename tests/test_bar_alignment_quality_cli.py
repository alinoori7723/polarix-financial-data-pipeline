from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

from tests.test_bar_alignment_quality import _matched_overlap, _seed

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "bar_alignment_quality_report.py"


def _load_cli_main() -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location("bar_alignment_quality_cli", CLI_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def test_cli_writes_json_and_txt(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_nq, mt5_ndx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "5s",
        ]
    )
    assert rc == 0
    assert (reports / "bar_alignment_quality_2026-05-18.json").exists()
    assert (reports / "bar_alignment_quality_2026-05-18.txt").exists()


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "5s",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not (reports / "bar_alignment_quality_2026-05-18.json").exists()
    assert not (reports / "bar_alignment_quality_2026-05-18.txt").exists()


def test_cli_bucket_sizes_parsed(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "1s,5s,15s,60s",
        ]
    )
    assert rc in (0, 1)
    parsed = json.loads(
        (reports / "bar_alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["bucket_sizes"] == ["1s", "5s", "15s", "60s"]


def test_cli_symbol_map_parsed(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"FOO": mt5_spx, "NDX100": []}
    )
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "5s",
            "--symbol-map",
            "ES=FOO",
        ]
    )
    assert rc in (0, 1)
    parsed = json.loads(
        (reports / "bar_alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["symbol_map"] == {"ES": "FOO"}


def test_cli_missing_data_does_not_crash(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "absent_cme"),
            "--mt5-root",
            str(tmp_path / "absent_mt5"),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "5s",
        ]
    )
    assert rc == 1
    parsed = json.loads(
        (reports / "bar_alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["quality_decision"] == "FAIL"
    assert parsed["decision_reason"] in ("MISSING_CME_REFERENCE", "MISSING_MT5_SILVER")


def test_cli_rejects_bad_symbol_map(tmp_path: Path) -> None:
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "x"),
            "--mt5-root",
            str(tmp_path / "y"),
            "--reports-root",
            str(tmp_path / "rep"),
            "--symbol-map",
            "ESSPX500",
        ]
    )
    assert rc == 2


def test_cli_rejects_bad_bucket_sizes(tmp_path: Path) -> None:
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "x"),
            "--mt5-root",
            str(tmp_path / "y"),
            "--reports-root",
            str(tmp_path / "rep"),
            "--bucket-sizes",
            "abc",
        ]
    )
    assert rc == 2


def test_cli_write_buckets_emits_parquet(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched_overlap(base, n=4, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    reports = tmp_path / "reports"
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(cme_root),
            "--mt5-root",
            str(mt5_root),
            "--reports-root",
            str(reports),
            "--bucket-sizes",
            "5s",
            "--write-buckets",
        ]
    )
    assert rc in (0, 1)
    assert (reports / "bar_alignment_buckets_2026-05-18.parquet").exists()
