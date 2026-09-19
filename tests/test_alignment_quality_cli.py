from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

from tests.test_alignment_quality import _seed_datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "alignment_quality_report.py"


def _load_cli_main() -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location("alignment_quality_report_cli", CLI_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def test_cli_writes_json_and_txt(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 1000 * 1000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": 999, "safe": True}], "NDX100": []},
    )
    reports_root = tmp_path / "reports"
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
            str(reports_root),
        ]
    )
    assert rc == 0
    assert (reports_root / "alignment_quality_2026-05-18.json").exists()
    assert (reports_root / "alignment_quality_2026-05-18.txt").exists()


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 1000 * 1000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": 999, "safe": True}], "NDX100": []},
    )
    reports_root = tmp_path / "reports"
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
            str(reports_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not (reports_root / "alignment_quality_2026-05-18.json").exists()
    assert not (reports_root / "alignment_quality_2026-05-18.txt").exists()


def test_cli_symbol_map_parsed(tmp_path: Path) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 1000 * 1000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"FOO": [{"ts_ms": 999, "safe": True}], "NDX100": []},
    )
    reports_root = tmp_path / "reports"
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
            str(reports_root),
            "--symbol-map",
            "ES=FOO",
        ]
    )
    assert rc == 0
    parsed = json.loads(
        (reports_root / "alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["symbol_map"] == {"ES": "FOO"}
    assert "ES" in parsed["per_symbol"]


def test_cli_alignment_tolerance_overrides_default(tmp_path: Path, capsys) -> None:
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": 1200 * 1000000, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": 1000, "safe": True}], "NDX100": []},
    )
    reports_root = tmp_path / "reports"
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
            str(reports_root),
            "--alignment-tolerance-ms",
            "250",
        ]
    )
    assert rc == 0
    parsed = json.loads(
        (reports_root / "alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["per_symbol"]["ES"]["alignment_match_rate"] == 1.0
    assert parsed["alignment_tolerance_ms"] == 250


def test_cli_missing_overlap_does_not_crash(tmp_path: Path) -> None:
    cme_event_ns = int(1779076500000000000)
    mt5_ts_ms = int(1779081406483)
    cme_root, mt5_root = _seed_datasets(
        tmp_path,
        "2026-05-18",
        {"ES": [{"event_ns": cme_event_ns, "price": 5000.0, "size": 1}], "NQ": []},
        {"SPX500": [{"ts_ms": mt5_ts_ms, "safe": True}], "NDX100": []},
    )
    reports_root = tmp_path / "reports"
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
            str(reports_root),
        ]
    )
    assert rc == 1
    parsed = json.loads(
        (reports_root / "alignment_quality_2026-05-18.json").read_text(encoding="utf-8")
    )
    assert parsed["quality_decision"] == "FAIL"
    assert parsed["decision_reason"] == "MISSING_OVERLAP_DATA"
    assert parsed["real_overlap_present"] is False


def test_cli_rejects_bad_symbol_map(tmp_path: Path) -> None:
    main = _load_cli_main()
    rc = main(
        [
            "--date",
            "2026-05-18",
            "--cme-root",
            str(tmp_path / "cme"),
            "--mt5-root",
            str(tmp_path / "mt5"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--symbol-map",
            "ESSPX500",
        ]
    )
    assert rc == 2
