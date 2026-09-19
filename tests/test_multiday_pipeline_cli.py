from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_CLI = REPO_ROOT / "scripts" / "build_multiday_plan.py"
PIPELINE_CLI = REPO_ROOT / "scripts" / "run_multiday_pipeline.py"


def _load_main(path: Path, name: str) -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def _seed_mt5(tmp_path: Path, dates: list[str]) -> Path:
    import datetime as _dt

    data = tmp_path / "data"
    mt5 = data / "normalized" / "mt5_ticks"
    for d in dates:
        t0 = int(_dt.datetime.fromisoformat(f"{d}T05:00:00+00:00").timestamp() * 1000)
        for sym in ("SPX500", "NDX100"):
            part = mt5 / f"symbol={sym}" / f"date={d}" / "part-0001.parquet"
            part.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "symbol": [sym, sym],
                    "time_msc_utc_ms": pa.array([t0, t0 + 60000], type=pa.int64()),
                }
            )
            pq.write_table(table, part, compression="zstd")
    return data


def test_build_multiday_plan_writes_json_and_txt(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18", "2026-05-19"])
    main = _load_main(PLAN_CLI, "build_multiday_plan_cli")
    rc = main(["--data-root", str(data), "--reports-root", str(tmp_path / "reports")])
    assert rc == 0
    plans = list((tmp_path / "reports").glob("multiday_plan_*.json"))
    assert plans
    parsed = json.loads(plans[0].read_text(encoding="utf-8"))
    assert parsed["summary"]["total_dates"] == 2


def test_run_multiday_pipeline_dry_run_no_download(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_dry")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0


def test_run_multiday_pipeline_fail_closed_without_allow_databento(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_block")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--download-cme",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 2


def test_run_multiday_pipeline_both_flags_dry_run_does_not_download(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_both")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--download-cme",
            "--allow-databento-download",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0


def test_run_multiday_pipeline_real_run_requires_acknowledge_cost_risk(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_no_ack")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--download-cme",
            "--allow-databento-download",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 2


def test_run_multiday_pipeline_dry_run_with_all_flags_succeeds(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_all_flags_dry")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--download-cme",
            "--allow-databento-download",
            "--acknowledge-cost-risk",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0
    reports = tmp_path / "reports"
    dry_run_reports = list(reports.glob("multiday_pipeline_dry_run_*.json"))
    assert dry_run_reports, "dry-run must persist the detailed JSON report"


def test_run_multiday_pipeline_bad_run_selection_policy_returns_2(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_bad_policy")
    with pytest.raises(SystemExit) as ei:
        main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(data),
                "--reports-root",
                str(tmp_path / "reports"),
                "--dry-run",
                "--run-selection-policy",
                "definitely-not-a-policy",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    assert ei.value.code == 2


def test_run_multiday_pipeline_bad_run_id_map_returns_2(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_bad_run_id_map")
    rc = main(
        [
            "--dates",
            "2026-05-18",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--run-id-map",
            "no-equals-sign",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 2


def test_run_multiday_pipeline_dates_parses_comma_list(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18", "2026-05-19"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_dates")
    rc = main(
        [
            "--dates",
            "2026-05-18,2026-05-19",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0


def test_run_multiday_pipeline_start_end_expands(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, ["2026-05-18", "2026-05-19", "2026-05-20"])
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_range")
    rc = main(
        [
            "--start-date",
            "2026-05-18",
            "--end-date",
            "2026-05-20",
            "--data-root",
            str(data),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0


def test_run_multiday_pipeline_missing_mt5_skipped_when_flag(tmp_path: Path) -> None:
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_missing_mt5_skip")
    rc = main(
        [
            "--dates",
            "2026-01-01",
            "--data-root",
            str(tmp_path / "data_absent"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--dry-run",
            "--continue-on-missing-mt5",
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 0


def test_run_multiday_pipeline_bad_args_returns_2(tmp_path: Path) -> None:
    main = _load_main(PIPELINE_CLI, "run_multiday_pipeline_cli_bad_args")
    rc = main(
        [
            "--dates",
            "",
            "--data-root",
            str(tmp_path / "data"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--min-free-disk-gb",
            "0",
            "--min-available-memory-gb",
            "0",
        ]
    )
    assert rc == 2


def test_powershell_wrapper_contains_no_trading_logic() -> None:
    text = (REPO_ROOT / "scripts" / "run_multiday_pipeline.ps1").read_text(encoding="utf-8")
    forbidden = (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for token in forbidden:
        assert token not in text, f"PowerShell wrapper contains forbidden {token!r}"


def test_powershell_wrapper_uses_python_unbuffered() -> None:
    text = (REPO_ROOT / "scripts" / "run_multiday_pipeline.ps1").read_text(encoding="utf-8")
    assert "python -u" in text or "-u " in text, "PowerShell wrapper must use python -u"


def test_powershell_wrapper_calls_run_multiday_pipeline_py() -> None:
    text = (REPO_ROOT / "scripts" / "run_multiday_pipeline.ps1").read_text(encoding="utf-8")
    assert "run_multiday_pipeline.py" in text
