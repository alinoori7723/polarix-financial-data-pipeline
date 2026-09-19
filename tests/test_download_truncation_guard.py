from __future__ import annotations

import ast
import datetime as _dt
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.alignment.alignment_quality import (
    AlignmentQualityConfig,
    AlignmentQualityThresholds,
    build_alignment_quality_report,
)
from polarix.ingestion import cme_downloader as cme_dl
from polarix.ingestion.cme_reference_ingest import IngestConfig, ingest
from polarix.orchestration import pipeline_status as ps
from polarix.orchestration.databento_download_plan import plan_day_download
from polarix.orchestration.multiday_dry_run_report import DryRunConfig, build_per_day_plan
from polarix.orchestration.pipeline_orchestrator import OrchestratorConfig, process_one_date
from polarix.quality.cme_reference_quality import CmeQualityConfig, build_quality_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_download_script():
    spec = importlib.util.spec_from_file_location(
        "download_databento_sample_under_test",
        REPO_ROOT / "scripts" / "download_databento_sample.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_metadata_sidecar(
    parquet_path: Path,
    *,
    completeness: str,
    records_downloaded: Optional[int] = 1000000,
    max_download_records: Optional[int] = 1000000,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        pq.write_table(pa.table({"x": [1]}), parquet_path)
    payload = {
        "requested_start_utc": "2026-05-18T05:00:00Z",
        "requested_end_utc": "2026-05-18T06:00:00Z",
        "dataset": "GLBX.MDP3",
        "schema": "mbp-1",
        "symbols": ["ES.c.0", "NQ.c.0"],
        "stype_in": "continuous",
        "output_path": str(parquet_path),
        "file_size_bytes": parquet_path.stat().st_size,
        "max_download_records": max_download_records,
        "records_downloaded": records_downloaded,
        "data_completeness_status": completeness,
        "truncation_warning": completeness == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT,
        "created_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "databento_sdk_version": "fake-test",
        "api_call_parameters": {"limit": max_download_records},
        "physical_limit_applied": max_download_records is not None,
        "physical_limit_type": "record_limit" if max_download_records is not None else None,
        "physical_limit_value": max_download_records,
    }
    sidecar = cme_dl.metadata_path_for(parquet_path)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def _seed_mt5(data_root: Path, date: str) -> None:
    t0 = int(_dt.datetime.fromisoformat(f"{date}T05:00:00+00:00").timestamp() * 1000)
    for sym in ("SPX500", "NDX100"):
        path = (
            data_root
            / "normalized"
            / "mt5_ticks"
            / f"symbol={sym}"
            / f"date={date}"
            / "part-0001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "symbol": [sym] * 2,
                    "time_msc_utc_ms": pa.array([t0, t0 + 10000], type=pa.int64()),
                }
            ),
            path,
            compression="zstd",
        )


def _make_orchestrator_config(tmp_path: Path, dates: list[str], **overrides) -> OrchestratorConfig:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    cfg = OrchestratorConfig(
        dates=dates,
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        python_exe=Path("python"),
        dry_run=False,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_11_cli_resolves_date_partition_output_path(tmp_path: Path) -> None:
    script = _import_download_script()
    argv = [
        "--start",
        "2026-05-18T05:16:46Z",
        "--end",
        "2026-05-18T06:16:46Z",
        "--symbols",
        "ES.c.0,NQ.c.0",
        "--stype-in",
        "continuous",
        "--output-root",
        str(tmp_path / "raw" / "cme_sample"),
        "--max-download-records",
        "1000",
        "--date-partition-output",
    ]
    args = script.parse_args(argv)
    resolved = script._resolve_output_path(args, ["ES.c.0", "NQ.c.0"])
    assert "date=2026-05-18" in str(resolved)
    argv2 = argv + ["--date", "2026-05-19"]
    args2 = script.parse_args(argv2)
    resolved2 = script._resolve_output_path(args2, ["ES.c.0", "NQ.c.0"])
    assert "date=2026-05-19" in str(resolved2)


def test_12_cli_passes_max_download_records_through(tmp_path: Path, capsys) -> None:
    script = _import_download_script()
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0,NQ.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
            "--max-download-records",
            "1000",
            "--date-partition-output",
            "--metadata-only",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["status"] == cme_dl.DOWNLOAD_METADATA_ONLY
    assert doc["request"]["max_download_records"] == 1000


def test_13_cli_fails_closed_without_max_download_records(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-test-key")
    script = _import_download_script()
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["status"] == cme_dl.DOWNLOAD_BLOCKED_NO_RECORD_LIMIT
    assert doc["records_downloaded"] is None


def test_14_cli_prints_metadata_path_when_present(tmp_path: Path, capsys) -> None:
    script = _import_download_script()
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
            "--max-download-records",
            "1000",
            "--metadata-only",
        ]
    )
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["planned_output_path"]
    assert "out.parquet".replace("out", "")
    assert doc["request"]["output_path"].endswith(".parquet")


def test_15_cli_supports_date_partition_output(tmp_path: Path, capsys) -> None:
    script = _import_download_script()
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0,NQ.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
            "--max-download-records",
            "1000",
            "--date-partition-output",
            "--metadata-only",
        ]
    )
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert "date=2026-05-18" in doc["planned_output_path"]


def test_16_cli_returns_nonzero_on_blocked_download(tmp_path: Path, monkeypatch, capsys) -> None:
    script = _import_download_script()
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
            "--max-download-records",
            "1000",
        ]
    )
    assert rc == 2
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == cme_dl.DOWNLOAD_BLOCKED_MISSING_API_KEY


def test_17_cli_metadata_only_does_not_construct_databento_client(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-test-key")
    script = _import_download_script()
    real_run = script.run_download
    seen: dict = {}

    def boom(*_a, **_kw):
        raise AssertionError("metadata-only must not construct the Databento client")

    def spy_run(request, *, client_factory=None):
        seen["request"] = request
        seen["client_factory"] = client_factory
        return real_run(request, client_factory=boom)

    monkeypatch.setattr(script, "run_download", spy_run)
    rc = script.main(
        [
            "--start",
            "2026-05-18T05:16:46Z",
            "--end",
            "2026-05-18T05:17:46Z",
            "--symbols",
            "ES.c.0",
            "--stype-in",
            "continuous",
            "--output-root",
            str(tmp_path / "raw"),
            "--max-download-records",
            "1000",
            "--metadata-only",
        ]
    )
    assert rc == 0
    assert seen["request"].metadata_only is True
    out = capsys.readouterr().out
    assert json.loads(out)["status"] == cme_dl.DOWNLOAD_METADATA_ONLY


def test_18_no_get_range_callers_outside_cme_downloader() -> None:
    offenders: list[str] = []
    for py in (REPO_ROOT / "src").rglob("*.py"):
        if py.name == "cme_downloader.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "get_range"
                and isinstance(fn.value, ast.Attribute)
                and (fn.value.attr == "timeseries")
            ):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"unexpected get_range callers outside cme_downloader: {offenders}"


def test_19_orchestrator_argv_carries_max_download_records(tmp_path: Path) -> None:
    plan = plan_day_download(
        date="2026-05-18",
        mt5_window_ms=(
            int(_dt.datetime(2026, 5, 18, 5, 0, tzinfo=_dt.timezone.utc).timestamp() * 1000),
            int(_dt.datetime(2026, 5, 18, 6, 0, tzinfo=_dt.timezone.utc).timestamp() * 1000),
        ),
        output_root=tmp_path,
        download_cme=True,
        allow_databento_download=True,
        api_key_present=True,
        max_download_records=1000000,
    )
    assert "--max-download-records" in plan.argv
    idx = plan.argv.index("--max-download-records")
    assert plan.argv[idx + 1] == "1000000"


def test_20_truncated_by_limit_blocks_downstream_by_default(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    raw = data / "raw" / "cme_sample" / "date=2026-05-18" / "esnq.parquet"
    _write_metadata_sidecar(raw, completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    cfg = _make_orchestrator_config(
        tmp_path, ["2026-05-18"], dry_run=False, allow_truncated_cme_sample=False
    )
    invoked: list[list[str]] = []

    def runner(argv, timeout, cwd):
        invoked.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_TRUNCATED_SAMPLE_BLOCKED
    assert ps.STATUS_CME_RAW_TRUNCATED_BY_LIMIT in day.stage_statuses
    assert invoked == [], "no downstream stage may run when sample is truncated"


def test_21_allow_truncated_cme_sample_marks_exploratory_warning(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    raw = data / "raw" / "cme_sample" / "date=2026-05-18" / "esnq.parquet"
    _write_metadata_sidecar(raw, completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    cfg = _make_orchestrator_config(
        tmp_path, ["2026-05-18"], dry_run=False, allow_truncated_cme_sample=True
    )

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert ps.STATUS_CME_RAW_TRUNCATED_BY_LIMIT in day.stage_statuses
    assert ps.STATUS_TRUNCATED_SAMPLE_ALLOWED_EXPLORATORY in day.stage_statuses
    assert any(("exploratory" in w for w in day.warnings))
    assert day.final_status == ps.STATUS_DAY_COMPLETE


def test_22_dry_run_never_calls_download_runner(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    cfg = _make_orchestrator_config(
        tmp_path, ["2026-05-18"], dry_run=True, download_cme=True, allow_databento_download=True
    )

    def download_runner(argv, timeout):
        raise AssertionError("dry-run must never call Databento")

    day = process_one_date(cfg, "2026-05-18", download_runner=download_runner)
    assert day.download_outcome["status"] == "DRY_RUN"


def test_23_physical_limit_unsupported_still_blocks_unless_override(tmp_path: Path) -> None:

    class _FakeTSNoLimit:
        def get_range(self, *, dataset, schema, symbols, stype_in, start, end):
            raise AssertionError("must not be called when limit is unsupported and override is off")

    class _FakeClient:
        def __init__(self):
            self.timeseries = _FakeTSNoLimit()

    req = cme_dl.DownloadRequest(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=("ES.c.0",),
        stype_in="continuous",
        start_utc="2026-05-18T05:16:46Z",
        end_utc="2026-05-18T06:16:46Z",
        output_path=tmp_path / "out.parquet",
        max_download_records=1000,
        allow_download_without_physical_limit=False,
        api_key_present=True,
    )
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient())
    assert result.status == cme_dl.DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED


def test_24_existing_cost_gate_behavior_intact(tmp_path: Path) -> None:
    from polarix.orchestration.databento_cost_guard import (
        NOT_REQUESTED_DRY_RUN,
        PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
        CostEstimate,
        GateInputs,
        PhysicalLimitCapability,
        decide_download_gate,
    )

    gate = decide_download_gate(
        GateInputs(
            download_cme=True,
            allow_databento_download=True,
            acknowledge_cost_risk=True,
            allow_unestimated_download=False,
            cost_estimate_required=True,
            require_physical_download_limit=True,
            allow_download_without_physical_limit=False,
            dry_run=True,
            api_key_present=True,
            max_estimated_cost_usd=10.0,
            max_estimated_size_gb=5.0,
            max_download_records=1000000,
            max_download_size_gb=5.0,
            max_download_cost_usd=10.0,
        ),
        cost_estimate=CostEstimate(
            status=NOT_REQUESTED_DRY_RUN,
            source="dry_run",
            message="dry-run",
            max_estimated_cost_usd=10.0,
            max_estimated_size_gb=5.0,
        ),
        physical_limit=PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN, message="dry-run"
        ),
    )
    assert gate.decision == "DOWNLOAD_BLOCKED_DRY_RUN"


def test_25_cme_reference_quality_warns_on_truncated_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "cme_sample"
    fixture = raw_dir / "date=2026-05-18" / "ok.parquet"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "ts_event": [1, 2, 3],
                "ts_recv": [10, 20, 30],
                "raw_symbol": ["ESM6", "ESM6", "NQM6"],
                "action": ["T", "T", "T"],
                "side": ["B", "A", "B"],
                "price": [5000.0, 5000.25, 18000.0],
                "size": [1, 1, 1],
                "bid_px_00": [4999.75, 5000.0, 17999.75],
                "ask_px_00": [5000.25, 5000.5, 18000.25],
                "bid_sz_00": [5, 5, 5],
                "ask_sz_00": [5, 5, 5],
            }
        ),
        fixture,
        compression="zstd",
    )
    _write_metadata_sidecar(fixture, completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    ingest(
        IngestConfig(
            input_root=raw_dir,
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
            date="2026-05-18",
            force=True,
        )
    )
    rep = build_quality_report(
        CmeQualityConfig(
            date="2026-05-18",
            input_root=raw_dir,
            output_root=tmp_path / "normalized" / "cme_reference",
            reports_root=tmp_path / "reports",
        )
    )
    assert rep["cme_raw_completeness_status"] == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    assert any(("TRUNCATED_BY_LIMIT" in w for w in rep["warnings"]))
    assert rep["decision"] == "PARTIAL"


def test_26_alignment_quality_warns_on_truncated_raw(tmp_path: Path) -> None:
    cme_root = tmp_path / "cme" / "reference_trades"
    mt5_root = tmp_path / "mt5"
    cme_dir = cme_root / "symbol=ES" / "date=2026-05-18"
    mt5_dir = mt5_root / "symbol=SPX500" / "date=2026-05-18"
    cme_dir.mkdir(parents=True, exist_ok=True)
    mt5_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "source": ["databento"],
                "vendor": ["Databento"],
                "dataset": ["GLBX.MDP3"],
                "schema": ["mbp-1"],
                "symbol": ["ES"],
                "event_time_utc_ns": [1000000000],
                "price": [5000.0],
                "size": [1],
                "aggressor_side": ["BUY"],
                "signed_size": [1.0],
                "is_reference_trade_valid": [True],
            }
        ),
        cme_dir / "part-0001.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "symbol": ["SPX500"],
                "time_msc_utc_ms": pa.array([1000], type=pa.int64()),
                "is_join_safe": [True],
                "mid": [100.0],
                "spread_price": [0.25],
                "spread_points": pa.array([1], type=pa.int64()),
            }
        ),
        mt5_dir / "part-0001.parquet",
        compression="zstd",
    )
    raw_dir = tmp_path / "raw" / "cme_sample" / "date=2026-05-18"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_pq = raw_dir / "trunc.parquet"
    _write_metadata_sidecar(raw_pq, completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    cfg = AlignmentQualityConfig(
        date="2026-05-18",
        cme_root=cme_root,
        mt5_root=mt5_root,
        reports_root=tmp_path / "reports",
        symbol_map={"ES": "SPX500"},
        cme_raw_input_root=raw_dir,
        thresholds=AlignmentQualityThresholds(),
        write_unmatched_sample=False,
    )
    rep, _sample = build_alignment_quality_report(cfg)
    assert rep["cme_raw_completeness_status"] == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    assert any(("TRUNCATED_BY_LIMIT" in w for w in rep["warnings"]))
    assert rep["quality_decision"] != "PASS"


def test_27_multiday_dry_run_shows_cme_raw_completeness_status(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _seed_mt5(data, "2026-05-18")
    raw = data / "raw" / "cme_sample" / "date=2026-05-18" / "esnq.parquet"
    _write_metadata_sidecar(raw, completeness=cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT)
    cfg = DryRunConfig(
        dates=["2026-05-18"], data_root=data, reports_root=tmp_path / "reports", repo_root=REPO_ROOT
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["cme_raw_completeness_status"] == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    assert day["cme_raw_truncation_warning"] is not None
    assert "TRUNCATED_BY_LIMIT" in day["cme_raw_truncation_warning"]


PHASE_2G2_FILES = (
    "src/polarix/ingestion/cme_downloader.py",
    "src/polarix/orchestration/pipeline_orchestrator.py",
    "src/polarix/orchestration/databento_download_plan.py",
    "src/polarix/orchestration/pipeline_status.py",
    "src/polarix/orchestration/multiday_dry_run_report.py",
    "src/polarix/quality/cme_reference_quality.py",
    "src/polarix/alignment/alignment_quality.py",
    "scripts/download_databento_sample.py",
    "scripts/run_multiday_pipeline.py",
)


def _phase_2g2_texts() -> dict[str, str]:
    return {p: (REPO_ROOT / p).read_text(encoding="utf-8") for p in PHASE_2G2_FILES}


def test_28_no_trading_functions_in_phase_2g2_files() -> None:
    forbidden = (
        "order_send",
        "order_check",
        "order_calc_margin",
        "order_calc_profit",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_29_no_mt5_sdk_imports_in_phase_2g2_files() -> None:
    forbidden = ("import MetaTrader5", "from MetaTrader5", "mt5.order_send", "mt5.copy_ticks_from")
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_30_no_order_send_anywhere() -> None:
    for path, text in _phase_2g2_texts().items():
        assert "order_send" not in text, f"{path}: order_send leaked"


def test_31_no_model_training_imports_in_phase_2g2_files() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "model.fit(",
        "train_test_split",
    )
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_32_no_label_or_target_construction() -> None:
    forbidden = ("labels_y", "target_y", "y_true", "y_pred", "make_labels", "build_targets")
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_33_no_cvd_cumulative_aggregation_in_phase_2g2_files() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_34_no_broad_multiday_databento_request_path() -> None:
    forbidden = ("days=7", "days=30", "weeks=", "timedelta(days=7", "TimeDelta(days=7")
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_35_no_automatic_deletion_of_raw_data() -> None:
    forbidden_fn_names = {"unlink", "rmtree", "remove"}
    for path in PHASE_2G2_FILES:
        try:
            tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                if name in forbidden_fn_names:
                    pytest.fail(f"{path}:{node.lineno}: forbidden deletion call {name!r}")


def test_36_no_fake_cost_estimates_in_phase_2g2_files() -> None:
    forbidden = (
        "estimated_cost_usd=0.50",
        "estimated_cost_usd=1.99",
        "estimated_size_gb=0.5,",
        "estimated_size_gb=1.5,",
    )
    for path, text in _phase_2g2_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: fake estimate {token!r}"


def test_37_no_fake_physical_limit_constants() -> None:
    text = _phase_2g2_texts()["src/polarix/ingestion/cme_downloader.py"]
    forbidden = (
        "physical_limit_value=1000)",
        "physical_limit_value=1_000_000)",
        "physical_limit_value=100)",
    )
    for token in forbidden:
        assert token not in text, f"cme_downloader.py: fake limit {token!r}"


def test_38_no_duplicate_get_range_outside_cme_downloader() -> None:
    offenders: list[str] = []
    for sub in ("src", "scripts"):
        for py in (REPO_ROOT / sub).rglob("*.py"):
            if py.name == "cme_downloader.py":
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "get_range"
                    and isinstance(fn.value, ast.Attribute)
                    and (fn.value.attr == "timeseries")
                ):
                    offenders.append(f"{py.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"duplicate get_range callers detected: {offenders}"


def test_39_no_download_path_ignores_max_download_records() -> None:
    cli = (REPO_ROOT / "scripts" / "download_databento_sample.py").read_text(encoding="utf-8")
    assert "--max-download-records" in cli
    plan_src = (
        REPO_ROOT / "src" / "polarix" / "orchestration" / "databento_download_plan.py"
    ).read_text(encoding="utf-8")
    assert "--max-download-records" in plan_src
    dl = (REPO_ROOT / "src" / "polarix" / "ingestion" / "cme_downloader.py").read_text(
        encoding="utf-8"
    )
    assert "max_download_records" in dl
    assert "request.max_download_records" in dl
