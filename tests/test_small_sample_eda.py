from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.features.bar_aggregation import parse_bucket_sizes
from polarix.features.bar_feature_builder import BuilderConfig, build
from polarix.features.feature_eda import (
    DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS,
    DECISION_REASON_MISSING_FEATURE_DATA,
    DECISION_REASON_MISSING_FEATURE_MANIFEST,
    TINY_SAMPLE_NON_PREDICTIVE_WARNING,
    FeatureEDAConfig,
    run_eda,
    write_reports,
)
from polarix.orchestration import pipeline_status as ps
from polarix.orchestration.pipeline_orchestrator import OrchestratorConfig, process_one_date
from tests.test_bar_feature_builder import _matched, _seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_gold(
    tmp_path: Path,
    *,
    n_matched: int = 12,
    bucket_sizes: str = "15s,60s",
    include_5s: bool = False,
    date: str = "2026-05-18",
) -> Path:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=n_matched, bucket_seconds=15)
    cme_nq, mt5_ndx = _matched(base, n=n_matched, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, date, {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    cfg = BuilderConfig(
        date=date,
        cme_root=cme_root,
        mt5_root=mt5_root,
        output_root=tmp_path / "features",
        reports_root=tmp_path / "reports",
        bucket_sizes=parse_bucket_sizes(bucket_sizes),
        include_diagnostic_5s=include_5s,
        force=True,
    )
    build(cfg)
    return cfg.output_root


def _eda_config(
    tmp_path: Path, features_root: Path, *, date: str = "2026-05-18", **kw
) -> FeatureEDAConfig:
    return FeatureEDAConfig(
        date=date, features_root=features_root, reports_root=tmp_path / "reports", force=True, **kw
    )


def test_1_feature_files_exist_but_eligible_below_threshold_is_partial(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = _eda_config(tmp_path, features_root, small_sample_min_rows=5000)
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "PARTIAL"
    assert res.report["decision_reason"] == DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS
    assert res.report["small_sample_flag"] is True
    assert res.report["feature_files_present"] is True
    assert res.report["feature_file_count"] > 0
    assert any((TINY_SAMPLE_NON_PREDICTIVE_WARNING in w for w in res.report["warnings"]))


def test_2_no_feature_parquet_files_is_fail_missing_feature_data(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    for pq_file in features_root.rglob("part-*.parquet"):
        pq_file.unlink()
    cfg = _eda_config(tmp_path, features_root)
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert res.report["decision_reason"] == DECISION_REASON_MISSING_FEATURE_DATA
    assert res.report["feature_files_present"] is False


def test_3_missing_manifest_is_fail(tmp_path: Path) -> None:
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=tmp_path / "absent_features",
        reports_root=tmp_path / "reports",
    )
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert res.report["decision_reason"] == DECISION_REASON_MISSING_FEATURE_MANIFEST


def test_4_tiny_sample_still_writes_json_and_txt(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = _eda_config(tmp_path, features_root)
    res = run_eda(cfg)
    json_path, txt_path = write_reports(cfg, res)
    assert json_path.exists() and txt_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["quality_decision"] == "PARTIAL"
    assert parsed["decision_reason"] == DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS
    assert (
        TINY_SAMPLE_NON_PREDICTIVE_WARNING in txt_path.read_text(encoding="utf-8")
        or "small_sample" in txt_path.read_text(encoding="utf-8")
        or "small_sample_min_rows" in txt_path.read_text(encoding="utf-8")
    )


def test_5_tiny_sample_does_not_run_model_training(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = _eda_config(tmp_path, features_root)
    res = run_eda(cfg)
    assert res.report["data_mutation_check"]["transformations_applied"] == []
    text = (REPO_ROOT / "src" / "polarix" / "features" / "feature_eda.py").read_text(
        encoding="utf-8"
    )
    for tok in ("model.fit(", "make_labels(", "build_targets(", "train_test_split("):
        assert tok not in text, f"feature_eda.py introduced forbidden {tok!r}"


def test_6_correlation_summary_does_not_turn_partial_into_fail(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = _eda_config(tmp_path, features_root)
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "PARTIAL"
    summary = res.correlation_summary
    if summary is not None and summary.height > 0:
        flags = summary["is_small_sample_warning"].to_list()
        assert all(flags), "every correlation row should be flagged small-sample"


def test_7_orchestrator_treats_partial_eda_as_non_fatal_diagnostic(tmp_path: Path) -> None:
    data = tmp_path / "data"
    t0 = int(_dt.datetime.fromisoformat("2026-05-19T05:00:00+00:00").timestamp() * 1000)
    for sym in ("SPX500", "NDX100"):
        path = (
            data
            / "normalized"
            / "mt5_ticks"
            / f"symbol={sym}"
            / "date=2026-05-19"
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
    raw = data / "raw" / "cme_sample" / "date=2026-05-19" / "today.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), raw)
    cfg = OrchestratorConfig(
        dates=["2026-05-19"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        python_exe=Path("python"),
        dry_run=False,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
    )

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-19", runner=runner)
    eda_status = next((s["status"] for s in day.stages if s["name"] == "FEATURE_EDA"), None)
    assert eda_status == ps.STAGE_OK, (
        f"FEATURE_EDA should be STAGE_OK when CLI exits 0; got {eda_status!r}"
    )
    assert day.final_status != ps.STATUS_DAY_FAILED


def test_8_2026_05_19_like_fixture_returns_partial_not_fail(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path, n_matched=20, date="2026-05-19")
    cfg = _eda_config(tmp_path, features_root, date="2026-05-19")
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "PARTIAL"
    assert res.report["decision_reason"] == DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS
    assert res.report["feature_files_present"] is True
    assert res.report["decision_reason"] != DECISION_REASON_MISSING_FEATURE_DATA


def test_9_no_trading_functions_in_phase_2e1_files() -> None:
    forbidden = (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    files = ("src/polarix/features/feature_eda.py", "scripts/feature_eda_report.py")
    for path in files:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{path}: forbidden {tok!r}"


def test_10_no_mt5_sdk_imports_in_phase_2e1_files() -> None:
    forbidden = ("import MetaTrader5", "from MetaTrader5", "mt5.order_send", "mt5.copy_ticks_from")
    files = ("src/polarix/features/feature_eda.py", "scripts/feature_eda_report.py")
    for path in files:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_11_no_model_training_imports_in_phase_2e1_files() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "model.fit(",
        "train_test_split",
    )
    files = ("src/polarix/features/feature_eda.py", "scripts/feature_eda_report.py")
    for path in files:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{path}: {tok!r} present"


def test_12_no_cvd_cumulative_aggregation_in_phase_2e1_files() -> None:
    eda_text = (REPO_ROOT / "src" / "polarix" / "features" / "feature_eda.py").read_text(
        encoding="utf-8"
    )
    for call_site in ("cumsum(", "cumulative_sum("):
        assert call_site not in eda_text, (
            f"feature_eda.py introduced forbidden call site {call_site!r}"
        )
    cli_text = (REPO_ROOT / "scripts" / "feature_eda_report.py").read_text(encoding="utf-8")
    for tok in ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative"):
        assert tok not in cli_text, f"feature_eda_report.py: {tok!r} present"


def test_13_no_winsorization_or_data_mutation_in_phase_2e1_files() -> None:
    forbidden_substr = (
        "winsorize",
        ".clip(",
        "clip_upper",
        "clip_lower",
        "impute",
        "MinMaxScaler",
        "StandardScaler",
        "fit_transform",
        "transform_inplace",
    )
    text = (REPO_ROOT / "src" / "polarix" / "features" / "feature_eda.py").read_text(
        encoding="utf-8"
    )
    for tok in forbidden_substr:
        assert tok not in text, f"feature_eda.py: forbidden {tok!r}"
