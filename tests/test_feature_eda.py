from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from polarix.features.bar_aggregation import parse_bucket_sizes
from polarix.features.bar_feature_builder import BuilderConfig, build
from polarix.features.feature_eda import FeatureEDAConfig, FeatureEDAError, run_eda, write_reports
from tests.test_bar_feature_builder import _matched, _seed


def _build_gold(tmp_path: Path, *, include_5s: bool = False, bucket_sizes: str = "15s,60s") -> Path:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=12, bucket_seconds=15)
    cme_nq, mt5_ndx = _matched(base, n=12, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    cfg = BuilderConfig(
        date="2026-05-18",
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


def test_run_eda_loads_manifest_and_excludes_absolute_prices(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        bucket_sizes=("15s", "60s"),
        small_sample_min_rows=5000,
        dry_run=True,
    )
    res = run_eda(cfg)
    roles = res.report["feature_roles"]
    for col in ("cme_close_price", "mt5_mid_close", "mt5_mid_twap", "cme_vwap", "cme_open_price"):
        assert col not in roles["model_feature_candidate_columns"]
        assert col in roles["non_feature_columns"]


def test_run_eda_fails_if_manifest_missing(tmp_path: Path) -> None:
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=tmp_path / "absent_features",
        reports_root=tmp_path / "reports",
    )
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert res.report["decision_reason"] == "MISSING_FEATURE_MANIFEST"


def test_run_eda_fails_if_absolute_price_in_candidates(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    manifest_path = features_root / "date=2026-05-18" / "bar_features_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_contract"]["model_feature_candidate_columns"].append("cme_close_price")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        dry_run=True,
    )
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert any(("absolute-price" in e for e in res.report["errors"]))


def test_run_eda_fails_if_forbidden_column_in_candidates(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    manifest_path = features_root / "date=2026-05-18" / "bar_features_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_contract"]["model_feature_candidate_columns"].append("label")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        dry_run=True,
    )
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "FAIL"
    assert any(("forbidden" in e for e in res.report["errors"]))


def test_run_eda_partial_when_sample_below_threshold(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    res = run_eda(cfg)
    assert res.report["quality_decision"] == "PARTIAL"
    assert res.report["small_sample_flag"] is True
    assert any(("small_sample_min_rows" in w for w in res.report["warnings"]))
    assert res.paths["missingness_summary_path"]
    assert res.paths["distribution_summary_path"]
    assert res.paths["correlation_summary_path"]


def test_run_eda_writes_three_summary_parquets(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    res = run_eda(cfg)
    for k in ("missingness_summary_path", "distribution_summary_path", "correlation_summary_path"):
        assert Path(res.paths[k]).exists()


def test_write_reports_emits_json_and_txt(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    res = run_eda(cfg)
    json_path, txt_path = write_reports(cfg, res)
    assert json_path.exists() and txt_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["quality_decision"] == res.report["quality_decision"]
    assert "Feature EDA Report" in txt_path.read_text(encoding="utf-8")


def test_diagnostic_5s_kept_separate_from_main_eda(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path, include_5s=True)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        include_diagnostic_5s=True,
        force=True,
    )
    res = run_eda(cfg)
    dist = res.distribution_summary
    assert dist is not None
    assert "5s" not in set(dist["bucket_size"].drop_nulls().to_list())
    assert res.report["diagnostic_block"]["rows"] > 0


def test_existing_output_requires_force(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    run_eda(cfg)
    cfg2 = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=False,
    )
    with pytest.raises(FeatureEDAError):
        run_eda(cfg2)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        dry_run=True,
    )
    run_eda(cfg)
    assert not list((tmp_path / "reports").glob("feature_*_2026-05-18*"))


def test_stability_summary_present_and_intra_sample(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    res = run_eda(cfg)
    assert res.stability_summary is not None
    assert res.stability_summary.height > 0
    txt = json.dumps(res.report)
    assert "small_sample" in txt or "small_sample_min_rows" in txt


PHASE_2E_FILES = (
    "src/polarix/features/feature_eda.py",
    "src/polarix/features/feature_statistics.py",
    "src/polarix/features/feature_correlation.py",
    "scripts/feature_eda_report.py",
)


def _phase_2e_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2E_FILES}


def test_phase_2e_no_trading_functions() -> None:
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
    for path, text in _phase_2e_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2e_no_mt5_sdk_imports() -> None:
    forbidden = (
        "import MetaTrader5",
        "from MetaTrader5",
        "import polarix.ingestion.mt5_readonly",
        "from polarix.ingestion.mt5_readonly",
        "mt5.order_send",
        "mt5.copy_ticks_from",
    )
    for path, text in _phase_2e_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2e_no_model_training_imports() -> None:
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
        "model_training",
        "train_test_split",
    )
    for path, text in _phase_2e_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2e_no_cvd_cumulative_aggregation() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _phase_2e_texts().items():
        if path.endswith("feature_eda.py"):
            continue
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2e_no_winsorization_clipping_mutation() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_call_names = {"winsorize", "winsorise", "clip", "clip_min", "clip_max"}
    for path in PHASE_2E_FILES:
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = None
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname in forbidden_call_names:
                    pytest.fail(f"{path}:{node.lineno}: forbidden call to {fname!r}")


def test_phase_2e_does_not_weaken_50ms_tick_contract() -> None:
    for path, text in _phase_2e_texts().items():
        assert "DEFAULT_ALIGNMENT_TOLERANCE_MS" not in text, (
            f"{path}: must not redefine tick-level tolerance"
        )
    from polarix.alignment.alignment_contract import DEFAULT_ALIGNMENT_TOLERANCE_MS as TC
    from polarix.alignment.alignment_quality import DEFAULT_ALIGNMENT_TOLERANCE_MS as TQ

    assert TC == 50
    assert TQ == 50


def test_phase_2e_no_synthetic_overlap_real_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2e_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2e_does_not_write_back_to_features_root(tmp_path: Path) -> None:
    features_root = _build_gold(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in features_root.rglob("*.parquet")}
    cfg = FeatureEDAConfig(
        date="2026-05-18",
        features_root=features_root,
        reports_root=tmp_path / "reports",
        force=True,
    )
    run_eda(cfg)
    after = {p: p.stat().st_mtime_ns for p in features_root.rglob("*.parquet")}
    assert before == after, "Phase 2E must not modify any file under features_root"
