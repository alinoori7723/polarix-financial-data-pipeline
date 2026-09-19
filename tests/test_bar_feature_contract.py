from __future__ import annotations

import ast
from pathlib import Path

import pytest

from polarix.features.bar_feature_contract import (
    ABSOLUTE_PRICE_NON_FEATURE_COLUMNS,
    DIAGNOSTIC_FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    IDENTITY_COLUMNS,
    MODEL_FEATURE_CANDIDATE_COLUMNS,
    NON_FEATURE_COLUMNS,
    QUALITY_COLUMNS,
    FeatureContract,
    FeatureContractError,
    assert_contract_invariants,
)


@pytest.mark.parametrize(
    "col",
    [
        "cme_open_price",
        "cme_close_price",
        "cme_high_price",
        "cme_low_price",
        "cme_vwap",
        "mt5_mid_open",
        "mt5_mid_close",
        "mt5_mid_high",
        "mt5_mid_low",
        "mt5_mid_twap",
        "mt5_mid_mean",
    ],
)
def test_absolute_price_columns_are_non_feature(col: str) -> None:
    assert col in NON_FEATURE_COLUMNS
    assert col not in MODEL_FEATURE_CANDIDATE_COLUMNS


def test_absolute_price_columns_listed_explicitly() -> None:
    for col in ABSOLUTE_PRICE_NON_FEATURE_COLUMNS:
        assert col in NON_FEATURE_COLUMNS


def test_diagnostic_basis_vwap_twap_not_model_candidate() -> None:
    assert "diagnostic_basis_vwap_twap" in DIAGNOSTIC_FEATURE_COLUMNS
    assert "diagnostic_basis_vwap_twap_bps" in DIAGNOSTIC_FEATURE_COLUMNS
    assert "diagnostic_basis_vwap_twap" not in MODEL_FEATURE_CANDIDATE_COLUMNS
    assert "diagnostic_basis_vwap_twap_bps" not in MODEL_FEATURE_CANDIDATE_COLUMNS


def test_basis_close_bps_and_basis_change_bps_are_candidates() -> None:
    assert "basis_close_bps" in MODEL_FEATURE_CANDIDATE_COLUMNS
    assert "basis_change_bps" in MODEL_FEATURE_CANDIDATE_COLUMNS
    assert "return_diff_close_to_close" in MODEL_FEATURE_CANDIDATE_COLUMNS


def test_raw_basis_close_is_non_feature_metadata() -> None:
    assert "basis_close" in NON_FEATURE_COLUMNS
    assert "basis_close" not in MODEL_FEATURE_CANDIDATE_COLUMNS
    assert "basis_change" in NON_FEATURE_COLUMNS
    assert "basis_change" not in MODEL_FEATURE_CANDIDATE_COLUMNS


def test_no_decision_lag_ms_in_contract() -> None:
    all_cols = set(
        IDENTITY_COLUMNS
        + QUALITY_COLUMNS
        + NON_FEATURE_COLUMNS
        + DIAGNOSTIC_FEATURE_COLUMNS
        + MODEL_FEATURE_CANDIDATE_COLUMNS
    )
    assert "decision_lag_ms" not in all_cols
    assert "decision_lag_ms" in FORBIDDEN_COLUMNS


def test_no_decision_lag_ms_in_phase_2d_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    phase_2d_paths = [
        repo_root / "src" / "polarix" / "features" / "bar_feature_contract.py",
        repo_root / "src" / "polarix" / "features" / "bar_feature_builder.py",
        repo_root / "src" / "polarix" / "quality" / "bar_feature_quality.py",
        repo_root / "scripts" / "build_bar_features.py",
        repo_root / "scripts" / "bar_feature_quality_report.py",
    ]
    for p in phase_2d_paths:
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == "decision_lag_ms":
                    assert p.name == "bar_feature_contract.py", (
                        f"{p}:{node.lineno}: 'decision_lag_ms' must only appear in FORBIDDEN_COLUMNS in the contract module"
                    )


def test_no_target_label_columns_anywhere() -> None:
    all_cols = set(
        IDENTITY_COLUMNS
        + QUALITY_COLUMNS
        + NON_FEATURE_COLUMNS
        + DIAGNOSTIC_FEATURE_COLUMNS
        + MODEL_FEATURE_CANDIDATE_COLUMNS
    )
    for forbidden in (
        "label",
        "target",
        "y",
        "y_true",
        "future_return",
        "forward_return",
        "return_t_plus_1",
        "outcome",
    ):
        assert forbidden not in all_cols


def test_assert_contract_invariants_default_ok() -> None:
    assert_contract_invariants(FeatureContract(builder_version="0.1.0"))


def test_contract_rejects_absolute_price_as_candidate() -> None:
    bad = FeatureContract(
        builder_version="bad",
        model_feature_candidate_columns=MODEL_FEATURE_CANDIDATE_COLUMNS + ("cme_close_price",),
    )
    with pytest.raises(FeatureContractError):
        assert_contract_invariants(bad)


def test_contract_rejects_forbidden_column_as_candidate() -> None:
    bad = FeatureContract(
        builder_version="bad",
        model_feature_candidate_columns=MODEL_FEATURE_CANDIDATE_COLUMNS + ("decision_lag_ms",),
    )
    with pytest.raises(FeatureContractError):
        assert_contract_invariants(bad)


def test_contract_rejects_forbidden_column_anywhere() -> None:
    bad = FeatureContract(
        builder_version="bad",
        non_feature_columns=NON_FEATURE_COLUMNS + ("cme_max_1s_volume_share",),
    )
    with pytest.raises(FeatureContractError):
        assert_contract_invariants(bad)


def test_contract_all_columns_deduplicates_quality_candidate_overlap() -> None:
    contract = FeatureContract(builder_version="0.1.0")
    cols = contract.all_columns()
    assert cols.count("mt5_join_safe_tick_ratio") == 1


PHASE_2D_FILES = (
    "src/polarix/features/bar_feature_contract.py",
    "src/polarix/features/bar_feature_builder.py",
    "src/polarix/quality/bar_feature_quality.py",
    "scripts/build_bar_features.py",
    "scripts/bar_feature_quality_report.py",
)


def _phase_2d_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2D_FILES}


def test_phase_2d_no_trading_functions() -> None:
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
    for path, text in _phase_2d_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2d_no_mt5_sdk_imports() -> None:
    forbidden = (
        "import MetaTrader5",
        "from MetaTrader5",
        "import polarix.ingestion.mt5_readonly",
        "from polarix.ingestion.mt5_readonly",
        "mt5.order_send",
        "mt5.copy_ticks_from",
    )
    for path, text in _phase_2d_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2d_no_model_training_imports() -> None:
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
    for path, text in _phase_2d_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2d_no_cvd_cumulative_aggregation() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _phase_2d_texts().items():
        if path.endswith("bar_feature_contract.py"):
            continue
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"
    for path, text in _phase_2d_texts().items():
        if path.endswith("bar_feature_contract.py"):
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.strip().lower() == "cvd":
                    pytest.fail(f"{path}:{node.lineno}: bare 'cvd' literal")


def test_phase_2d_no_max_1s_feature_names() -> None:
    forbidden = (
        "cme_max_1s_volume_share",
        "cme_max_1s_signed_volume_share",
        "max_1s_volume_share",
        "max_1s_signed_volume_share",
    )
    for path, text in _phase_2d_texts().items():
        if path.endswith("bar_feature_contract.py"):
            continue
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2d_no_synthetic_overlap_real_report_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2d_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2d_does_not_weaken_50ms_tick_contract() -> None:
    for path, text in _phase_2d_texts().items():
        assert "DEFAULT_ALIGNMENT_TOLERANCE_MS" not in text, (
            f"{path}: must not redefine tick-level tolerance"
        )
    from polarix.alignment.alignment_contract import DEFAULT_ALIGNMENT_TOLERANCE_MS as TOL_CONTRACT
    from polarix.alignment.alignment_quality import DEFAULT_ALIGNMENT_TOLERANCE_MS as TOL_QUALITY

    assert TOL_CONTRACT == 50
    assert TOL_QUALITY == 50
