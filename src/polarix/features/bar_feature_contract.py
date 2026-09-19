from __future__ import annotations

from dataclasses import dataclass

IDENTITY_COLUMNS: tuple[str, ...] = (
    "feature_row_id",
    "date",
    "cme_symbol",
    "mt5_symbol",
    "symbol_pair",
    "bucket_size",
    "bucket_start_utc_ns",
    "bucket_end_utc_ns",
    "feature_timestamp_utc_ns",
    "data_layer",
    "builder_version",
)
QUALITY_COLUMNS: tuple[str, ...] = (
    "has_cme_data",
    "has_mt5_data",
    "is_bar_aligned",
    "bar_reject_reason",
    "cme_volume_alignment_ratio",
    "mt5_join_safe_tick_ratio",
    "feature_quality_flag",
    "is_model_eligible_candidate",
)
FEATURE_QUALITY_OK = "OK"
FEATURE_QUALITY_DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
FEATURE_QUALITY_LOW_JOIN_SAFE_RATIO = "LOW_JOIN_SAFE_RATIO"
FEATURE_QUALITY_MISSING_CME = "MISSING_CME"
FEATURE_QUALITY_MISSING_MT5 = "MISSING_MT5"
FEATURE_QUALITY_ZERO_VOLUME = "ZERO_VOLUME"
FEATURE_QUALITY_SPREAD_EXTREME = "SPREAD_EXTREME"
FEATURE_QUALITY_INVALID_PRICE = "INVALID_PRICE"
FEATURE_QUALITY_OTHER_REJECT = "OTHER_REJECT"
FEATURE_QUALITY_VALUES: tuple[str, ...] = (
    FEATURE_QUALITY_OK,
    FEATURE_QUALITY_DIAGNOSTIC_ONLY,
    FEATURE_QUALITY_LOW_JOIN_SAFE_RATIO,
    FEATURE_QUALITY_MISSING_CME,
    FEATURE_QUALITY_MISSING_MT5,
    FEATURE_QUALITY_ZERO_VOLUME,
    FEATURE_QUALITY_SPREAD_EXTREME,
    FEATURE_QUALITY_INVALID_PRICE,
    FEATURE_QUALITY_OTHER_REJECT,
)
NON_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "basis_close",
    "basis_change",
    "cme_buy_volume",
    "cme_sell_volume",
    "cme_unknown_aggressor_volume",
    "cme_net_signed_volume",
)
DIAGNOSTIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "diagnostic_basis_vwap_twap",
    "diagnostic_basis_vwap_twap_bps",
)
MODEL_FEATURE_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "cme_return_close_to_close",
    "cme_vwap_return",
    "cme_high_low_range_bps",
    "cme_trade_count",
    "cme_total_volume",
    "cme_buy_volume_ratio",
    "cme_sell_volume_ratio",
    "cme_unknown_aggressor_volume_ratio",
    "cme_signed_volume_ratio",
    "cme_volume_per_trade",
    "cme_trade_intensity",
    "cme_max_single_trade_volume",
    "cme_trade_size_p99",
    "cme_trade_size_mean",
    "cme_trade_size_std",
    "cme_volume_zscore_by_symbol_bucket",
    "cme_trade_count_zscore_by_symbol_bucket",
    "cme_range_zscore_by_symbol_bucket",
    "mt5_mid_return_close_to_close",
    "mt5_mid_range_bps",
    "mt5_spread_price_mean",
    "mt5_spread_price_p50",
    "mt5_spread_price_p95",
    "mt5_spread_price_max",
    "mt5_spread_points_mean",
    "mt5_spread_points_p95",
    "mt5_spread_points_max",
    "mt5_join_safe_tick_ratio",
    "mt5_tick_count",
    "mt5_residual_ms_p50",
    "mt5_residual_ms_p95",
    "mt5_residual_ms_max",
    "mt5_spread_zscore_by_symbol_bucket",
    "mt5_tick_count_zscore_by_symbol_bucket",
    "basis_close_bps",
    "basis_change_bps",
    "return_diff_close_to_close",
)
ABSOLUTE_PRICE_NON_FEATURE_COLUMNS: tuple[str, ...] = (
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
)
FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "label",
    "target",
    "y",
    "y_true",
    "future_return",
    "forward_return",
    "return_t_plus_1",
    "outcome",
    "cvd",
    "cvd_cumulative",
    "cvd_running",
    "cvd_total",
    "cme_max_1s_volume_share",
    "cme_max_1s_signed_volume_share",
    "decision_lag_ms",
)


@dataclass(frozen=True)
class FeatureContract:
    builder_version: str
    identity_columns: tuple[str, ...] = IDENTITY_COLUMNS
    quality_columns: tuple[str, ...] = QUALITY_COLUMNS
    non_feature_columns: tuple[str, ...] = NON_FEATURE_COLUMNS
    diagnostic_feature_columns: tuple[str, ...] = DIAGNOSTIC_FEATURE_COLUMNS
    model_feature_candidate_columns: tuple[str, ...] = MODEL_FEATURE_CANDIDATE_COLUMNS
    forbidden_columns: tuple[str, ...] = FORBIDDEN_COLUMNS

    def all_columns(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for col in (
            self.identity_columns
            + self.quality_columns
            + self.non_feature_columns
            + self.diagnostic_feature_columns
            + self.model_feature_candidate_columns
        ):
            if col not in seen:
                seen.add(col)
                out.append(col)
        return tuple(out)

    def to_dict(self) -> dict:
        return {
            "builder_version": self.builder_version,
            "identity_columns": list(self.identity_columns),
            "quality_columns": list(self.quality_columns),
            "non_feature_columns": list(self.non_feature_columns),
            "diagnostic_feature_columns": list(self.diagnostic_feature_columns),
            "model_feature_candidate_columns": list(self.model_feature_candidate_columns),
            "forbidden_columns": list(self.forbidden_columns),
        }


class FeatureContractError(ValueError):
    pass


def assert_contract_invariants(contract: FeatureContract) -> None:
    sets = {
        "identity": set(contract.identity_columns),
        "quality": set(contract.quality_columns),
        "non_feature": set(contract.non_feature_columns),
        "diagnostic": set(contract.diagnostic_feature_columns),
        "candidate": set(contract.model_feature_candidate_columns),
    }
    ALLOWED_OVERLAP_PAIRS = {frozenset({"quality", "candidate"})}
    for name, cols in sets.items():
        for other_name, other in sets.items():
            if other_name <= name:
                continue
            if frozenset({name, other_name}) in ALLOWED_OVERLAP_PAIRS:
                continue
            overlap = cols & other
            if overlap:
                raise FeatureContractError(
                    f"contract sets {name!r} and {other_name!r} overlap on {sorted(overlap)}"
                )
    for col in ABSOLUTE_PRICE_NON_FEATURE_COLUMNS:
        if col not in contract.non_feature_columns:
            raise FeatureContractError(
                f"absolute-price column {col!r} must be in non_feature_columns"
            )
        if col in contract.model_feature_candidate_columns:
            raise FeatureContractError(
                f"absolute-price column {col!r} must NOT be a model feature candidate"
            )
    forbidden = set(contract.forbidden_columns)
    bad_candidates = forbidden & set(contract.model_feature_candidate_columns)
    if bad_candidates:
        raise FeatureContractError(
            f"forbidden columns appear as model feature candidates: {sorted(bad_candidates)}"
        )
    bad_diag = set(contract.diagnostic_feature_columns) & set(
        contract.model_feature_candidate_columns
    )
    if bad_diag:
        raise FeatureContractError(
            f"diagnostic features must not be model feature candidates: {sorted(bad_diag)}"
        )
    all_cols = set(contract.all_columns())
    bad_any = forbidden & all_cols
    if bad_any:
        raise FeatureContractError(f"forbidden columns appear in schema: {sorted(bad_any)}")


assert_contract_invariants(FeatureContract(builder_version="0.0.0"))
