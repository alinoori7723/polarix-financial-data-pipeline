from __future__ import annotations

import ast
from pathlib import Path

import pytest

from polarix.alignment.alignment_contract import (
    DEFAULT_ALIGNMENT_TOLERANCE_MS,
    MS_TO_NS,
    MT5QuoteSlice,
    align_many,
    align_one,
)


def _quotes(
    t_ms: list[int], mid: list[float], spread: list[float], safe: list[bool]
) -> MT5QuoteSlice:
    return MT5QuoteSlice(time_msc_utc_ms=t_ms, mid=mid, spread_price=spread, is_join_safe=safe)


def test_aligns_to_latest_quote_at_or_before_event() -> None:
    q = _quotes(
        t_ms=[1000, 1010, 1020],
        mid=[100.0, 100.1, 100.2],
        spread=[0.5, 0.5, 0.5],
        safe=[True, True, True],
    )
    cme_event = 1020 * MS_TO_NS
    r = align_one(cme_event, q)
    assert r.is_aligned
    assert r.matched_mt5_time_msc_utc_ms == 1020
    assert r.matched_mt5_mid == 100.2
    assert r.alignment_delta_ms == 0.0
    r2 = align_one(1025 * MS_TO_NS, q)
    assert r2.is_aligned
    assert r2.matched_mt5_time_msc_utc_ms == 1020
    assert r2.alignment_delta_ms == 5.0


def test_does_not_select_future_quote() -> None:
    q = _quotes(t_ms=[1000, 1050], mid=[100.0, 100.5], spread=[0.5, 0.5], safe=[True, True])
    r = align_one(1010 * MS_TO_NS, q)
    assert r.is_aligned
    assert r.matched_mt5_time_msc_utc_ms == 1000


def test_rejects_match_outside_tolerance() -> None:
    q = _quotes(t_ms=[1000], mid=[100.0], spread=[0.5], safe=[True])
    r = align_one(1060 * MS_TO_NS, q)
    assert r.is_aligned is False
    assert r.matched_mt5_time_msc_utc_ms is None
    assert r.alignment_delta_ms is None


def test_skips_join_unsafe_rows_and_picks_earlier_safe() -> None:
    q = _quotes(
        t_ms=[1000, 1010, 1020],
        mid=[100.0, 100.1, 100.2],
        spread=[0.5, 0.5, 0.5],
        safe=[True, False, False],
    )
    r = align_one(1020 * MS_TO_NS, q)
    assert r.is_aligned
    assert r.matched_mt5_time_msc_utc_ms == 1000


def test_unaligned_when_all_recent_quotes_unsafe() -> None:
    q = _quotes(
        t_ms=[990, 1000, 1010],
        mid=[100.0, 100.1, 100.2],
        spread=[0.5, 0.5, 0.5],
        safe=[False, False, False],
    )
    r = align_one(1020 * MS_TO_NS, q)
    assert r.is_aligned is False


def test_alignment_delta_ms_value() -> None:
    q = _quotes(t_ms=[1000], mid=[100.0], spread=[0.5], safe=[True])
    r = align_one(1037 * MS_TO_NS, q)
    assert r.is_aligned
    assert r.alignment_delta_ms == 37.0


def test_default_tolerance_is_50ms() -> None:
    assert DEFAULT_ALIGNMENT_TOLERANCE_MS == 50
    q = _quotes(t_ms=[1000], mid=[100.0], spread=[0.5], safe=[True])
    assert align_one(1050 * MS_TO_NS, q).is_aligned is True
    assert align_one(1051 * MS_TO_NS, q).is_aligned is False


def test_align_many_works_for_multiple_events() -> None:
    q = _quotes(
        t_ms=[1000, 1100, 1200],
        mid=[100.0, 100.1, 100.2],
        spread=[0.5, 0.5, 0.5],
        safe=[True, True, True],
    )
    events_ns = [t * MS_TO_NS for t in (1010, 1100, 1300)]
    results = align_many(events_ns, q)
    assert len(results) == 3
    assert results[0].matched_mt5_time_msc_utc_ms == 1000
    assert results[1].matched_mt5_time_msc_utc_ms == 1100
    assert results[2].is_aligned is False


def test_align_one_rejects_invalid_tolerance() -> None:
    q = _quotes(t_ms=[1000], mid=[100.0], spread=[0.5], safe=[True])
    with pytest.raises(ValueError):
        align_one(1000 * MS_TO_NS, q, alignment_tolerance_ms=0)


def test_mt5_quote_slice_rejects_unsorted_input() -> None:
    with pytest.raises(ValueError):
        MT5QuoteSlice(
            time_msc_utc_ms=[1000, 990],
            mid=[100.0, 99.9],
            spread_price=[0.5, 0.5],
            is_join_safe=[True, True],
        )


PHASE_2A_FILES = (
    "src/polarix/ingestion/cme_databento_schema.py",
    "src/polarix/ingestion/cme_reference_ingest.py",
    "src/polarix/quality/cme_reference_quality.py",
    "src/polarix/alignment/alignment_contract.py",
    "scripts/ingest_cme_reference_sample.py",
    "scripts/cme_reference_quality_report.py",
)


def _phase_2a_file_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2A_FILES}


def test_no_trading_functions_in_phase_2a() -> None:
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
    for path, text in _phase_2a_file_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden token {token!r}"


def test_no_model_training_in_phase_2a() -> None:
    forbidden = (
        "model.fit(",
        "model_training",
        "train_test_split",
        "xgboost",
        "lightgbm",
        "sklearn.train",
    )
    for path, text in _phase_2a_file_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden token {token!r}"


def test_no_cvd_aggregation_in_phase_2a() -> None:
    forbidden_substrings = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running")
    for path, text in _phase_2a_file_texts().items():
        for token in forbidden_substrings:
            assert token not in text, f"{path}: forbidden token {token!r}"
    for path, text in _phase_2a_file_texts().items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value.lower() != "cvd", f"{path}:{node.lineno}: forbidden 'cvd' literal"


def test_no_hardcoded_broker_offset_in_phase_2a() -> None:
    for path, text in _phase_2a_file_texts().items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 180:
                pytest.fail(f"{path}:{node.lineno}: forbidden numeric literal 180")
