from __future__ import annotations

import datetime as _dt

import pyarrow as pa
import pytest

from polarix.ingestion.cme_databento_schema import (
    AGGRESSOR_BUY,
    AGGRESSOR_SELL,
    AGGRESSOR_UNKNOWN,
    CmeSchemaError,
    detect_binding,
    map_aggressor,
    signed_size,
    to_ns_int64,
)


def test_detect_binding_with_full_mbp1_layout() -> None:
    cols = [
        "ts_event",
        "ts_recv",
        "raw_symbol",
        "instrument_id",
        "action",
        "side",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
    ]
    b = detect_binding(cols)
    assert b.ts_event == "ts_event"
    assert b.ts_recv == "ts_recv"
    assert b.action == "action"
    assert b.side == "side"
    assert b.price == "price"
    assert b.size == "size"
    assert b.raw_symbol == "raw_symbol"
    assert b.instrument_id == "instrument_id"
    assert b.is_bbo_available is True


@pytest.mark.parametrize(
    "drop,expected_token", [("ts_event", "ts_event"), ("action", "action"), ("side", "side")]
)
def test_detect_binding_rejects_missing_required(drop: str, expected_token: str) -> None:
    cols = ["ts_event", "ts_recv", "raw_symbol", "action", "side", "price", "size"]
    cols.remove(drop)
    with pytest.raises(CmeSchemaError) as ei:
        detect_binding(cols)
    assert expected_token in str(ei.value)


def test_detect_binding_accepts_direct_symbol_column() -> None:
    cols = ["ts_event", "ts_recv", "symbol", "action", "side", "price", "size"]
    b = detect_binding(cols)
    assert b.raw_symbol == "symbol"
    assert b.instrument_id is None


def test_detect_binding_requires_map_for_instrument_id_only() -> None:
    cols = ["ts_event", "ts_recv", "instrument_id", "action", "side", "price", "size"]
    with pytest.raises(CmeSchemaError) as ei:
        detect_binding(cols)
    assert "instrument_id_to_symbol_map" in str(ei.value)
    b = detect_binding(cols, instrument_id_to_symbol={1: "ES"})
    assert b.raw_symbol is None
    assert b.instrument_id == "instrument_id"


def test_detect_binding_detects_bbo_present() -> None:
    cols = [
        "ts_event",
        "ts_recv",
        "raw_symbol",
        "action",
        "side",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    ]
    b = detect_binding(cols)
    assert b.is_bbo_available is True


def test_detect_binding_marks_bbo_missing() -> None:
    cols = ["ts_event", "ts_recv", "raw_symbol", "action", "side", "price", "size"]
    b = detect_binding(cols)
    assert b.is_bbo_available is False
    assert b.bid_px is None
    assert b.ask_px is None


def test_aggressor_mapping_buy() -> None:
    assert map_aggressor("B") == AGGRESSOR_BUY
    assert signed_size("B", 5) == 5.0
    assert signed_size("B", 1) > 0


def test_aggressor_mapping_sell() -> None:
    assert map_aggressor("A") == AGGRESSOR_SELL
    assert signed_size("A", 5) == -5.0
    assert signed_size("A", 1) < 0


@pytest.mark.parametrize("value", ["N", "", None, "?", "X"])
def test_aggressor_unknown_for_non_buy_sell(value) -> None:
    assert map_aggressor(value) == AGGRESSOR_UNKNOWN
    assert signed_size(value, 5) is None


def test_aggressor_mapping_not_inverted() -> None:
    for _ in range(1000):
        assert map_aggressor("B") != AGGRESSOR_SELL
        assert map_aggressor("A") != AGGRESSOR_BUY


def test_signed_size_none_when_size_none() -> None:
    assert signed_size("B", None) is None
    assert signed_size("A", None) is None
    assert signed_size("N", None) is None


def test_to_ns_int64_passes_int64_through() -> None:
    arr = pa.array([1700000000000000000, 1700000000000000001], type=pa.int64())
    out = to_ns_int64(arr, name="ts_event")
    assert out.type == pa.int64()
    assert out.to_pylist() == arr.to_pylist()


def test_to_ns_int64_accepts_utc_timestamp() -> None:
    dts = [
        _dt.datetime(2026, 5, 18, 12, 0, 0, tzinfo=_dt.timezone.utc),
        _dt.datetime(2026, 5, 18, 12, 0, 1, tzinfo=_dt.timezone.utc),
    ]
    arr = pa.array(dts, type=pa.timestamp("ns", tz="UTC"))
    out = to_ns_int64(arr, name="ts_event")
    assert out.type == pa.int64()
    diffs = [b - a for a, b in zip(out.to_pylist(), out.to_pylist()[1:])]
    assert diffs == [1000000000]


def test_to_ns_int64_rejects_naive_timestamp() -> None:
    arr = pa.array([_dt.datetime(2026, 5, 18, 12, 0, 0)], type=pa.timestamp("ns"))
    with pytest.raises(CmeSchemaError):
        to_ns_int64(arr, name="ts_event")


def test_to_ns_int64_rejects_non_timestamp() -> None:
    arr = pa.array(["2026-05-18"], type=pa.string())
    with pytest.raises(CmeSchemaError):
        to_ns_int64(arr, name="ts_event")
