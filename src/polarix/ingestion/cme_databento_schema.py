from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pyarrow as pa

VENDOR = "Databento"
DATASET = "GLBX.MDP3"
SCHEMA = "mbp-1"
TS_EVENT_CANDIDATES = ("ts_event",)
TS_RECV_CANDIDATES = ("ts_recv",)
RAW_SYMBOL_CANDIDATES = ("raw_symbol", "symbol")
INSTRUMENT_ID_CANDIDATES = ("instrument_id",)
ACTION_CANDIDATES = ("action",)
SIDE_CANDIDATES = ("side",)
PRICE_CANDIDATES = ("price",)
SIZE_CANDIDATES = ("size",)
BID_PX_LEVEL0_CANDIDATES = ("bid_px_00", "best_bid_px")
ASK_PX_LEVEL0_CANDIDATES = ("ask_px_00", "best_ask_px")
BID_SZ_LEVEL0_CANDIDATES = ("bid_sz_00", "best_bid_sz")
ASK_SZ_LEVEL0_CANDIDATES = ("ask_sz_00", "best_ask_sz")
BID_CT_LEVEL0_CANDIDATES = ("bid_ct_00", "best_bid_ct")
ASK_CT_LEVEL0_CANDIDATES = ("ask_ct_00", "best_ask_ct")


class CmeSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class CmeColumnBinding:
    ts_event: str
    ts_recv: str
    action: str
    side: str
    price: str
    size: str
    raw_symbol: Optional[str]
    instrument_id: Optional[str]
    bid_px: Optional[str]
    ask_px: Optional[str]
    bid_sz: Optional[str]
    ask_sz: Optional[str]
    bid_ct: Optional[str]
    ask_ct: Optional[str]

    @property
    def is_bbo_available(self) -> bool:
        return all((c is not None for c in (self.bid_px, self.ask_px, self.bid_sz, self.ask_sz)))


def _first_present(names: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None


def detect_binding(
    column_names: list[str], *, instrument_id_to_symbol: Optional[Mapping[int, str]] = None
) -> CmeColumnBinding:
    missing: list[str] = []
    ts_event = _first_present(column_names, TS_EVENT_CANDIDATES)
    if ts_event is None:
        missing.append("ts_event")
    ts_recv = _first_present(column_names, TS_RECV_CANDIDATES)
    if ts_recv is None:
        missing.append("ts_recv")
    action = _first_present(column_names, ACTION_CANDIDATES)
    if action is None:
        missing.append("action")
    side = _first_present(column_names, SIDE_CANDIDATES)
    if side is None:
        missing.append("side")
    price = _first_present(column_names, PRICE_CANDIDATES)
    if price is None:
        missing.append("price")
    size = _first_present(column_names, SIZE_CANDIDATES)
    if size is None:
        missing.append("size")
    raw_symbol = _first_present(column_names, RAW_SYMBOL_CANDIDATES)
    instrument_id = _first_present(column_names, INSTRUMENT_ID_CANDIDATES)
    if raw_symbol is None and instrument_id is None:
        missing.append("symbol_or_instrument_id")
    if raw_symbol is None and instrument_id is not None and (not instrument_id_to_symbol):
        missing.append("instrument_id_to_symbol_map")
    if missing:
        raise CmeSchemaError(
            f"required Databento mbp-1 columns missing/unresolvable: {missing}; input columns were: {sorted(column_names)}"
        )
    return CmeColumnBinding(
        ts_event=ts_event,
        ts_recv=ts_recv,
        action=action,
        side=side,
        price=price,
        size=size,
        raw_symbol=raw_symbol,
        instrument_id=instrument_id,
        bid_px=_first_present(column_names, BID_PX_LEVEL0_CANDIDATES),
        ask_px=_first_present(column_names, ASK_PX_LEVEL0_CANDIDATES),
        bid_sz=_first_present(column_names, BID_SZ_LEVEL0_CANDIDATES),
        ask_sz=_first_present(column_names, ASK_SZ_LEVEL0_CANDIDATES),
        bid_ct=_first_present(column_names, BID_CT_LEVEL0_CANDIDATES),
        ask_ct=_first_present(column_names, ASK_CT_LEVEL0_CANDIDATES),
    )


def to_ns_int64(column: pa.Array | pa.ChunkedArray, *, name: str) -> pa.Array:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    t = column.type
    if pa.types.is_integer(t):
        return column.cast(pa.int64())
    if pa.types.is_timestamp(t):
        tz = t.tz
        if tz is None:
            raise CmeSchemaError(
                f"{name}: timezone-naive timestamp column is ambiguous; Databento mbp-1 timestamps must be UTC-tagged or int64 ns"
            )
        if t.unit != "ns":
            column = column.cast(pa.timestamp("ns", tz=tz))
        return column.cast(pa.int64())
    raise CmeSchemaError(f"{name}: unsupported dtype {t}; expected int64 ns or timestamp with tz")


SIDE_BUY_AGGRESSOR = "B"
SIDE_SELL_AGGRESSOR = "A"
SIDE_UNKNOWN_VALUES = ("N", "", None)
AGGRESSOR_BUY = "BUY"
AGGRESSOR_SELL = "SELL"
AGGRESSOR_UNKNOWN = "UNKNOWN"


def map_aggressor(side_raw: object) -> str:
    if side_raw is None:
        return AGGRESSOR_UNKNOWN
    s = str(side_raw)
    if s == SIDE_BUY_AGGRESSOR:
        return AGGRESSOR_BUY
    if s == SIDE_SELL_AGGRESSOR:
        return AGGRESSOR_SELL
    return AGGRESSOR_UNKNOWN


def signed_size(side_raw: object, size: int | float | None) -> Optional[float]:
    if size is None:
        return None
    agg = map_aggressor(side_raw)
    if agg == AGGRESSOR_BUY:
        return float(size)
    if agg == AGGRESSOR_SELL:
        return -float(size)
    return None
