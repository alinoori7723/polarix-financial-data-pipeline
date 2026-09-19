from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_HEARTBEAT_MAX_GAP_MS = 1000
REASON_DUPLICATE = "duplicate_quote_no_state_change"


def scale_price(price: float, scale: int) -> int:
    if price is None:
        return 0
    return int(round(price * scale))


@dataclass
class FilteredEvent:
    symbol: str
    time_msc_raw: int
    recv_time_utc_ms: int
    monotonic_ns: int
    bid: float
    ask: float
    last: float
    bid_scaled: int
    ask_scaled: int
    last_scaled: int
    volume: int
    flags: int
    spread_points: int
    suppressed_count: int = 0
    first_suppressed_time_ms: int | None = None
    last_suppressed_time_ms: int | None = None
    suppressed_reason: str | None = None


@dataclass
class _SymbolState:
    last_bid_scaled: int | None = None
    last_ask_scaled: int | None = None
    last_last_scaled: int | None = None
    last_volume: int | None = None
    last_flags: int | None = None
    last_time_msc_raw: int | None = None
    suppressed_count: int = 0
    first_suppressed_time_ms: int | None = None
    last_suppressed_time_ms: int | None = None


@dataclass
class TickFilter:
    price_scale: int = 100
    stale_heartbeat_max_gap_ms: int = DEFAULT_HEARTBEAT_MAX_GAP_MS
    _state: dict[str, _SymbolState] = field(default_factory=dict)
    total_seen: int = 0
    total_emitted: int = 0
    total_suppressed: int = 0

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState()
            self._state[symbol] = st
        return st

    def consider(
        self,
        symbol: str,
        time_msc_raw: int,
        recv_time_utc_ms: int,
        monotonic_ns: int,
        bid: float,
        ask: float,
        last: float,
        volume: int,
        flags: int,
        spread_points: int,
    ) -> FilteredEvent | None:
        self.total_seen += 1
        st = self._state_for(symbol)
        bid_s = scale_price(bid, self.price_scale)
        ask_s = scale_price(ask, self.price_scale)
        last_s = scale_price(last, self.price_scale)
        state_changed = (
            st.last_bid_scaled != bid_s
            or st.last_ask_scaled != ask_s
            or st.last_last_scaled != last_s
            or (st.last_volume != volume)
            or (st.last_flags != flags)
        )
        time_advanced = (
            st.last_time_msc_raw is not None
            and time_msc_raw - st.last_time_msc_raw >= self.stale_heartbeat_max_gap_ms
        )
        first_ever = st.last_bid_scaled is None and st.last_ask_scaled is None
        should_emit = first_ever or state_changed or time_advanced
        if not should_emit:
            st.suppressed_count += 1
            self.total_suppressed += 1
            if st.first_suppressed_time_ms is None:
                st.first_suppressed_time_ms = recv_time_utc_ms
            st.last_suppressed_time_ms = recv_time_utc_ms
            return None
        event = FilteredEvent(
            symbol=symbol,
            time_msc_raw=time_msc_raw,
            recv_time_utc_ms=recv_time_utc_ms,
            monotonic_ns=monotonic_ns,
            bid=bid,
            ask=ask,
            last=last,
            bid_scaled=bid_s,
            ask_scaled=ask_s,
            last_scaled=last_s,
            volume=volume,
            flags=flags,
            spread_points=spread_points,
            suppressed_count=st.suppressed_count,
            first_suppressed_time_ms=st.first_suppressed_time_ms,
            last_suppressed_time_ms=st.last_suppressed_time_ms,
            suppressed_reason=REASON_DUPLICATE if st.suppressed_count > 0 else None,
        )
        st.last_bid_scaled = bid_s
        st.last_ask_scaled = ask_s
        st.last_last_scaled = last_s
        st.last_volume = volume
        st.last_flags = flags
        st.last_time_msc_raw = time_msc_raw
        st.suppressed_count = 0
        st.first_suppressed_time_ms = None
        st.last_suppressed_time_ms = None
        self.total_emitted += 1
        return event

    def metrics(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_emitted": self.total_emitted,
            "total_suppressed": self.total_suppressed,
            "symbols": {
                sym: {
                    "suppressed_count": st.suppressed_count,
                    "first_suppressed_time_ms": st.first_suppressed_time_ms,
                    "last_suppressed_time_ms": st.last_suppressed_time_ms,
                }
                for sym, st in self._state.items()
            },
        }
