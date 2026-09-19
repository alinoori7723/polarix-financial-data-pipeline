from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_ALIGNMENT_TOLERANCE_MS = 50
MS_TO_NS = 1000000


@dataclass(frozen=True)
class AlignmentResult:
    cme_row_index: int
    cme_event_time_utc_ns: int
    matched_mt5_time_msc_utc_ms: Optional[int]
    matched_mt5_mid: Optional[float]
    matched_mt5_spread_price: Optional[float]
    alignment_delta_ms: Optional[float]
    is_aligned: bool


@dataclass
class MT5QuoteSlice:
    time_msc_utc_ms: Sequence[int]
    mid: Sequence[Optional[float]]
    spread_price: Sequence[Optional[float]]
    is_join_safe: Sequence[bool]

    def __post_init__(self) -> None:
        n = len(self.time_msc_utc_ms)
        if not (
            len(self.mid) == n and len(self.spread_price) == n and (len(self.is_join_safe) == n)
        ):
            raise ValueError("MT5QuoteSlice arrays must all have the same length")
        prev = None
        for t in self.time_msc_utc_ms:
            if prev is not None and t < prev:
                raise ValueError("MT5QuoteSlice.time_msc_utc_ms must be non-decreasing")
            prev = t

    def __len__(self) -> int:
        return len(self.time_msc_utc_ms)


def align_one(
    cme_event_time_utc_ns: int,
    quotes: MT5QuoteSlice,
    *,
    alignment_tolerance_ms: int = DEFAULT_ALIGNMENT_TOLERANCE_MS,
    cme_row_index: int = 0,
) -> AlignmentResult:
    if alignment_tolerance_ms <= 0:
        raise ValueError("alignment_tolerance_ms must be > 0")
    if cme_event_time_utc_ns < 0:
        raise ValueError("cme_event_time_utc_ns must be non-negative")
    cme_event_ms = cme_event_time_utc_ns // MS_TO_NS
    times = quotes.time_msc_utc_ms
    upper = bisect_right(times, cme_event_ms)
    for i in range(upper - 1, -1, -1):
        t = times[i]
        delta_ms = cme_event_ms - t
        if delta_ms > alignment_tolerance_ms:
            break
        if not quotes.is_join_safe[i]:
            continue
        return AlignmentResult(
            cme_row_index=cme_row_index,
            cme_event_time_utc_ns=cme_event_time_utc_ns,
            matched_mt5_time_msc_utc_ms=int(t),
            matched_mt5_mid=quotes.mid[i],
            matched_mt5_spread_price=quotes.spread_price[i],
            alignment_delta_ms=float(delta_ms),
            is_aligned=True,
        )
    return AlignmentResult(
        cme_row_index=cme_row_index,
        cme_event_time_utc_ns=cme_event_time_utc_ns,
        matched_mt5_time_msc_utc_ms=None,
        matched_mt5_mid=None,
        matched_mt5_spread_price=None,
        alignment_delta_ms=None,
        is_aligned=False,
    )


def align_many(
    cme_event_times_utc_ns: Sequence[int],
    quotes: MT5QuoteSlice,
    *,
    alignment_tolerance_ms: int = DEFAULT_ALIGNMENT_TOLERANCE_MS,
) -> list[AlignmentResult]:
    return [
        align_one(ev, quotes, alignment_tolerance_ms=alignment_tolerance_ms, cme_row_index=i)
        for i, ev in enumerate(cme_event_times_utc_ns)
    ]
