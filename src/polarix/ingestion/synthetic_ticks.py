from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SyntheticTick:
    symbol: str
    time_msc_raw: int
    recv_time_utc_ms: int
    monotonic_ns: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


def _digits_for(symbol: str) -> int:
    return 1


@dataclass
class TickStream:
    symbols: tuple[str, ...]
    rate_per_second_total: int
    duration_seconds: float
    start_recv_time_utc_ms: int
    broker_offset_minutes: int = 0
    seed: int = 12345
    spread_points_min: int = 1
    spread_points_max: int = 4

    def normal(self) -> Iterator[SyntheticTick]:
        rng = random.Random(self.seed)
        per_symbol_rate = max(1, self.rate_per_second_total // max(1, len(self.symbols)))
        total_ticks_per_symbol = int(per_symbol_rate * self.duration_seconds)
        if total_ticks_per_symbol == 0:
            return
        interval_ms = 1000.0 / per_symbol_rate
        price_state = {s: 5000.0 + 100 * i for i, s in enumerate(self.symbols)}
        mono = 1000000000
        for n in range(total_ticks_per_symbol):
            for symbol in self.symbols:
                price_state[symbol] += rng.gauss(0.0, 0.5)
                mid = price_state[symbol]
                spread_pts = rng.randint(self.spread_points_min, self.spread_points_max)
                bid = round(mid - spread_pts * 0.05, _digits_for(symbol))
                ask = round(mid + spread_pts * 0.05, _digits_for(symbol))
                last = round(mid, _digits_for(symbol))
                recv_ms = int(self.start_recv_time_utc_ms + n * interval_ms)
                time_msc_raw = recv_ms + self.broker_offset_minutes * 60000 + rng.randint(-10, 10)
                mono += int(interval_ms * 1000000)
                yield SyntheticTick(
                    symbol=symbol,
                    time_msc_raw=time_msc_raw,
                    recv_time_utc_ms=recv_ms,
                    monotonic_ns=mono,
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume=rng.randint(1, 5),
                    flags=2,
                )

    def duplicate_burst(self, n: int = 200) -> Iterator[SyntheticTick]:
        symbol = self.symbols[0]
        recv_ms = self.start_recv_time_utc_ms
        time_msc_raw = recv_ms + self.broker_offset_minutes * 60000
        mono = 2000000000
        bid, ask, last = (5000.0, 5000.1, 5000.0)
        for _ in range(n):
            yield SyntheticTick(
                symbol=symbol,
                time_msc_raw=time_msc_raw,
                recv_time_utc_ms=recv_ms,
                monotonic_ns=mono,
                bid=bid,
                ask=ask,
                last=last,
                volume=1,
                flags=2,
            )

    def stale_burst(self, n: int = 100, age_ms: int = 30000) -> Iterator[SyntheticTick]:
        symbol = self.symbols[0]
        rng = random.Random(self.seed + 1)
        recv_ms = self.start_recv_time_utc_ms
        mono = 3000000000
        for i in range(n):
            recv_now = recv_ms + i * 10
            time_msc_raw = recv_now + self.broker_offset_minutes * 60000 - age_ms
            yield SyntheticTick(
                symbol=symbol,
                time_msc_raw=time_msc_raw,
                recv_time_utc_ms=recv_now,
                monotonic_ns=mono + i * 1000000,
                bid=4999.5 + rng.uniform(-0.1, 0.1),
                ask=5000.0 + rng.uniform(-0.1, 0.1),
                last=4999.8,
                volume=1,
                flags=2,
            )

    def reconnect_gap(self, gap_seconds: int = 30) -> Iterator[SyntheticTick]:
        before = list(self.normal())
        if not before:
            return
        last_recv = before[-1].recv_time_utc_ms
        yield from before
        gap_ms = gap_seconds * 1000
        shifted = TickStream(
            symbols=self.symbols,
            rate_per_second_total=self.rate_per_second_total,
            duration_seconds=self.duration_seconds,
            start_recv_time_utc_ms=last_recv + gap_ms,
            broker_offset_minutes=self.broker_offset_minutes,
            seed=self.seed + 7,
            spread_points_min=self.spread_points_min,
            spread_points_max=self.spread_points_max,
        )
        yield from shifted.normal()

    def monday_open(self, stale_seconds: int = 5) -> Iterator[SyntheticTick]:
        yield from self.stale_burst(
            n=stale_seconds * max(1, self.rate_per_second_total // len(self.symbols)), age_ms=120000
        )
        shifted = TickStream(
            symbols=self.symbols,
            rate_per_second_total=self.rate_per_second_total,
            duration_seconds=self.duration_seconds,
            start_recv_time_utc_ms=self.start_recv_time_utc_ms + stale_seconds * 1000,
            broker_offset_minutes=self.broker_offset_minutes,
            seed=self.seed + 11,
        )
        yield from shifted.normal()


def total_ticks(it: Iterable[SyntheticTick]) -> int:
    return sum((1 for _ in it))
