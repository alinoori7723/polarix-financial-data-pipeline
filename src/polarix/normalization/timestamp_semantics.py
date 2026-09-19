from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable

STATUS_PENDING = "TIMESTAMP_SEMANTICS_PENDING"
STATUS_OFFSET_VERIFIED = "OFFSET_VERIFIED_FOR_SESSION"
STATUS_UTC_VERIFIED = "UTC_EPOCH_VERIFIED"
STATUS_UNSAFE_FOR_JOIN = "CALIBRATION_UNSAFE_FOR_JOIN"
EVENT_DST_SHIFT = "DST_SHIFT_DETECTED"
EVENT_OFFSET_CHANGE = "OFFSET_CHANGE_DETECTED"
EVENT_DOWNGRADED = "DOWNGRADED_AFTER_STALE_GAP"
EVENT_MARKED_UNSAFE = "MARKED_UNSAFE_FOR_JOIN"
P50_MAX_MS = 1000
P95_MAX_MS = 5000
MIN_DISTINCT_TIME_MSC = 2
DOWNGRADE_AFTER_STALE_MS_DEFAULT = 30000
UNSAFE_AFTER_STALE_MS_DEFAULT = 5 * 60000


@dataclass(frozen=True)
class FreshnessGate:
    max_future_jitter_ms: int
    max_live_tick_age_ms: int

    def age_is_fresh(self, age_ms: float) -> bool:
        return age_ms >= -self.max_future_jitter_ms and age_ms <= self.max_live_tick_age_ms


@dataclass
class _CandidateState:
    offset_min: int
    fresh_count: int = 0
    distinct_time_msc: set = field(default_factory=set)
    residuals_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=1024))


def _percentile(values: Iterable[float], pct: float) -> float:
    arr = sorted(values)
    if not arr:
        return math.inf
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] * (c - k) + arr[c] * (k - f)


@dataclass
class TimestampSemantics:
    plausible_offsets_minutes: tuple[int, ...]
    gate: FreshnessGate
    required_fresh_ticks: int
    cached_offset_min: int | None = None
    status: str = STATUS_PENDING
    verified_offset_min: int | None = None
    downgrade_after_stale_ms: int = DOWNGRADE_AFTER_STALE_MS_DEFAULT
    unsafe_after_stale_ms: int = UNSAFE_AFTER_STALE_MS_DEFAULT
    candidates: dict[int, _CandidateState] = field(default_factory=dict, init=False)
    last_event: str | None = None
    first_observation_recv_ms: int | None = field(default=None, init=False)
    last_fresh_observation_recv_ms: int | None = field(default=None, init=False)
    last_recv_time_utc_ms: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._reset_candidates()

    def _reset_candidates(self) -> None:
        self.candidates = {o: _CandidateState(offset_min=o) for o in self.plausible_offsets_minutes}

    def reset_for_reconnect(self) -> None:
        prev_verified = self.verified_offset_min
        self.status = STATUS_PENDING
        self.verified_offset_min = None
        self.last_event = None
        self.first_observation_recv_ms = None
        self.last_fresh_observation_recv_ms = None
        self._reset_candidates()
        if prev_verified is not None:
            self.cached_offset_min = prev_verified

    def observe(self, time_msc_raw: int, recv_time_utc_ms: int) -> dict:
        if self.first_observation_recv_ms is None:
            self.first_observation_recv_ms = recv_time_utc_ms
        self.last_recv_time_utc_ms = recv_time_utc_ms
        fresh_offsets: list[int] = []
        for off_min, state in self.candidates.items():
            time_msc_utc_ms = time_msc_raw - off_min * 60000
            age_ms = float(recv_time_utc_ms - time_msc_utc_ms)
            if not self.gate.age_is_fresh(age_ms):
                continue
            fresh_offsets.append(off_min)
            state.fresh_count += 1
            state.distinct_time_msc.add(time_msc_raw)
            state.residuals_ms.append(age_ms)
        if fresh_offsets:
            self.last_fresh_observation_recv_ms = recv_time_utc_ms
            if self.status == STATUS_UNSAFE_FOR_JOIN:
                self.status = STATUS_PENDING
        event = self._maybe_verify()
        return {"status": self.status, "event": event, "fresh_offsets": fresh_offsets}

    def evaluate_staleness(self, now_recv_ms: int) -> str | None:
        if self.status in (STATUS_OFFSET_VERIFIED, STATUS_UTC_VERIFIED):
            last_fresh = self.last_fresh_observation_recv_ms
            if last_fresh is None:
                gap = self.downgrade_after_stale_ms + 1
            else:
                gap = now_recv_ms - last_fresh
            if gap >= self.downgrade_after_stale_ms:
                self.cached_offset_min = self.verified_offset_min
                self.verified_offset_min = None
                self.status = STATUS_PENDING
                self._reset_candidates()
                self.last_event = EVENT_DOWNGRADED
                return EVENT_DOWNGRADED
        if self.status == STATUS_PENDING:
            ref = (
                self.last_fresh_observation_recv_ms
                if self.last_fresh_observation_recv_ms is not None
                else self.first_observation_recv_ms
            )
            if ref is not None and now_recv_ms - ref >= self.unsafe_after_stale_ms:
                self.status = STATUS_UNSAFE_FOR_JOIN
                self.last_event = EVENT_MARKED_UNSAFE
                return EVENT_MARKED_UNSAFE
        return None

    def _maybe_verify(self) -> str | None:
        if self.status != STATUS_PENDING:
            return None
        eligible: list[_CandidateState] = []
        for state in self.candidates.values():
            if state.fresh_count < self.required_fresh_ticks:
                continue
            if len(state.distinct_time_msc) < MIN_DISTINCT_TIME_MSC:
                continue
            p50 = _percentile(state.residuals_ms, 0.5)
            p95 = _percentile(state.residuals_ms, 0.95)
            if p50 <= P50_MAX_MS and p95 <= P95_MAX_MS:
                eligible.append(state)
        if not eligible:
            return None
        chosen = min(eligible, key=lambda s: abs(_percentile(s.residuals_ms, 0.5)))
        self.verified_offset_min = chosen.offset_min
        self.status = STATUS_UTC_VERIFIED if chosen.offset_min == 0 else STATUS_OFFSET_VERIFIED
        event: str | None = None
        if self.cached_offset_min is not None and self.cached_offset_min != chosen.offset_min:
            delta = abs(chosen.offset_min - self.cached_offset_min)
            event = EVENT_DST_SHIFT if delta == 60 else EVENT_OFFSET_CHANGE
            self.last_event = event
        return event

    def manifest(self) -> dict:
        return {
            "status": self.status,
            "verified_offset_min": self.verified_offset_min,
            "cached_offset_hint_min": self.cached_offset_min,
            "plausible_offsets_minutes": list(self.plausible_offsets_minutes),
            "required_fresh_ticks": self.required_fresh_ticks,
            "downgrade_after_stale_ms": self.downgrade_after_stale_ms,
            "unsafe_after_stale_ms": self.unsafe_after_stale_ms,
            "last_event": self.last_event,
            "last_fresh_observation_recv_ms": self.last_fresh_observation_recv_ms,
        }

    def is_join_safe(self) -> bool:
        return self.status in (STATUS_OFFSET_VERIFIED, STATUS_UTC_VERIFIED)
