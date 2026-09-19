from __future__ import annotations

from polarix.normalization.timestamp_semantics import (
    EVENT_DOWNGRADED,
    EVENT_DST_SHIFT,
    EVENT_MARKED_UNSAFE,
    EVENT_OFFSET_CHANGE,
    STATUS_OFFSET_VERIFIED,
    STATUS_PENDING,
    STATUS_UNSAFE_FOR_JOIN,
    STATUS_UTC_VERIFIED,
    FreshnessGate,
    TimestampSemantics,
)


def _make(
    *,
    required_fresh: int = 20,
    cached: int | None = None,
    offsets: tuple[int, ...] = (-300, -240, 0, 60, 120, 180),
    downgrade_ms: int = 30000,
    unsafe_ms: int = 5 * 60000,
) -> TimestampSemantics:
    return TimestampSemantics(
        plausible_offsets_minutes=offsets,
        gate=FreshnessGate(max_future_jitter_ms=250, max_live_tick_age_ms=5000),
        required_fresh_ticks=required_fresh,
        cached_offset_min=cached,
        downgrade_after_stale_ms=downgrade_ms,
        unsafe_after_stale_ms=unsafe_ms,
    )


def _drive_fresh(
    ts: TimestampSemantics,
    true_offset_min: int,
    n: int,
    start_recv_ms: int = 1700000000000,
    step_ms: int = 200,
) -> int:
    last = start_recv_ms
    for i in range(n):
        last = start_recv_ms + i * step_ms
        time_msc = last + true_offset_min * 60000 + 50
        ts.observe(time_msc_raw=time_msc, recv_time_utc_ms=last)
    return last


def test_full_arc_verify_downgrade_reverify():
    ts = _make(required_fresh=20, downgrade_ms=30000)
    last_recv = _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=1000000)
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.verified_offset_min == 0
    assert ts.is_join_safe()
    now_recv = last_recv + 60000
    event = ts.evaluate_staleness(now_recv_ms=now_recv)
    assert event == EVENT_DOWNGRADED
    assert ts.status == STATUS_PENDING
    assert ts.cached_offset_min == 0
    assert ts.verified_offset_min is None
    assert not ts.is_join_safe()
    _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=now_recv + 1000)
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.verified_offset_min == 0
    assert ts.is_join_safe()


def test_downgrade_does_not_fire_below_threshold():
    ts = _make(required_fresh=20, downgrade_ms=30000)
    last_recv = _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=1000000)
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.evaluate_staleness(now_recv_ms=last_recv + 10000) is None
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.is_join_safe()


def test_cached_offset_only_hint_after_downgrade():
    ts = _make(required_fresh=20, downgrade_ms=30000)
    last = _drive_fresh(ts, true_offset_min=180, n=30, start_recv_ms=1000000)
    assert ts.verified_offset_min == 180
    ts.evaluate_staleness(now_recv_ms=last + 60000)
    assert ts.cached_offset_min == 180
    assert ts.verified_offset_min is None
    assert ts.status == STATUS_PENDING
    assert not ts.is_join_safe()


def test_reverify_after_downgrade_uses_new_offset_and_emits_event():
    ts = _make(required_fresh=20, downgrade_ms=30000)
    last = _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=1000000)
    ts.evaluate_staleness(now_recv_ms=last + 60000)
    assert ts.status == STATUS_PENDING
    assert ts.cached_offset_min == 0
    _drive_fresh(ts, true_offset_min=180, n=30, start_recv_ms=last + 120000)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 180
    assert ts.last_event == EVENT_OFFSET_CHANGE


def test_reverify_dst_shift_after_downgrade_emits_dst_event():
    ts = _make(required_fresh=20, downgrade_ms=30000)
    last = _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=1000000)
    ts.evaluate_staleness(now_recv_ms=last + 60000)
    _drive_fresh(ts, true_offset_min=60, n=30, start_recv_ms=last + 120000)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 60
    assert ts.last_event == EVENT_DST_SHIFT


def test_cached_offset_fails_other_succeeds_at_first_verification():
    ts = _make(required_fresh=20, cached=0)
    _drive_fresh(ts, true_offset_min=180, n=30, start_recv_ms=1000000)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 180
    assert ts.last_event == EVENT_OFFSET_CHANGE


def test_no_plausible_offset_matches_marks_unsafe():
    ts = _make(required_fresh=20, unsafe_ms=300000)
    base_recv = 1000000
    for i in range(200):
        recv = base_recv + i * 50
        ts.observe(time_msc_raw=recv - 3600000, recv_time_utc_ms=recv)
    assert ts.status == STATUS_PENDING
    assert ts.last_fresh_observation_recv_ms is None
    event = ts.evaluate_staleness(now_recv_ms=base_recv + 400000)
    assert event == EVENT_MARKED_UNSAFE
    assert ts.status == STATUS_UNSAFE_FOR_JOIN
    assert not ts.is_join_safe()
    assert ts.verified_offset_min is None


def test_unsafe_recovers_when_fresh_ticks_return():
    ts = _make(required_fresh=20, unsafe_ms=300000)
    base = 1000000
    for i in range(200):
        recv = base + i * 50
        ts.observe(time_msc_raw=recv - 3600000, recv_time_utc_ms=recv)
    ts.evaluate_staleness(now_recv_ms=base + 400000)
    assert ts.status == STATUS_UNSAFE_FOR_JOIN
    _drive_fresh(ts, true_offset_min=0, n=30, start_recv_ms=base + 500000)
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.is_join_safe()


def test_reset_for_reconnect_downgrades_and_keeps_hint():
    ts = _make(required_fresh=20)
    _drive_fresh(ts, true_offset_min=120, n=30, start_recv_ms=1000000)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 120
    ts.reset_for_reconnect()
    assert ts.status == STATUS_PENDING
    assert ts.verified_offset_min is None
    assert ts.cached_offset_min == 120
    assert not ts.is_join_safe()
    assert ts.last_fresh_observation_recv_ms is None
    assert ts.first_observation_recv_ms is None


def test_join_safety_blocked_during_pending_and_unsafe():
    ts = _make(required_fresh=20, unsafe_ms=300000)
    assert ts.status == STATUS_PENDING
    assert not ts.is_join_safe()
    base = 1000000
    for i in range(50):
        recv = base + i * 50
        ts.observe(time_msc_raw=recv - 3600000, recv_time_utc_ms=recv)
    ts.evaluate_staleness(now_recv_ms=base + 400000)
    assert ts.status == STATUS_UNSAFE_FOR_JOIN
    assert not ts.is_join_safe()
