from __future__ import annotations

from polarix.normalization.timestamp_semantics import (
    EVENT_DST_SHIFT,
    EVENT_OFFSET_CHANGE,
    STATUS_OFFSET_VERIFIED,
    STATUS_PENDING,
    STATUS_UTC_VERIFIED,
    FreshnessGate,
    TimestampSemantics,
)


def _make(
    *,
    required_fresh: int = 20,
    cached: int | None = None,
    offsets: tuple[int, ...] = (-300, -240, 0, 60, 120, 180),
) -> TimestampSemantics:
    return TimestampSemantics(
        plausible_offsets_minutes=offsets,
        gate=FreshnessGate(max_future_jitter_ms=250, max_live_tick_age_ms=5000),
        required_fresh_ticks=required_fresh,
        cached_offset_min=cached,
    )


def _drive_fresh_ticks(
    ts: TimestampSemantics,
    true_offset_min: int,
    n: int,
    base_recv_ms: int = 1700000000000,
    step_ms: int = 200,
) -> None:
    for i in range(n):
        recv = base_recv_ms + i * step_ms
        time_msc = recv + true_offset_min * 60000 + 50
        ts.observe(time_msc_raw=time_msc, recv_time_utc_ms=recv)


def test_starts_pending():
    ts = _make()
    assert ts.status == STATUS_PENDING


def test_stale_ticks_do_not_verify():
    ts = _make()
    base_recv = 1700000000000
    for i in range(200):
        recv = base_recv + i * 50
        time_msc = recv - 3600000
        ts.observe(time_msc_raw=time_msc, recv_time_utc_ms=recv)
    assert ts.status == STATUS_PENDING
    assert ts.verified_offset_min is None


def test_verifies_zero_offset_as_utc():
    ts = _make(required_fresh=20)
    _drive_fresh_ticks(ts, true_offset_min=0, n=30)
    assert ts.status == STATUS_UTC_VERIFIED
    assert ts.verified_offset_min == 0


def test_verifies_nonzero_plausible_offset():
    ts = _make(required_fresh=20)
    _drive_fresh_ticks(ts, true_offset_min=180, n=30)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 180


def test_rejects_arbitrary_offset_outside_config():
    ts = _make(offsets=(0, 120))
    _drive_fresh_ticks(ts, true_offset_min=45, n=200)
    assert ts.status == STATUS_PENDING


def test_cached_offset_alone_does_not_verify():
    ts = _make(cached=120)
    assert ts.status == STATUS_PENDING
    assert ts.verified_offset_min is None


def test_reconnect_reevaluates_all_offsets():
    ts = _make(required_fresh=20)
    _drive_fresh_ticks(ts, true_offset_min=120, n=30)
    assert ts.status == STATUS_OFFSET_VERIFIED
    assert ts.verified_offset_min == 120
    ts.reset_for_reconnect()
    assert ts.status == STATUS_PENDING
    assert ts.verified_offset_min is None
    assert ts.cached_offset_min == 120


def test_dst_shift_event_emitted_after_cache_change():
    ts = _make(cached=120, required_fresh=20)
    _drive_fresh_ticks(ts, true_offset_min=180, n=30)
    assert ts.last_event == EVENT_DST_SHIFT


def test_offset_change_event_emitted_for_large_jump():
    ts = _make(cached=0, required_fresh=20)
    _drive_fresh_ticks(ts, true_offset_min=180, n=30)
    assert ts.last_event == EVENT_OFFSET_CHANGE


def test_join_safety_blocks_until_verified():
    ts = _make(required_fresh=20)
    assert not ts.is_join_safe()
    _drive_fresh_ticks(ts, true_offset_min=0, n=30)
    assert ts.is_join_safe()
