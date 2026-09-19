from __future__ import annotations

from polarix.ingestion.tick_filter import REASON_DUPLICATE, TickFilter


def _consider(tf: TickFilter, **kwargs):
    return tf.consider(
        symbol=kwargs.get("symbol", "SPX500"),
        time_msc_raw=kwargs.get("time_msc_raw", 1000),
        recv_time_utc_ms=kwargs.get("recv_time_utc_ms", 1000),
        monotonic_ns=kwargs.get("monotonic_ns", 1000),
        bid=kwargs.get("bid", 5000.0),
        ask=kwargs.get("ask", 5000.1),
        last=kwargs.get("last", 5000.0),
        volume=kwargs.get("volume", 1),
        flags=kwargs.get("flags", 2),
        spread_points=kwargs.get("spread_points", 1),
    )


def test_first_tick_emits():
    tf = TickFilter()
    ev = _consider(tf)
    assert ev is not None
    assert ev.suppressed_count == 0


def test_duplicate_quotes_are_suppressed_and_counted():
    tf = TickFilter(stale_heartbeat_max_gap_ms=10000)
    first = _consider(tf, recv_time_utc_ms=1000, time_msc_raw=1000)
    assert first is not None
    for i in range(99):
        dup = _consider(tf, recv_time_utc_ms=1001 + i, time_msc_raw=1000)
        assert dup is None
    next_change = _consider(tf, bid=5000.5, time_msc_raw=1500, recv_time_utc_ms=2500)
    assert next_change is not None
    assert next_change.suppressed_count == 99
    assert next_change.suppressed_reason == REASON_DUPLICATE
    assert next_change.first_suppressed_time_ms is not None
    assert next_change.last_suppressed_time_ms is not None


def test_state_change_on_price():
    tf = TickFilter()
    _consider(tf, bid=5000.0)
    ev = _consider(tf, bid=5000.5)
    assert ev is not None


def test_state_change_on_flags():
    tf = TickFilter()
    _consider(tf, flags=2)
    ev = _consider(tf, flags=6)
    assert ev is not None


def test_heartbeat_emits_on_time_gap():
    tf = TickFilter(stale_heartbeat_max_gap_ms=1000)
    _consider(tf, time_msc_raw=1000)
    ev = _consider(tf, time_msc_raw=2500)
    assert ev is not None


def test_metrics_account_for_all_ticks():
    tf = TickFilter(stale_heartbeat_max_gap_ms=10000)
    _consider(tf, time_msc_raw=1)
    for _ in range(10):
        _consider(tf, time_msc_raw=1)
    _consider(tf, bid=5001, time_msc_raw=2)
    m = tf.metrics()
    assert m["total_seen"] == 12
    assert m["total_emitted"] == 2
    assert m["total_suppressed"] == 10
