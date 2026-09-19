from polarix.common.clock_health import (
    STATUS_COARSE_OK,
    STATUS_UNKNOWN,
    STATUS_UNSAFE,
    check_clock_health,
    classify_offset,
)


def test_classify_offset_safe():
    assert classify_offset(0, 50) == STATUS_COARSE_OK
    assert classify_offset(49.9, 50) == STATUS_COARSE_OK
    assert classify_offset(-50, 50) == STATUS_COARSE_OK


def test_classify_offset_unsafe():
    assert classify_offset(51, 50) == STATUS_UNSAFE
    assert classify_offset(-1000, 50) == STATUS_UNSAFE


def test_no_ntp_fails_safe():
    h = check_clock_health(
        threshold_ms=50, ntp_servers=("10.255.255.1", "10.255.255.2"), timeout=0.01
    )
    assert h.status in (STATUS_UNKNOWN, STATUS_UNSAFE)
    assert h.status != STATUS_COARSE_OK
