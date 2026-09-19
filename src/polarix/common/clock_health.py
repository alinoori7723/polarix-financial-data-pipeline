from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Iterable

STATUS_UNSAFE = "CALIBRATION_UNSAFE"
STATUS_COARSE_OK = "CALIBRATION_COARSE_OK"
STATUS_UNKNOWN = "CALIBRATION_UNKNOWN"
_NTP_DELTA = 2208988800
DEFAULT_NTP_SERVERS: tuple[str, ...] = ("time.windows.com", "time.google.com", "pool.ntp.org")


@dataclass(frozen=True)
class ClockHealth:
    status: str
    offset_ms: float | None
    threshold_ms: int
    source: str
    error: str | None = None

    @property
    def safe_for_join(self) -> bool:
        return self.status == STATUS_COARSE_OK


def _query_ntp(server: str, timeout: float = 2.0) -> float:
    packet = b"\x1b" + 47 * b"\x00"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t1 = time.time()
        s.sendto(packet, (server, 123))
        data, _ = s.recvfrom(48)
        t4 = time.time()
    secs, frac = struct.unpack("!II", data[40:48])
    server_tx = secs - _NTP_DELTA + frac / 2**32
    estimated_host_at_server_tx = (t1 + t4) / 2.0
    offset_seconds = estimated_host_at_server_tx - server_tx
    return offset_seconds * 1000.0


def check_clock_health(
    threshold_ms: int, ntp_servers: Iterable[str] = DEFAULT_NTP_SERVERS, timeout: float = 2.0
) -> ClockHealth:
    last_error: str | None = None
    for server in ntp_servers:
        try:
            offset_ms = _query_ntp(server, timeout=timeout)
        except Exception as exc:
            last_error = f"{server}: {exc!r}"
            continue
        abs_off = abs(offset_ms)
        status = STATUS_COARSE_OK if abs_off <= threshold_ms else STATUS_UNSAFE
        return ClockHealth(
            status=status, offset_ms=offset_ms, threshold_ms=threshold_ms, source=server
        )
    return ClockHealth(
        status=STATUS_UNKNOWN,
        offset_ms=None,
        threshold_ms=threshold_ms,
        source="ntp",
        error=last_error,
    )


def classify_offset(offset_ms: float, threshold_ms: int) -> str:
    return STATUS_COARSE_OK if abs(offset_ms) <= threshold_ms else STATUS_UNSAFE
