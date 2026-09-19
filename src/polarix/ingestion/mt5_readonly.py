from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import MetaTrader5 as mt5

    HAS_MT5 = True
except Exception:
    mt5 = None
    HAS_MT5 = False
FORBIDDEN_TRADING_SYMBOLS: tuple[str, ...] = (
    "order_send",
    "order_check",
    "order_calc_margin",
    "order_calc_profit",
    "positions_close",
    "position_close",
    "position_modify",
    "trade_request",
)


class KillSwitchTripped(RuntimeError):
    pass


class MT5UnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalSnapshot:
    name: str
    company: str
    path: str
    build: int
    trade_allowed: bool
    dlls_allowed: bool


@dataclass(frozen=True)
class AccountSnapshot:
    login_hash: str
    server: str
    company: str
    currency: str
    leverage: int
    trade_allowed: bool
    trade_expert: bool


@dataclass(frozen=True)
class SymbolSnapshot:
    name: str
    description: str
    digits: int
    point: float
    trade_mode: int
    spread: int
    spread_float: bool


@dataclass(frozen=True)
class TickSnapshot:
    symbol: str
    time: int
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


def _mask_login(login: int | str | None) -> str:
    if login is None:
        return "UNKNOWN"
    digest = hashlib.sha256(f"polarix-login::{login}".encode("utf-8")).hexdigest()
    return digest[:16]


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


class ReadOnlyMT5:
    def __init__(self, fail_if_terminal_trade_allowed: bool = True) -> None:
        self._initialized = False
        self._fail_if_terminal_trade_allowed = fail_if_terminal_trade_allowed

    def initialize(self) -> None:
        if not HAS_MT5:
            raise MT5UnavailableError("MetaTrader5 module not importable; install on Windows host")
        if not mt5.initialize():
            err = mt5.last_error()
            raise MT5UnavailableError(f"mt5.initialize() failed: {err}")
        self._initialized = True

    def shutdown(self) -> None:
        if HAS_MT5 and self._initialized:
            try:
                mt5.shutdown()
            finally:
                self._initialized = False

    def check_kill_switch(self) -> None:
        if not self._fail_if_terminal_trade_allowed:
            return
        ti = self.terminal_info()
        if ti.trade_allowed:
            raise KillSwitchTripped(
                "terminal_info.trade_allowed is True; refusing to operate while MT5 AutoTrading is enabled"
            )

    def terminal_info(self) -> TerminalSnapshot:
        info = mt5.terminal_info()
        if info is None:
            raise MT5UnavailableError("terminal_info() returned None")
        return TerminalSnapshot(
            name=str(_attr(info, "name", "")),
            company=str(_attr(info, "company", "")),
            path=str(_attr(info, "path", "")),
            build=int(_attr(info, "build", 0) or 0),
            trade_allowed=bool(_attr(info, "trade_allowed", False)),
            dlls_allowed=bool(_attr(info, "dlls_allowed", False)),
        )

    def account_info(self) -> AccountSnapshot:
        info = mt5.account_info()
        if info is None:
            raise MT5UnavailableError("account_info() returned None")
        return AccountSnapshot(
            login_hash=_mask_login(_attr(info, "login")),
            server=str(_attr(info, "server", "")),
            company=str(_attr(info, "company", "")),
            currency=str(_attr(info, "currency", "")),
            leverage=int(_attr(info, "leverage", 0) or 0),
            trade_allowed=bool(_attr(info, "trade_allowed", False)),
            trade_expert=bool(_attr(info, "trade_expert", False)),
        )

    def select_symbols(self, symbols: Iterable[str]) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for s in symbols:
            ok = bool(mt5.symbol_select(s, True))
            out[s] = ok
        return out

    def symbol_info(self, symbol: str) -> SymbolSnapshot | None:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return SymbolSnapshot(
            name=str(_attr(info, "name", symbol)),
            description=str(_attr(info, "description", "")),
            digits=int(_attr(info, "digits", 0) or 0),
            point=float(_attr(info, "point", 0.0) or 0.0),
            trade_mode=int(_attr(info, "trade_mode", 0) or 0),
            spread=int(_attr(info, "spread", 0) or 0),
            spread_float=bool(_attr(info, "spread_float", False)),
        )

    def symbol_info_tick(self, symbol: str) -> TickSnapshot | None:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return TickSnapshot(
            symbol=symbol,
            time=int(_attr(tick, "time", 0) or 0),
            time_msc=int(_attr(tick, "time_msc", 0) or 0),
            bid=float(_attr(tick, "bid", 0.0) or 0.0),
            ask=float(_attr(tick, "ask", 0.0) or 0.0),
            last=float(_attr(tick, "last", 0.0) or 0.0),
            volume=int(_attr(tick, "volume", 0) or 0),
            flags=int(_attr(tick, "flags", 0) or 0),
        )

    def now_recv_ms(self) -> int:
        return int(time.time() * 1000)

    def now_monotonic_ns(self) -> int:
        return time.monotonic_ns()
