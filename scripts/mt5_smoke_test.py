import MetaTrader5 as mt5

SYMBOLS = ["SPX500", "NDX100"]


def main() -> int:
    print("Starting MT5 smoke test...")
    if not mt5.initialize():
        error = mt5.last_error()
        error_code = (
            error[0]
            if isinstance(error, tuple) and error and isinstance(error[0], int)
            else "unavailable"
        )
        print(f"MT5 initialize failed (code: {error_code}).")
        print(
            "Make sure MetaTrader 5 is installed, open, and logged into the sandbox/research account."
        )
        return 1
    try:
        terminal = mt5.terminal_info()
        if terminal is None:
            print("ERROR: Terminal information unavailable.")
            return 1
        print("Terminal: available")
        account = mt5.account_info()
        if account is None:
            print("\nERROR: No MT5 account info returned. Are you logged in?")
            return 1
        print("Account: available")
        print("\n=== SYMBOL CHECK ===")
        for symbol in SYMBOLS:
            selected = mt5.symbol_select(symbol, True)
            print(f"\nSYMBOL: {symbol}")
            print("selected:", selected)
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if info is None:
                print("INFO: None")
            else:
                print("description:", info.description)
                print("digits:", info.digits)
                print("point:", info.point)
                print("trade_contract_size:", info.trade_contract_size)
                print("volume_min:", info.volume_min)
                print("volume_max:", info.volume_max)
                print("volume_step:", info.volume_step)
                print("trade_mode:", info.trade_mode)
                print("trade_calc_mode:", info.trade_calc_mode)
                print("filling_mode:", info.filling_mode)
                print("spread:", info.spread)
                print("spread_float:", info.spread_float)
            if tick is None:
                print("TICK: None")
            else:
                print("tick.time:", tick.time)
                print("tick.time_msc:", tick.time_msc)
                print("tick.bid:", tick.bid)
                print("tick.ask:", tick.ask)
                print("tick.last:", tick.last)
                print("tick.volume:", tick.volume)
                print("tick.flags:", tick.flags)
        print("\nMT5 smoke test completed.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
