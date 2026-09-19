from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from polarix.ingestion.parquet_writer import PARQUET_SCHEMA
from polarix.orchestration.artifacts import publish_json

FIXTURE_DATE = "2026-01-05"
FIXTURE_OFFSET_MINUTES = 120
PAIRS = (("ES", "SPX500", 5000.0), ("NQ", "NDX100", 18000.0))
SCENARIOS = ("normal", "invalid-timestamps", "crossed-quotes")


@dataclass(frozen=True)
class FixtureConfig:
    seed: int = 42
    seconds: int = 1200
    scenario: str = "normal"

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if not 120 <= self.seconds <= 3600 or self.seconds % 60:
            raise ValueError("seconds must be a multiple of 60 between 120 and 3600")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")

    @property
    def start_ms(self) -> int:
        start = dt.datetime.fromisoformat(f"{FIXTURE_DATE}T12:00:00+00:00")
        return int(start.timestamp() * 1000)

    @property
    def end_ns(self) -> int:
        return (self.start_ms + self.seconds * 1000) * 1_000_000


def generate_bronze(run_dir: Path, config: FixtureConfig) -> tuple[Path, ...]:
    paths = []
    cme_rows = []
    for pair_index, (cme_symbol, mt5_symbol, base_price) in enumerate(PAIRS):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, pair_index]))
        rows = []
        price = base_price
        for index in range(config.seconds * 4):
            event_ms = config.start_ms + index * 250
            price = round(price + float(rng.normal(0, 0.08)), 4)
            spread = round(0.2 + float(rng.uniform(0, 0.15)), 4)
            bid = round(price - spread / 2, 4)
            ask = round(price + spread / 2, 4)
            if config.scenario == "crossed-quotes" and index % 5 == 0:
                ask = round(bid - spread, 4)
            rows.append(
                {
                    "symbol": mt5_symbol,
                    "time_msc_raw": event_ms + FIXTURE_OFFSET_MINUTES * 60_000,
                    "recv_time_utc_ms": event_ms + int(rng.integers(5, 25)),
                    "monotonic_ns": index * 250_000_000,
                    "bid": bid,
                    "ask": ask,
                    "last": 0.0,
                    "bid_scaled": round(bid * 10000),
                    "ask_scaled": round(ask * 10000),
                    "last_scaled": 0,
                    "volume": 0,
                    "flags": 6,
                    "spread_points": round((ask - bid) * 10000),
                    "suppressed_count": 0,
                    "first_suppressed_time_ms": None,
                    "last_suppressed_time_ms": None,
                    "suppressed_reason": None,
                }
            )
            size = int(rng.integers(1, 20))
            side = "B" if rng.random() > 0.48 else "A"
            if index % 7 != 0:
                trade_price = round(price + 1.0 + float(rng.normal(0, 0.04)), 4)
                cme_rows.append(
                    {
                        "ts_event": event_ms * 1_000_000 + 30_000_000,
                        "ts_recv": event_ms * 1_000_000 + 32_000_000,
                        "raw_symbol": f"{cme_symbol}H6",
                        "instrument_id": pair_index + 1,
                        "action": "T",
                        "side": side,
                        "price": trade_price,
                        "size": size,
                        "bid_px_00": trade_price - 0.25,
                        "ask_px_00": trade_price + 0.25,
                        "bid_sz_00": size + 2,
                        "ask_sz_00": size + 3,
                    }
                )
        path = (
            run_dir
            / "bronze"
            / "mt5"
            / f"symbol={mt5_symbol}"
            / f"date={FIXTURE_DATE}"
            / "hour=12"
            / "part-00001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA), path, compression="zstd")
        paths.append(path)
    cme_path = run_dir / "bronze" / "cme" / f"date={FIXTURE_DATE}" / "synthetic-mbp1.parquet"
    cme_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cme_rows).sort(["ts_event", "raw_symbol"]).write_parquet(
        cme_path, compression="zstd"
    )
    paths.append(cme_path)
    metadata_path = run_dir / "bronze" / "logger_manifest.json"
    publish_json(
        metadata_path,
        {
            "data_origin": "synthetic",
            "generator": "polarix.ingestion.fixtures",
            "generator_version": "0.2.0",
            "seed": config.seed,
            "scenario": config.scenario,
            "symbols": [pair[1] for pair in PAIRS],
            "timestamp_offset_source": "generator_contract",
            "timestamp_semantics": {
                "status": "UNVERIFIED"
                if config.scenario == "invalid-timestamps"
                else "OFFSET_VERIFIED_FOR_SESSION",
                "verified_offset_min": None
                if config.scenario == "invalid-timestamps"
                else FIXTURE_OFFSET_MINUTES,
            },
            "window_start_utc_ns": config.start_ms * 1_000_000,
            "window_end_utc_ns_exclusive": config.end_ns,
            "market_data_downloaded": False,
        },
    )
    paths.append(metadata_path)
    return tuple(paths)
