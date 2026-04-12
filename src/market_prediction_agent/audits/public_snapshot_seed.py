from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import Settings, resolve_storage_path, update_settings
from market_prediction_agent.storage.parquet_store import ParquetStore


@dataclass(slots=True)
class PublicSnapshotSeedResult:
    payload: dict[str, Any]


def seed_public_snapshots(
    settings: Settings,
    *,
    tickers: list[str] | None = None,
    history_days: int | None = None,
    as_of_time: pd.Timestamp | None = None,
) -> PublicSnapshotSeedResult:
    run_settings = update_settings(
        settings,
        data={
            "source_mode": "live",
            "primary_source": "yahoo_chart",
            "fallback_source": "fred_market_proxy",
            "macro_source": "fred_csv",
            **({"dummy_days": history_days} if history_days is not None else {}),
        },
    )
    as_of_timestamp = as_of_time or pd.Timestamp.now(tz="UTC")
    end_date = as_of_timestamp.normalize()
    start_date = end_date - pd.tseries.offsets.BDay(run_settings.data.dummy_days - 1)
    storage_path = resolve_storage_path(run_settings)
    data_agent = DataAgent(run_settings, ParquetStore(storage_path))
    seed_tickers = tickers or data_agent.default_tickers()
    if not seed_tickers:
        raise ValueError("At least one ticker is required to seed public snapshots.")
    artifacts = data_agent.generate_or_fetch(
        tickers=seed_tickers,
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat(),
        as_of_time=as_of_timestamp,
    )
    payload = {
        "seeded_at": as_of_timestamp.isoformat(),
        "window": {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
        },
        "tickers": seed_tickers,
        "storage_path": str(storage_path),
        "data_sources": {
            "ohlcv_source": artifacts.ohlcv_metadata.get("used_source"),
            "proxy_ohlcv_used": artifacts.ohlcv_metadata.get("used_source") == "fred_market_proxy",
            "macro_source": artifacts.ohlcv_metadata.get("macro_source"),
            "ohlcv_transport": artifacts.ohlcv_metadata.get("public_data_transport", {}),
            "macro_transport": artifacts.ohlcv_metadata.get("macro_public_data_transport", {}),
        },
        "rows": {
            "ohlcv": int(len(artifacts.raw_ohlcv)),
            "macro": int(len(artifacts.raw_macro)),
        },
    }
    return PublicSnapshotSeedResult(payload=payload)
