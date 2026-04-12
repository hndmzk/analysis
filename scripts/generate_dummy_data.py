from __future__ import annotations

import argparse
import json

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, resolve_storage_path, update_settings
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy OHLCV and macro datasets.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--dummy-mode",
        choices=["null_random_walk", "predictable_momentum"],
        default=None,
        help="Override dummy generation mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.dummy_mode:
        settings = update_settings(settings, data={"dummy_mode": args.dummy_mode})
    configure_logging(settings.app.log_level)
    storage_path = resolve_storage_path(settings)
    data_agent = DataAgent(settings, ParquetStore(storage_path))
    as_of_timestamp = pd.Timestamp.now(tz="UTC")
    end_date = as_of_timestamp.normalize()
    start_date = end_date - pd.tseries.offsets.BDay(settings.data.dummy_days - 1)
    tickers = [f"TICK{i:03d}" for i in range(settings.data.dummy_ticker_count)]
    result = data_agent.generate_or_fetch(
        tickers=tickers,
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat(),
        as_of_time=as_of_timestamp,
    )
    summary = {
        "storage_path": str(storage_path),
        "dummy_mode": settings.data.dummy_mode,
        "ohlcv_rows": len(result.processed_ohlcv),
        "macro_rows": len(result.processed_macro),
        "used_source": result.ohlcv_metadata["used_source"],
        "fallback_used": result.ohlcv_metadata["fallback_used"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
