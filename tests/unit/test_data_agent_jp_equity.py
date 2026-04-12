from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.data.adapters import OHLCVRequest
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-data-agent-jp-equity" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_fetch_ohlcv_with_fallback_routes_tokyo_tickers_to_jp_equity_source(monkeypatch) -> None:
    temp_dir = _workspace_temp_dir()
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "live",
            "primary_source": "polygon",
            "fallback_source": "alphavantage",
            "crypto_enabled": False,
            "jp_equity": {
                "enabled": True,
                "source": "stooq",
            },
        },
    )
    agent = DataAgent(settings, ParquetStore(temp_dir))

    class FakeAdapter:
        def __init__(self, source_name: str) -> None:
            self.name = source_name
            self.last_fetch_metadata = {"origins": ["dummy"], "cache_used": False, "snapshot_used": False}

        def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "timestamp_utc": pd.Timestamp(request.start_date, tz="UTC"),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1_000_000.0,
                        "source": self.name,
                        "fetched_at": pd.Timestamp("2026-04-09", tz="UTC"),
                        "stale_data_flag": False,
                    }
                    for ticker in request.tickers
                ]
            )

    monkeypatch.setattr(
        agent,
        "_build_ohlcv_adapter",
        lambda source_name: FakeAdapter(source_name),
    )

    try:
        frame, metadata = agent._fetch_ohlcv_with_fallback(
            OHLCVRequest(tickers=["SPY", "7203.T"], start_date="2026-04-01", end_date="2026-04-03"),
            as_of_timestamp=pd.Timestamp("2026-04-03", tz="UTC"),
        )

        assert sorted(frame["ticker"].unique().tolist()) == ["7203.T", "SPY"]
        assert metadata["used_sources"] == ["polygon", "stooq"]
        assert frame.loc[frame["ticker"] == "7203.T", "source"].iloc[0] == "stooq"
        assert frame.loc[frame["ticker"] == "SPY", "source"].iloc[0] == "polygon"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
