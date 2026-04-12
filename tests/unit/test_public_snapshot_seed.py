from __future__ import annotations

import pandas as pd

from market_prediction_agent.agents.data_agent import DataArtifacts
from market_prediction_agent.audits.public_snapshot_seed import seed_public_snapshots
from market_prediction_agent.config import load_settings


def test_seed_public_snapshots_builds_summary(monkeypatch) -> None:
    settings = load_settings("config/default.yaml")

    def fake_generate_or_fetch(self, tickers, start_date, end_date, as_of_time=None) -> DataArtifacts:
        del self, start_date, end_date, as_of_time
        raw_ohlcv = pd.DataFrame(
            [
                {
                    "ticker": tickers[0],
                    "timestamp_utc": pd.Timestamp("2026-04-01", tz="UTC"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000_000.0,
                    "source": "fred_market_proxy",
                    "fetched_at": pd.Timestamp("2026-04-03", tz="UTC"),
                    "stale_data_flag": False,
                }
            ]
        )
        raw_macro = pd.DataFrame(
            [
                {
                    "series_id": "VIXCLS",
                    "date": pd.Timestamp("2026-04-01", tz="UTC"),
                    "value": 20.0,
                    "available_at": pd.Timestamp("2026-04-02", tz="UTC"),
                    "source": "fred_csv",
                }
            ]
        )
        return DataArtifacts(
            raw_ohlcv=raw_ohlcv,
            raw_macro=raw_macro,
            raw_news=pd.DataFrame(),
            raw_fundamentals=pd.DataFrame(),
            raw_sector_map=pd.DataFrame(),
            processed_ohlcv=raw_ohlcv,
            processed_macro=raw_macro,
            processed_news=pd.DataFrame(),
            processed_fundamentals=pd.DataFrame(),
            processed_sector_map=pd.DataFrame(),
            ohlcv_metadata={
                "used_source": "fred_market_proxy",
                "macro_source": "fred_csv",
                "public_data_transport": {"origins": ["network"], "cache_used": False, "snapshot_used": False},
                "macro_public_data_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
            },
        )

    monkeypatch.setattr(
        "market_prediction_agent.agents.data_agent.DataAgent.generate_or_fetch",
        fake_generate_or_fetch,
    )
    result = seed_public_snapshots(
        settings,
        tickers=["SPY", "QQQ"],
        history_days=10,
        as_of_time=pd.Timestamp("2026-04-03", tz="UTC"),
    )
    assert result.payload["tickers"] == ["SPY", "QQQ"]
    assert result.payload["data_sources"]["proxy_ohlcv_used"] is True
    assert result.payload["rows"]["ohlcv"] == 1
    assert result.payload["rows"]["macro"] == 1
