from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.storage.parquet_store import ParquetStore


class SuccessfulOHLCVAdapter:
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.calls = 0

    def fetch(self, request) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame(
            [
                {
                    "ticker": request.tickers[0],
                    "timestamp_utc": pd.Timestamp(request.start_date, tz="UTC"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.5,
                    "volume": 1_000_000.0,
                    "source": self.source_name,
                    "fetched_at": pd.Timestamp.now(tz="UTC"),
                    "stale_data_flag": False,
                }
            ]
        )


class SuccessfulMacroAdapter:
    def fetch(self, request) -> pd.DataFrame:
        rows = []
        for series_id in request.series_ids:
            rows.append(
                {
                    "series_id": series_id,
                    "date": pd.Timestamp(request.start_date, tz="UTC"),
                    "value": 1.0,
                    "available_at": pd.Timestamp(request.start_date, tz="UTC"),
                    "source": "mock",
                }
            )
        return pd.DataFrame(rows)


class FailingAdapter:
    def __init__(self, message: str) -> None:
        self.message = message

    def fetch(self, request) -> pd.DataFrame:
        raise RuntimeError(self.message)


def make_live_agent() -> DataAgent:
    settings = update_settings(
        load_settings(),
        data={
            "source_mode": "live",
            "storage_path": str(Path(".test-artifacts") / uuid4().hex / "storage"),
            "fundamentals_source": "offline_fundamental_proxy",
            "fundamentals_fallback_source": "offline_fundamental_proxy",
        },
    )
    store = ParquetStore(Path(settings.data.storage_path))
    return DataAgent(settings, store)


def test_primary_source_success_does_not_trigger_fallback(monkeypatch, caplog) -> None:
    agent = make_live_agent()
    primary = SuccessfulOHLCVAdapter("polygon")
    fallback = SuccessfulOHLCVAdapter("alphavantage")
    monkeypatch.setattr(
        agent,
        "_build_ohlcv_adapter",
        lambda source_name: {"polygon": primary, "alphavantage": fallback}[source_name],
    )
    monkeypatch.setattr(agent, "_build_macro_adapter", lambda: SuccessfulMacroAdapter())
    artifacts = agent.generate_or_fetch(["AAA"], "2026-01-01", "2026-01-10")
    assert artifacts.ohlcv_metadata["used_source"] == "polygon"
    assert artifacts.ohlcv_metadata["fallback_used"] is False
    assert primary.calls == 1
    assert fallback.calls == 0
    assert "fallback" not in caplog.text.lower()


def test_primary_failure_uses_fallback(monkeypatch, caplog) -> None:
    agent = make_live_agent()
    fallback = SuccessfulOHLCVAdapter("alphavantage")
    monkeypatch.setattr(
        agent,
        "_build_ohlcv_adapter",
        lambda source_name: {
            "polygon": FailingAdapter("polygon down"),
            "alphavantage": fallback,
        }[source_name],
    )
    monkeypatch.setattr(agent, "_build_macro_adapter", lambda: SuccessfulMacroAdapter())
    with caplog.at_level("WARNING"):
        artifacts = agent.generate_or_fetch(["AAA"], "2026-01-01", "2026-01-10")
    assert artifacts.ohlcv_metadata["used_source"] == "alphavantage"
    assert artifacts.ohlcv_metadata["fallback_used"] is True
    assert artifacts.ohlcv_metadata["fallback_reason"] == "polygon down"
    assert fallback.calls == 1
    assert "polygon down" in caplog.text


def test_primary_and_fallback_failure_raise_explicit_error(monkeypatch) -> None:
    agent = make_live_agent()
    monkeypatch.setattr(
        agent,
        "_build_ohlcv_adapter",
        lambda source_name: {
            "polygon": FailingAdapter("polygon down"),
            "alphavantage": FailingAdapter("alpha down"),
        }[source_name],
    )
    monkeypatch.setattr(agent, "_build_macro_adapter", lambda: SuccessfulMacroAdapter())
    with pytest.raises(RuntimeError, match="OHLCV fetch failed for all configured sources"):
        agent.generate_or_fetch(["AAA"], "2026-01-01", "2026-01-10")
