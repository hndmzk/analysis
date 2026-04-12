from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.data.adapters import NewsRequest, OfflineNewsProxyAdapter
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


class _FailingNewsAdapter:
    name = "yahoo_finance_rss"
    last_fetch_metadata = {
        "origins": ["network"],
        "cache_used": False,
        "snapshot_used": False,
        "stale_cache_used": False,
        "requests": [
            {
                "origin": "network",
                "cache_used": False,
                "snapshot_used": False,
                "stale_cache_used": False,
                "source_name": "yahoo_finance_rss",
            }
        ],
    }

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        del request
        raise RuntimeError("rss unavailable")

    def fetch_raw_items(self, request: NewsRequest) -> tuple[pd.DataFrame, dict[str, object]]:
        del request
        raise RuntimeError("rss unavailable")


class _SuccessfulNewsAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_fetch_metadata = {
            "origins": ["network"],
            "cache_used": False,
            "snapshot_used": False,
            "stale_cache_used": False,
            "used_sources": [name],
            "requests": [
                {
                    "origin": "network",
                    "cache_used": False,
                    "snapshot_used": False,
                    "stale_cache_used": False,
                    "source_name": name,
                }
            ],
        }

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        del request
        return pd.DataFrame(
            {
                "ticker": ["SPY"],
                "published_at": pd.to_datetime(["2026-04-03T12:00:00Z"]),
                "available_at": pd.to_datetime(["2026-04-03T00:00:00Z"]),
                "sentiment_score": [0.2],
                "relevance_score": [0.8],
                "headline_count": [1],
                "mapping_confidence": [1.0],
                "novelty_score": [1.0],
                "source_diversity": [1.0],
                "source_count": [1.0],
                "source_mix": [self.name],
                "session_bucket": ["pre_market"],
                "source": [self.name],
                "fetched_at": pd.to_datetime(["2026-04-03T13:00:00Z"]),
                "stale_data_flag": [False],
            }
        )

    def fetch_raw_items(self, request: NewsRequest) -> tuple[pd.DataFrame, dict[str, object]]:
        del request
        return (
            pd.DataFrame(
                {
                    "feed_ticker": ["SPY"],
                    "published_at": pd.to_datetime(["2026-04-03T12:00:00Z"]),
                    "body": ["SPY momentum improves"],
                    "headline_key": ["spy momentum improves"],
                    "source_key": [self.name],
                    "source_label": [self.name],
                    "source_name": [self.name],
                    "sentiment_score": [0.2],
                    "base_relevance_score": [0.8],
                }
            ),
            dict(self.last_fetch_metadata),
        )


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-data-agent-news" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_data_agent_news_falls_back_to_proxy_with_metadata(monkeypatch) -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(
            load_settings("config/default.yaml"),
            data={
                "source_mode": "live",
                "news_source": "yahoo_finance_rss",
                "news_fallback_source": "offline_news_proxy",
            },
        )
        agent = DataAgent(settings, ParquetStore(temp_dir))

        def fake_build_news_adapter(source_name: str | None = None):
            if source_name == "yahoo_finance_rss":
                return _FailingNewsAdapter()
            if source_name == "google_news_rss":
                return _FailingNewsAdapter()
            if source_name == "offline_news_proxy":
                return OfflineNewsProxyAdapter(seed=settings.app.seed, mode="live_proxy")
            raise AssertionError(source_name)

        monkeypatch.setattr(agent, "_build_news_adapter", fake_build_news_adapter)
        frame, metadata = agent._fetch_news_with_fallback(
            NewsRequest(tickers=["SPY"], start_date="2026-04-01", end_date="2026-04-03"),
            as_of_timestamp=pd.Timestamp("2026-04-04", tz="UTC"),
        )
        assert not frame.empty
        assert metadata["requested_source"] == "yahoo_finance_rss"
        assert metadata["used_source"] == "offline_news_proxy"
        assert metadata["requested_sources"] == ["yahoo_finance_rss", "google_news_rss"]
        assert metadata["fallback_used"] is True
        assert "rss unavailable" in str(metadata["fallback_reason"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_agent_news_records_multi_source_usage(monkeypatch) -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(
            load_settings("config/default.yaml"),
            data={
                "source_mode": "live",
                "news_source": "yahoo_finance_rss",
                "news_secondary_sources": ["google_news_rss"],
            },
        )
        agent = DataAgent(settings, ParquetStore(temp_dir))

        def fake_build_news_adapter(source_name: str | None = None):
            if source_name == "yahoo_finance_rss":
                return _SuccessfulNewsAdapter("yahoo_finance_rss")
            if source_name == "google_news_rss":
                return _SuccessfulNewsAdapter("google_news_rss")
            if source_name == "offline_news_proxy":
                raise AssertionError("fallback should not be used")
            raise AssertionError(source_name)

        monkeypatch.setattr(agent, "_build_news_adapter", fake_build_news_adapter)
        frame, metadata = agent._fetch_news_with_fallback(
            NewsRequest(tickers=["SPY"], start_date="2026-04-01", end_date="2026-04-03"),
            as_of_timestamp=pd.Timestamp("2026-04-04", tz="UTC"),
        )
        assert not frame.empty
        assert metadata["used_source"] == "yahoo_finance_rss+google_news_rss"
        assert metadata["used_sources"] == ["yahoo_finance_rss", "google_news_rss"]
        assert metadata["fallback_used"] is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
