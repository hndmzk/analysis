from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
from uuid import uuid4

import pytest

from market_prediction_agent.data.adapters import (
    CoinGeckoAdapter,
    FredCsvMacroAdapter,
    FredMarketProxyOHLCVAdapter,
    FundamentalsRequest,
    GoogleNewsRssAdapter,
    MacroRequest,
    MultiSourceNewsAdapter,
    NewsRequest,
    OHLCVRequest,
    PublicDataTransportConfig,
    SecCompanyFactsAdapter,
    StooqOHLCVAdapter,
    YahooFinanceNewsAdapter,
    YahooChartOHLCVAdapter,
)


STOOQ_PAGE_1 = """
<html><body>
<table>
  <thead>
    <tr><th>No.</th><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Change</th><th>Volume</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>3 Apr 2026</td><td>101.0</td><td>102.0</td><td>100.0</td><td>101.5</td><td>0.5%</td><td>1,500,000</td></tr>
    <tr><td>2</td><td>2 Apr 2026</td><td>100.0</td><td>101.0</td><td>99.0</td><td>100.5</td><td>0.5%</td><td>1,400,000</td></tr>
  </tbody>
</table>
</body></html>
"""

STOOQ_PAGE_2 = """
<html><body>
<table>
  <thead>
    <tr><th>No.</th><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Change</th><th>Volume</th></tr>
  </thead>
  <tbody>
    <tr><td>3</td><td>1 Apr 2026</td><td>99.0</td><td>100.0</td><td>98.0</td><td>99.5</td><td>0.5%</td><td>1,300,000</td></tr>
    <tr><td>4</td><td>31 Mar 2026</td><td>98.0</td><td>99.0</td><td>97.0</td><td>98.5</td><td>0.5%</td><td>1,200,000</td></tr>
  </tbody>
</table>
</body></html>
"""

FRED_CSV = """DATE,VIXCLS
2026-04-01,20.1
2026-04-02,.
2026-04-03,21.4
"""

YAHOO_CHART_JSON = json.dumps(
    {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "SPY"},
                    "timestamp": [1775001600, 1775088000, 1775174400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0, 102.0],
                                "high": [101.0, 102.0, 103.0],
                                "low": [99.0, 100.0, 101.0],
                                "close": [100.5, 101.5, 102.5],
                                "volume": [1_000_000, 1_100_000, 1_200_000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }
)

YAHOO_FINANCE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Yahoo Finance Headlines</title>
    <item>
      <title>SPY beats estimates as growth outlook improves</title>
      <description>Analysts stay bullish after strong demand.</description>
      <pubDate>Fri, 03 Apr 2026 13:00:00 GMT</pubDate>
    </item>
    <item>
      <title>SPY faces lawsuit risk after weak warning</title>
      <description>Shares fall as investors react to the warning.</description>
      <pubDate>Thu, 02 Apr 2026 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

YAHOO_FINANCE_ALIAS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Yahoo Finance Headlines</title>
    <item>
      <title>Apple expands AI infrastructure after strong quarter</title>
      <description>Analysts highlight Apple Inc. momentum.</description>
      <pubDate>Fri, 03 Apr 2026 13:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

GOOGLE_NEWS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Apple expands AI infrastructure after strong quarter</title>
      <description>Analysts highlight Apple Inc. momentum.</description>
      <link>https://news.google.com/rss/articles/CBMiY2h0dHBzOi8vd3d3LnJldXRlcnMuY29tL3Rlc3QtYXJ0aWNsZdIBAA</link>
      <source url="https://www.reuters.com">Reuters</source>
      <pubDate>Fri, 03 Apr 2026 21:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

GOOGLE_NEWS_PREMARKET_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Apple expands AI infrastructure after strong quarter</title>
      <description>Analysts highlight Apple Inc. momentum.</description>
      <link>https://news.google.com/rss/articles/CBMiY2h0dHBzOi8vd3d3LnJldXRlcnMuY29tL3Rlc3QtYXJ0aWNsZdIBAA</link>
      <source url="https://www.reuters.com">Reuters</source>
      <pubDate>Fri, 03 Apr 2026 12:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

SEC_TICKER_MAP = json.dumps(
    {
        "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
    }
)

SEC_COMPANYFACTS = json.dumps(
    {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "filed": "2025-05-01",
                                "form": "10-Q",
                                "val": 100.0,
                            },
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "filed": "2025-08-01",
                                "form": "10-Q",
                                "val": 110.0,
                            },
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "filed": "2025-05-01",
                                "form": "10-Q",
                                "val": 20.0,
                            },
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "filed": "2025-08-01",
                                "form": "10-Q",
                                "val": 22.0,
                            },
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "filed": "2025-05-01",
                                "form": "10-Q",
                                "val": 1.0,
                            },
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "filed": "2025-08-01",
                                "form": "10-Q",
                                "val": 1.1,
                            },
                        ]
                    }
                },
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {"end": "2025-03-31", "filed": "2025-05-01", "form": "10-Q", "val": 50.0},
                            {"end": "2025-06-30", "filed": "2025-08-01", "form": "10-Q", "val": 55.0},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"end": "2025-03-31", "filed": "2025-05-01", "form": "10-Q", "val": 200.0},
                            {"end": "2025-06-30", "filed": "2025-08-01", "form": "10-Q", "val": 220.0},
                        ]
                    }
                },
            }
        }
    }
)

YAHOO_CHART_FUNDAMENTALS_JSON = json.dumps(
    {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "AAPL"},
                    "timestamp": [1746057600, 1754006400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [99.0, 104.0],
                                "high": [101.0, 106.0],
                                "low": [98.0, 103.0],
                                "close": [100.0, 105.0],
                                "volume": [1_000_000, 1_100_000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }
)

COINGECKO_MARKET_CHART = json.dumps(
    {
        "prices": [
            [1714521600000, 65000.0],
            [1714550400000, 65500.0],
            [1714608000000, 66000.0],
            [1714636800000, 65800.0],
        ],
        "total_volumes": [
            [1714521600000, 1_000_000_000.0],
            [1714550400000, 1_100_000_000.0],
            [1714608000000, 1_200_000_000.0],
            [1714636800000, 1_250_000_000.0],
        ],
    }
)


@contextmanager
def workspace_temp_dir():
    path = Path("storage") / "test_public_adapters_tmp" / str(uuid4())
    path.mkdir(parents=True, exist_ok=True)
    yield path


def test_stooq_adapter_fetches_paginated_history(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        pages = [STOOQ_PAGE_1, STOOQ_PAGE_2]
        calls: list[str] = []

        def fake_get_text(url: str) -> str:
            calls.append(url)
            return pages[len(calls) - 1] if len(calls) <= len(pages) else "<html></html>"

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_get_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = StooqOHLCVAdapter(max_pages=3, transport=transport)
        frame = adapter.fetch(OHLCVRequest(tickers=["SPY"], start_date="2026-04-01", end_date="2026-04-03"))
        assert len(frame) == 3
        assert frame["source"].iloc[0] == "stooq"
        assert frame["timestamp_utc"].min().date().isoformat() == "2026-04-01"
        assert len(calls) == 2


def test_stooq_adapter_accepts_tokyo_ticker_suffix(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        calls: list[str] = []

        def fake_get_text(url: str) -> str:
            calls.append(url)
            return STOOQ_PAGE_1 if len(calls) == 1 else "<html></html>"

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_get_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = StooqOHLCVAdapter(max_pages=2, transport=transport)
        frame = adapter.fetch(OHLCVRequest(tickers=["7203.T"], start_date="2026-04-02", end_date="2026-04-03"))
        assert len(frame) == 2
        assert frame["ticker"].unique().tolist() == ["7203.T"]
        assert "s=7203.jp" in calls[0]


def test_yahoo_chart_adapter_parses_ohlcv(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: YAHOO_CHART_JSON)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = YahooChartOHLCVAdapter(transport=transport)
        frame = adapter.fetch(OHLCVRequest(tickers=["SPY"], start_date="2026-04-01", end_date="2026-04-03"))
        assert len(frame) == 3
        assert frame["source"].iloc[0] == "yahoo_chart"
        assert frame["close"].iloc[-1] == 102.5


def test_coingecko_adapter_parses_market_chart_range(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: COINGECKO_MARKET_CHART)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = CoinGeckoAdapter(transport=transport)
        frame = adapter.fetch(OHLCVRequest(tickers=["BTC-USD"], start_date="2024-05-01", end_date="2024-05-02"))
        assert len(frame) == 2
        assert frame["source"].unique().tolist() == ["coingecko"]
        assert frame["open"].iloc[0] == pytest.approx(65000.0)
        assert frame["close"].iloc[0] == pytest.approx(65500.0)
        assert frame["volume"].iloc[-1] == pytest.approx(1_250_000_000.0)


def test_coingecko_adapter_dummy_mode_skips_network(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr(
            "market_prediction_agent.data.adapters._read_text_response",
            lambda url: (_ for _ in ()).throw(AssertionError("network should not run")),
        )
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = CoinGeckoAdapter(
            transport=transport,
            source_mode="dummy",
            seed=42,
            dummy_mode="predictable_momentum",
        )
        frame = adapter.fetch(OHLCVRequest(tickers=["BTC-USD", "ETH-USD"], start_date="2024-05-01", end_date="2024-05-03"))
        assert not frame.empty
        assert frame["source"].unique().tolist() == ["dummy"]
        assert adapter.last_fetch_metadata["origins"] == ["dummy"]


def test_yahoo_finance_news_adapter_parses_feed(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: YAHOO_FINANCE_RSS)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = YahooFinanceNewsAdapter(transport=transport)
        frame = adapter.fetch(NewsRequest(tickers=["SPY"], start_date="2026-04-02", end_date="2026-04-03"))
        assert len(frame) == 2
        assert frame["source"].unique().tolist() == ["yahoo_finance_rss"]
        assert int(frame["headline_count"].sum()) == 2
        assert frame.loc[frame["headline_count"] > 0, "session_bucket"].iloc[0] == "mixed"
        assert adapter.last_fetch_metadata["network_used"] is True
        assert "network" in adapter.last_fetch_metadata["origins"]


def test_yahoo_finance_news_adapter_maps_company_alias_without_symbol_match(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: YAHOO_FINANCE_ALIAS_RSS)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = YahooFinanceNewsAdapter(transport=transport)
        frame = adapter.fetch(NewsRequest(tickers=["AAPL"], start_date="2026-04-02", end_date="2026-04-03"))
        assert len(frame) == 2
        assert int(frame["headline_count"].sum()) == 1
        assert float(frame["mapping_confidence"].max()) > 0.55
        assert float(frame["source_diversity"].max()) == 1.0


def test_google_news_adapter_uses_publisher_source_and_post_market_alignment(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: GOOGLE_NEWS_RSS)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = GoogleNewsRssAdapter(transport=transport)
        frame = adapter.fetch(NewsRequest(tickers=["AAPL"], start_date="2026-04-02", end_date="2026-04-03"))
        active = frame.loc[frame["headline_count"] > 0].iloc[0]
        assert active["source"] == "google_news_rss"
        assert active["session_bucket"] == "post_market"
        assert active["source_diversity"] == pytest.approx(1.0)
        assert active["source_count"] == pytest.approx(1.0)
        assert active["source_mix"] == "google_news_rss"


def test_multi_source_news_adapter_combines_sources(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        def fake_get_text(url: str) -> str:
            if "finance.yahoo.com" in url:
                return YAHOO_FINANCE_ALIAS_RSS
            if "news.google.com" in url:
                return GOOGLE_NEWS_PREMARKET_RSS
            raise AssertionError(url)

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_get_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = MultiSourceNewsAdapter(
            [
                YahooFinanceNewsAdapter(transport=transport),
                GoogleNewsRssAdapter(transport=transport),
            ]
        )
        frame = adapter.fetch(NewsRequest(tickers=["AAPL"], start_date="2026-04-02", end_date="2026-04-03"))
        active = frame.loc[frame["headline_count"] > 0].iloc[0]
        assert adapter.last_fetch_metadata["used_sources"] == ["yahoo_finance_rss", "google_news_rss"]
        assert active["source_count"] >= 2.0
        assert "google_news_rss" in active["source_mix"]
        assert "yahoo_finance_rss" in active["source_mix"]
        assert "alpha" not in active["source_session_breakdown"]
        assert "google_news_rss::pre_market" in active["source_session_breakdown"]
        assert "yahoo_finance_rss::pre_market" in active["source_session_breakdown"]


def test_sec_companyfacts_adapter_parses_quarterly_history(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        def fake_get_text(url: str, headers=None) -> str:
            if "company_tickers.json" in url:
                return SEC_TICKER_MAP
            if "companyfacts" in url:
                return SEC_COMPANYFACTS
            if "finance/chart/AAPL" in url:
                return YAHOO_CHART_FUNDAMENTALS_JSON
            raise AssertionError(url)

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_get_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = SecCompanyFactsAdapter(transport=transport)
        frame = adapter.fetch(FundamentalsRequest(tickers=["AAPL"], start_date="2025-01-01", end_date="2025-12-31"))
        assert len(frame) == 2
        assert frame["source"].unique().tolist() == ["sec_companyfacts"]
        assert frame["revenue_growth"].iloc[0] == pytest.approx(0.0)
        assert frame["revenue_growth"].iloc[1] == pytest.approx(0.1)
        assert frame["debt_to_equity"].iloc[0] == pytest.approx(0.25)
        assert frame["earnings_yield"].iloc[0] == pytest.approx(0.04)
        assert adapter.last_fetch_metadata["network_used"] is True
        assert "network" in adapter.last_fetch_metadata["origins"]


def test_fred_csv_adapter_filters_missing_values(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: FRED_CSV)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = FredCsvMacroAdapter(transport=transport)
        frame = adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        assert len(frame) == 2
        assert frame["source"].iloc[0] == "fred_csv"
        assert frame["series_id"].unique().tolist() == ["VIXCLS"]


def test_fred_market_proxy_builds_proxy_ohlcv(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        csv_by_series = {
            "SP500": "DATE,SP500\n2026-04-01,5600\n2026-04-02,5650\n2026-04-03,5630\n",
            "NASDAQCOM": "DATE,NASDAQCOM\n2026-04-01,18000\n2026-04-02,18100\n2026-04-03,18250\n",
        }

        def fake_get_text(url: str) -> str:
            for series_id, payload in csv_by_series.items():
                if f"id={series_id}" in url:
                    return payload
            raise AssertionError(url)

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_get_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = FredMarketProxyOHLCVAdapter(transport=transport)
        frame = adapter.fetch(OHLCVRequest(tickers=["SPY", "QQQ"], start_date="2026-04-01", end_date="2026-04-03"))
        assert sorted(frame["ticker"].unique().tolist()) == ["QQQ", "SPY"]
        assert frame["source"].unique().tolist() == ["fred_market_proxy"]
        assert frame["volume"].min() == 1_000_000_000.0


def test_public_data_transport_uses_cache_after_success(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        calls = {"count": 0}

        def fake_read_text(url: str) -> str:
            calls["count"] += 1
            return FRED_CSV

        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", fake_read_text)
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=24,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        adapter = FredCsvMacroAdapter(transport=transport)
        adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        assert calls["count"] == 1

        monkeypatch.setattr(
            "market_prediction_agent.data.adapters._read_text_response",
            lambda url: (_ for _ in ()).throw(RuntimeError("network should not run")),
        )
        adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        assert adapter.last_fetch_metadata["cache_used"] is True
        assert "cache" in adapter.last_fetch_metadata["origins"]


def test_public_data_transport_falls_back_to_snapshot(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=0,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: FRED_CSV)
        adapter = FredCsvMacroAdapter(transport=transport)
        adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        cache_files = list((root / "cache").rglob("*.json"))
        assert cache_files
        for path in cache_files:
            path.unlink()

        monkeypatch.setattr(
            "market_prediction_agent.data.adapters._read_text_response",
            lambda url: (_ for _ in ()).throw(RuntimeError("network down")),
        )
        frame = adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        assert not frame.empty
        assert adapter.last_fetch_metadata["snapshot_used"] is True
        assert "snapshot" in adapter.last_fetch_metadata["origins"]


def test_public_data_transport_reuses_compatible_snapshot_for_narrower_range(monkeypatch) -> None:
    with workspace_temp_dir() as root:
        transport = PublicDataTransportConfig(
            cache_dir=root / "cache",
            snapshot_dir=root / "snapshots",
            cache_ttl_hours=0,
            retry_count=1,
            retry_backoff_seconds=0.0,
        )
        monkeypatch.setattr("market_prediction_agent.data.adapters._read_text_response", lambda url: FRED_CSV)
        adapter = FredCsvMacroAdapter(transport=transport)
        adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-01", end_date="2026-04-03"))
        for path in (root / "cache").rglob("*.json"):
            path.unlink()
        monkeypatch.setattr(
            "market_prediction_agent.data.adapters._read_text_response",
            lambda url: (_ for _ in ()).throw(RuntimeError("network down")),
        )
        frame = adapter.fetch(MacroRequest(series_ids=["VIXCLS"], start_date="2026-04-02", end_date="2026-04-03"))
        assert not frame.empty
        assert adapter.last_fetch_metadata["snapshot_used"] is True
        assert "compatible_snapshot" in adapter.last_fetch_metadata["origins"]
