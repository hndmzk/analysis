from __future__ import annotations

from market_prediction_agent.config import JpEquityConfig, load_settings, update_settings
from market_prediction_agent.data.universe import (
    PointInTimeUniverse,
    load_ticker_list,
    resolve_active_constituents,
    resolve_default_tickers,
)


def test_point_in_time_universe_get_constituents_respects_change_dates() -> None:
    universe = PointInTimeUniverse.from_static(["AAA", "BBB", "CCC"])
    universe.add_change("2024-03-01", added=["DDD"], removed=["BBB"])

    assert universe.get_constituents("2024-02-29") == ["AAA", "BBB", "CCC"]
    assert universe.get_constituents("2024-03-01") == ["AAA", "CCC", "DDD"]


def test_point_in_time_universe_add_change_updates_membership() -> None:
    universe = PointInTimeUniverse.from_static(["AAA", "BBB"])
    universe.add_change("2024-01-15", added=["CCC"], removed=["AAA"])
    universe.add_change("2024-02-01", added=["DDD"], removed=["BBB"])

    assert universe.get_constituents("2024-01-31") == ["BBB", "CCC"]
    assert universe.get_constituents("2024-02-15") == ["CCC", "DDD"]


def test_point_in_time_universe_from_static_returns_fixed_ticker_list() -> None:
    universe = PointInTimeUniverse.from_static(["AAA", "BBB", "AAA"])

    assert universe.get_constituents("2025-01-01") == ["AAA", "BBB"]
    assert universe.changes == []


def test_point_in_time_universe_from_json_parses_sample_history() -> None:
    universe = PointInTimeUniverse.from_json("config/universe_history.json")

    assert universe.get_constituents("2020-01-01") == ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "NVDA", "TSLA", "DISH"]
    assert universe.get_constituents("2024-07-01") == ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "LLY", "META", "CRWD"]


def test_topix_core30_config_contains_30_tickers() -> None:
    tickers = load_ticker_list("config/topix_core30.json")

    assert len(tickers) == 30
    assert tickers[0] == "2914.T"
    assert tickers[-1] == "3382.T"


def test_jp_equity_config_dataclass_initializes_expected_values() -> None:
    config = JpEquityConfig(
        enabled=False,
        universe="topix_core30",
        source="stooq",
        tickers_file="config/topix_core30.json",
    )
    settings = load_settings("config/default.yaml")

    assert config.source == "stooq"
    assert config.tickers_file.endswith("topix_core30.json")
    assert settings.data.jp_equity == config


def test_resolve_default_tickers_includes_jp_equity_when_enabled() -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "live",
            "default_tickers": ["SPY", "QQQ"],
            "crypto_enabled": False,
            "jp_equity": {"enabled": True},
        },
    )

    tickers = resolve_default_tickers(settings)

    assert tickers[:2] == ["SPY", "QQQ"]
    assert "7203.T" in tickers
    assert len(tickers) == 32


def test_resolve_active_constituents_for_sp500_pit_includes_jp_equity_when_enabled() -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "universe": "sp500_pit",
            "jp_equity": {"enabled": True},
        },
    )

    tickers = resolve_active_constituents(settings, as_of_date="2024-07-01")

    assert tickers is not None
    assert "META" in tickers
    assert "FB" not in tickers
    assert "7203.T" in tickers
