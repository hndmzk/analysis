from __future__ import annotations

import pandas as pd

from market_prediction_agent.data.adapters import (
    DummyMacroAdapter,
    DummyOHLCVAdapter,
    FundamentalsRequest,
    MacroRequest,
    NewsRequest,
    OHLCVRequest,
    OfflineFundamentalProxyAdapter,
    OfflineNewsProxyAdapter,
    SectorRequest,
    StaticSectorMapAdapter,
)
from market_prediction_agent.data.normalizer import (
    normalize_fundamentals,
    normalize_macro,
    normalize_news,
    normalize_ohlcv,
    normalize_sector_map,
)
from market_prediction_agent.features.pipeline import FEATURE_COLUMNS, build_feature_frame, build_training_frame


NEWS_DIAGNOSTIC_COLUMNS = (
    "news_sentiment_score_unweighted",
    "news_sentiment_score_source_weighted",
    "news_sentiment_score_session_weighted",
    "news_sentiment_score_source_session_weighted",
    "news_relevance_score_unweighted",
    "news_relevance_score_source_weighted",
    "news_relevance_score_session_weighted",
    "news_relevance_score_source_session_weighted",
)


def _sample_ohlcv() -> pd.DataFrame:
    return normalize_ohlcv(
        pd.DataFrame(
            [
                {"ticker": "AAA", "timestamp_utc": "2024-01-01T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
                {"ticker": "AAA", "timestamp_utc": "2024-01-02T00:00:00Z", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100.0},
                {"ticker": "AAA", "timestamp_utc": "2024-01-03T00:00:00Z", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1200.0},
                {"ticker": "BBB", "timestamp_utc": "2024-01-01T00:00:00Z", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.5, "volume": 800.0},
                {"ticker": "BBB", "timestamp_utc": "2024-01-02T00:00:00Z", "open": 51.0, "high": 52.0, "low": 50.0, "close": 51.5, "volume": 900.0},
                {"ticker": "BBB", "timestamp_utc": "2024-01-03T00:00:00Z", "open": 52.0, "high": 53.0, "low": 51.0, "close": 52.5, "volume": 1000.0},
            ]
        )
    )


def _sample_macro() -> pd.DataFrame:
    return normalize_macro(
        pd.DataFrame(
            [
                {"series_id": "FEDFUNDS", "date": "2023-12-31T00:00:00Z", "value": 5.25, "available_at": "2023-12-31T00:00:00Z"},
                {"series_id": "T10Y2Y", "date": "2023-12-31T00:00:00Z", "value": 0.5, "available_at": "2023-12-31T00:00:00Z"},
                {"series_id": "VIXCLS", "date": "2023-12-31T00:00:00Z", "value": 18.0, "available_at": "2023-12-31T00:00:00Z"},
            ]
        )
    )


def _sample_fundamentals() -> pd.DataFrame:
    return normalize_fundamentals(
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "report_date": "2023-12-31T00:00:00Z",
                    "available_at": "2023-12-31T00:00:00Z",
                    "revenue_growth": 0.12,
                    "earnings_yield": 0.05,
                    "debt_to_equity": 1.1,
                    "profitability": 0.2,
                }
                ,
                {
                    "ticker": "BBB",
                    "report_date": "2023-12-31T00:00:00Z",
                    "available_at": "2023-12-31T00:00:00Z",
                    "revenue_growth": 0.08,
                    "earnings_yield": 0.04,
                    "debt_to_equity": 0.9,
                    "profitability": 0.18,
                },
            ]
        )
    )


def _sample_sector_map() -> pd.DataFrame:
    return normalize_sector_map(pd.DataFrame([{"ticker": "AAA", "sector": "technology"}, {"ticker": "BBB", "sector": "finance"}]))


def _build_feature_frame_for_test(news: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(
        _sample_ohlcv(),
        _sample_macro(),
        news,
        _sample_fundamentals(),
        _sample_sector_map(),
        horizon_days=1,
        direction_threshold=0.005,
    ).feature_frame


def test_feature_pipeline_builds_expected_columns() -> None:
    ohlcv = DummyOHLCVAdapter(seed=42).fetch(OHLCVRequest(tickers=["AAA", "BBB"], start_date="2022-01-01", end_date="2025-12-31"))
    macro = DummyMacroAdapter(seed=42).fetch(MacroRequest(series_ids=["FEDFUNDS", "T10Y2Y", "VIXCLS"], start_date="2022-01-01", end_date="2025-12-31"))
    news = OfflineNewsProxyAdapter(seed=42, mode="null_random_walk").fetch(
        NewsRequest(tickers=["AAA", "BBB"], start_date="2022-01-01", end_date="2025-12-31")
    )
    fundamentals = OfflineFundamentalProxyAdapter(seed=42).fetch(
        FundamentalsRequest(tickers=["AAA", "BBB"], start_date="2022-01-01", end_date="2025-12-31")
    )
    sector_map = StaticSectorMapAdapter().fetch(SectorRequest(tickers=["AAA", "BBB"]))
    result = build_feature_frame(
        normalize_ohlcv(ohlcv),
        normalize_macro(macro),
        normalize_news(news),
        normalize_fundamentals(fundamentals),
        normalize_sector_map(sector_map),
        horizon_days=1,
        direction_threshold=0.005,
        source_metadata={
            "used_source": "dummy",
            "macro_source": "dummy",
            "feature_sources": {
                "news": {"used_source": "offline_news_proxy", "stale_rate": 0.0},
                "fundamental": {"used_source": "offline_fundamental_proxy", "stale_rate": 0.0},
                "sector": {"used_source": "static_sector_map", "stale_rate": 0.0},
            },
        },
    )
    training = build_training_frame(result.feature_frame)
    assert set(FEATURE_COLUMNS).issubset(training.columns)
    assert len(training) > 0
    assert training["direction_label"].isin([0, 1, 2]).all()
    feature_families = {item["feature_family"] for item in result.feature_catalog}
    assert {"news", "fundamental", "sector"}.issubset(feature_families)


def test_feature_frame_preserves_news_diagnostic_weighting_columns() -> None:
    news = normalize_news(
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "published_at": "2024-01-02T00:00:00Z",
                    "available_at": "2024-01-02T00:00:00Z",
                    "sentiment_score": 0.25,
                    "sentiment_score_unweighted": 0.21,
                    "sentiment_score_source_weighted": 0.22,
                    "sentiment_score_session_weighted": 0.23,
                    "sentiment_score_source_session_weighted": 0.24,
                    "relevance_score": 0.65,
                    "relevance_score_unweighted": 0.61,
                    "relevance_score_source_weighted": 0.62,
                    "relevance_score_session_weighted": 0.63,
                    "relevance_score_source_session_weighted": 0.64,
                    "headline_count": 3,
                }
            ]
        )
    )

    feature_frame = _build_feature_frame_for_test(news)

    assert set(NEWS_DIAGNOSTIC_COLUMNS).issubset(feature_frame.columns)
    initial_row = feature_frame.loc[
        (feature_frame["ticker"] == "AAA") & (feature_frame["date"] == pd.Timestamp("2024-01-01T00:00:00Z"))
    ].iloc[0]
    weighted_row = feature_frame.loc[
        (feature_frame["ticker"] == "AAA") & (feature_frame["date"] == pd.Timestamp("2024-01-02T00:00:00Z"))
    ].iloc[0]
    for column in NEWS_DIAGNOSTIC_COLUMNS:
        assert initial_row[column] == 0.0
    assert weighted_row["news_sentiment_score_unweighted"] == 0.21
    assert weighted_row["news_sentiment_score_source_weighted"] == 0.22
    assert weighted_row["news_sentiment_score_session_weighted"] == 0.23
    assert weighted_row["news_sentiment_score_source_session_weighted"] == 0.24
    assert weighted_row["news_relevance_score_unweighted"] == 0.61
    assert weighted_row["news_relevance_score_source_weighted"] == 0.62
    assert weighted_row["news_relevance_score_session_weighted"] == 0.63
    assert weighted_row["news_relevance_score_source_session_weighted"] == 0.64


def test_weighted_news_diagnostics_are_not_model_features() -> None:
    assert set(NEWS_DIAGNOSTIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)


def test_feature_frame_keeps_news_diagnostic_columns_when_news_is_empty() -> None:
    empty_news = normalize_news(
        pd.DataFrame(columns=["ticker", "published_at", "available_at", "sentiment_score", "relevance_score", "headline_count"])
    )

    feature_frame = _build_feature_frame_for_test(empty_news)

    assert set(NEWS_DIAGNOSTIC_COLUMNS).issubset(feature_frame.columns)
    for column in NEWS_DIAGNOSTIC_COLUMNS:
        series = feature_frame[column]
        assert series.isna().all() or (series == 0.0).all()
