from __future__ import annotations

import pandas as pd
import pytest

from market_prediction_agent.evaluation.event_reaction import (
    build_event_reaction_summary,
    compute_event_abnormal_return,
    detect_earnings_events,
)


def test_compute_event_abnormal_return_returns_expected_values() -> None:
    dates = pd.bdate_range("2026-01-05", periods=5, tz="UTC")
    ticker_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, dtype=float)
    sector_returns = pd.Series([0.00, 0.01, 0.02, 0.01, 0.03], index=dates, dtype=float)

    result = compute_event_abnormal_return(
        ticker_returns=ticker_returns,
        sector_returns=sector_returns,
        event_date="2026-01-07",
        window=2,
    )

    assert result["pre_event_ar"] == pytest.approx(0.02)
    assert result["post_event_ar"] == pytest.approx(0.05)
    assert result["cumulative_ar"] == pytest.approx(0.07)
    assert result["t_statistic"] == pytest.approx(3.6556307751)
    assert result["observation_count"] == 4


def test_detect_earnings_events_extracts_report_dates_and_effective_dates() -> None:
    fundamentals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "report_date": "2026-01-02T00:00:00Z",
                "available_at": "2026-01-05T13:00:00Z",
                "revenue_growth": 0.10,
            },
            {
                "ticker": "AAA",
                "report_date": "2026-04-01T00:00:00Z",
                "available_at": "2026-04-01T22:00:00Z",
                "revenue_growth": 0.12,
            },
        ]
    )

    events = detect_earnings_events(fundamentals)

    assert events["event_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-04-01"]
    assert events["effective_event_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-05", "2026-04-01"]
    assert events["availability_lag_days"].tolist() == [3, 0]


def test_compute_event_abnormal_return_handles_zero_division_and_empty_inputs() -> None:
    zero_dates = pd.bdate_range("2026-01-05", periods=5, tz="UTC")
    zero_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=zero_dates, dtype=float)

    zero_result = compute_event_abnormal_return(
        ticker_returns=zero_returns,
        sector_returns=zero_returns,
        event_date="2026-01-07",
        window=2,
    )
    empty_result = compute_event_abnormal_return(
        ticker_returns=pd.Series(dtype=float),
        sector_returns=pd.Series(dtype=float),
        event_date="2026-01-07",
        window=2,
    )

    assert zero_result["cumulative_ar"] == pytest.approx(0.0)
    assert zero_result["t_statistic"] == pytest.approx(0.0)
    assert empty_result["observation_count"] == 0
    assert empty_result["pre_event_ar"] == pytest.approx(0.0)
    assert empty_result["post_event_ar"] == pytest.approx(0.0)


def test_build_event_reaction_summary_aggregates_sector_results() -> None:
    events = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "sector": "technology",
                "report_date": "2026-01-06T00:00:00Z",
                "available_at": "2026-01-07T08:00:00Z",
            }
        ]
    )
    ohlcv = pd.DataFrame(
        [
            {"ticker": "AAA", "timestamp_utc": "2026-01-05T00:00:00Z", "close": 100.0},
            {"ticker": "AAA", "timestamp_utc": "2026-01-06T00:00:00Z", "close": 101.0},
            {"ticker": "AAA", "timestamp_utc": "2026-01-07T00:00:00Z", "close": 102.0},
            {"ticker": "AAA", "timestamp_utc": "2026-01-08T00:00:00Z", "close": 106.0},
            {"ticker": "AAA", "timestamp_utc": "2026-01-09T00:00:00Z", "close": 108.0},
        ]
    )
    sector_returns = pd.DataFrame(
        [
            {"sector": "technology", "date": "2026-01-06T00:00:00Z", "return": 0.005},
            {"sector": "technology", "date": "2026-01-07T00:00:00Z", "return": 0.006},
            {"sector": "technology", "date": "2026-01-08T00:00:00Z", "return": 0.010},
            {"sector": "technology", "date": "2026-01-09T00:00:00Z", "return": 0.012},
        ]
    )

    summary = build_event_reaction_summary(events, ohlcv, sector_returns, window=1)

    assert summary["event_count"] == 1
    assert summary["analyzed_event_count"] == 1
    assert summary["average_cumulative_ar"] > 0.0
    assert summary["sector_summary"][0]["sector"] == "technology"
