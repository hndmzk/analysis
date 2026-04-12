from __future__ import annotations

import json

import pytest
import pandas as pd

from market_prediction_agent.config import LearnedWeightingConfig
from market_prediction_agent.evaluation.learned_weighting import (
    build_walk_forward_learned_weighting,
    fit_learned_source_session_weights,
)


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "date": pd.to_datetime(
                ["2026-04-01", "2026-04-02", "2026-04-01", "2026-04-02"],
                utc=True,
            ),
            "source_name": ["alpha", "alpha", "beta", "beta"],
            "session_bucket": ["pre_market", "pre_market", "pre_market", "pre_market"],
            "combo_key": ["alpha::pre_market", "alpha::pre_market", "beta::pre_market", "beta::pre_market"],
            "base_weight_sum": [1.0, 1.0, 1.0, 1.0],
            "sentiment_weighted_sum": [1.0, -1.0, 1.0, 1.0],
            "relevance_weighted_sum": [1.0, 1.0, 1.0, 1.0],
            "headline_count": [1, 1, 1, 1],
            "article_count": [1, 1, 1, 1],
            "signal_day_return": [1.0, -1.0, 1.0, -1.0],
        }
    )


def test_fit_learned_source_session_weights_respects_shrinkage_limits() -> None:
    history = _history_frame()
    no_shrink = fit_learned_source_session_weights(
        history=history,
        config=LearnedWeightingConfig(
            regularization_lambda=0.0,
            min_samples=1,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.0,
            fallback_mode="fixed",
        ),
    )
    full_shrink = fit_learned_source_session_weights(
        history=history,
        config=LearnedWeightingConfig(
            regularization_lambda=1_000_000_000.0,
            min_samples=1,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.0,
            fallback_mode="fixed",
        ),
    )

    assert no_shrink["valid"] is True
    assert full_shrink["valid"] is True
    no_shrink_stats = {
        str(item["combo_key"]): item
        for item in no_shrink["combo_stats"]
    }
    full_shrink_stats = {
        str(item["combo_key"]): item
        for item in full_shrink["combo_stats"]
    }
    assert no_shrink_stats["alpha::pre_market"]["shrunk_score"] == pytest.approx(1.0)
    assert no_shrink_stats["beta::pre_market"]["shrunk_score"] == pytest.approx(0.0)
    assert full_shrink["global_score"] == pytest.approx(0.5)
    assert full_shrink_stats["alpha::pre_market"]["shrunk_score"] == pytest.approx(0.5, rel=1e-6)
    assert full_shrink_stats["beta::pre_market"]["shrunk_score"] == pytest.approx(0.5, rel=1e-6)


def test_fit_learned_source_session_weights_applies_min_weight_clip() -> None:
    history = _history_frame()
    fitted = fit_learned_source_session_weights(
        history=history,
        config=LearnedWeightingConfig(
            regularization_lambda=0.0,
            min_samples=1,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.25,
            fallback_mode="fixed",
        ),
    )

    assert fitted["valid"] is True
    weights = fitted["normalized_weights"]
    assert weights["beta::pre_market"] == pytest.approx(0.25)
    assert weights["alpha::pre_market"] == pytest.approx(0.75)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_fit_learned_source_session_weights_falls_back_when_samples_are_insufficient() -> None:
    history = _history_frame().iloc[:2].copy()
    fitted = fit_learned_source_session_weights(
        history=history,
        config=LearnedWeightingConfig(
            regularization_lambda=30.0,
            min_samples=3,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.05,
            fallback_mode="fixed",
        ),
    )

    assert fitted["valid"] is False
    assert fitted["reason"] == "insufficient_total_samples"


def test_build_walk_forward_learned_weighting_uses_fallback_when_history_is_unavailable() -> None:
    news_panel = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2026-04-03"], utc=True),
            "signal_day_return": [0.01],
        }
    )
    news = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "available_at": pd.to_datetime(["2026-04-03"], utc=True),
            "source_session_breakdown": [
                json.dumps(
                    {
                        "alpha::pre_market": {
                            "source_name": "alpha",
                            "session_bucket": "pre_market",
                            "base_weight_sum": 1.0,
                            "sentiment_weighted_sum": 0.8,
                            "relevance_weighted_sum": 0.7,
                            "headline_count": 1,
                            "article_count": 1,
                        }
                    }
                )
            ],
        }
    )
    fallback_sentiment = pd.Series([0.33], dtype=float)
    fallback_relevance = pd.Series([0.44], dtype=float)

    result = build_walk_forward_learned_weighting(
        news_panel=news_panel,
        news=news,
        config=LearnedWeightingConfig(
            regularization_lambda=30.0,
            min_samples=20,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.05,
            fallback_mode="fixed",
        ),
        fallback_sentiment=fallback_sentiment,
        fallback_relevance=fallback_relevance,
    )

    assert result.sentiment.iloc[0] == pytest.approx(0.33)
    assert result.relevance.iloc[0] == pytest.approx(0.44)
    assert result.summary["fit_day_count"] == 0
    assert result.summary["fallback_day_count"] == 1
    assert result.summary["fallback_reasons"]["insufficient_total_samples"] == 1
