from __future__ import annotations

import pandas as pd

from market_prediction_agent.data.normalizer import normalize_news


SENTIMENT_VARIANTS = {
    "sentiment_score_unweighted": 0.11,
    "sentiment_score_source_weighted": 0.12,
    "sentiment_score_session_weighted": 0.13,
    "sentiment_score_source_session_weighted": 0.14,
}
RELEVANCE_VARIANTS = {
    "relevance_score_unweighted": 0.71,
    "relevance_score_source_weighted": 0.72,
    "relevance_score_session_weighted": 0.73,
    "relevance_score_source_session_weighted": 0.74,
}


def test_normalize_news_falls_back_to_production_scores_for_weighted_variants() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "published_at": "2024-01-02T09:00:00Z",
                "available_at": "2024-01-02T09:30:00Z",
                "sentiment_score": 0.25,
                "relevance_score": 0.65,
                "headline_count": 3,
            }
        ]
    )

    normalized = normalize_news(frame)
    row = normalized.iloc[0]

    for column in SENTIMENT_VARIANTS:
        assert row[column] == row["sentiment_score"]
    for column in RELEVANCE_VARIANTS:
        assert row[column] == row["relevance_score"]


def test_normalize_news_preserves_explicit_weighted_variant_scores() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "published_at": "2024-01-02T09:00:00Z",
                "available_at": "2024-01-02T09:30:00Z",
                "sentiment_score": 0.25,
                "relevance_score": 0.65,
                "headline_count": 3,
                **SENTIMENT_VARIANTS,
                **RELEVANCE_VARIANTS,
            }
        ]
    )

    normalized = normalize_news(frame)
    row = normalized.iloc[0]

    for column, expected in SENTIMENT_VARIANTS.items():
        assert row[column] == expected
    for column, expected in RELEVANCE_VARIANTS.items():
        assert row[column] == expected
