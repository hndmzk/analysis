from __future__ import annotations

import pandas as pd

from market_prediction_agent.config import LearnedWeightingConfig
from market_prediction_agent.evaluation.news_analysis import build_news_feature_utility_comparison


def test_news_feature_utility_comparison_detects_aggregation_coverage_improvement() -> None:
    ohlcv = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "timestamp_utc": pd.date_range("2026-04-01", periods=6, freq="B", tz="UTC"),
            "close": [100.0, 101.0, 100.5, 102.0, 101.5, 103.0],
        }
    )
    news = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "available_at": pd.to_datetime(["2026-04-02", "2026-04-07", "2026-04-08"], utc=True),
            "sentiment_score": [0.6, -0.2, 0.1],
            "relevance_score": [0.8, 0.7, 0.8],
            "headline_count": [2, 1, 1],
            "mapping_confidence": [0.9, 0.8, 0.9],
            "novelty_score": [0.9, 0.5, 0.7],
            "source_diversity": [1.0, 1.0, 2.0],
            "source_count": [1.0, 1.0, 2.0],
            "session_bucket": ["pre_market", "regular", "post_market"],
            "stale_data_flag": [False, False, False],
        }
    )

    analysis = build_news_feature_utility_comparison(ohlcv=ohlcv, news=news)

    assert analysis["baseline_1d"]["coverage"] < analysis["best_aggregation_coverage"]
    assert analysis["aggregation_improves_coverage"] is True
    assert analysis["best_feature_variant"] in {"sentiment_1d", "headline_count_1d", "novelty_5d", "source_diversity_5d"}
    assert analysis["best_weighted_variant"] in {
        "unweighted_1d",
        "source_weighted_1d",
        "session_weighted_1d",
        "source_session_weighted_1d",
        "learned_weighted_1d",
    }
    assert len(analysis["feature_variants"]) == 4
    assert len(analysis["weighted_variants"]) == 5
    assert {item["mode"] for item in analysis["weighting_mode_comparison"]} == {"unweighted", "fixed", "learned"}
    assert "overlap_rate" in analysis["baseline_1d"]
    session_buckets = {item["session_bucket"]: item for item in analysis["session_buckets"]}
    assert "pre_market" in session_buckets
    assert "post_market" in session_buckets
    diversity_buckets = {item["diversity_bucket"]: item for item in analysis["source_diversity_buckets"]}
    assert diversity_buckets["multi_source"]["coverage"] > 0.0
    assert "source_advantage_analysis" in analysis
    assert "mixed_session_conditions" in analysis
    assert "learned_weighting" in analysis
    assert "multi_source_weighting_improvement" in analysis


def test_news_feature_utility_comparison_reports_weighted_source_and_session_advantage() -> None:
    ohlcv = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 6,
            "timestamp_utc": pd.date_range("2026-04-01", periods=6, freq="B", tz="UTC"),
            "close": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0],
        }
    )
    news = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "available_at": pd.to_datetime(["2026-04-02", "2026-04-03", "2026-04-08"], utc=True),
            "published_at": pd.to_datetime(
                ["2026-04-01T13:00:00Z", "2026-04-02T21:30:00Z", "2026-04-07T12:00:00Z"],
                utc=True,
            ),
            "sentiment_score": [0.1, 0.2, 0.0],
            "sentiment_score_unweighted": [0.1, 0.2, 0.0],
            "sentiment_score_source_weighted": [0.1, 0.6, -0.1],
            "sentiment_score_session_weighted": [0.4, 0.2, -0.2],
            "sentiment_score_source_session_weighted": [0.5, 0.7, -0.2],
            "relevance_score": [0.7, 0.8, 0.4],
            "relevance_score_unweighted": [0.7, 0.8, 0.4],
            "relevance_score_source_weighted": [0.7, 0.9, 0.3],
            "relevance_score_session_weighted": [0.9, 0.8, 0.3],
            "relevance_score_source_session_weighted": [0.9, 0.95, 0.25],
            "headline_count": [1, 1, 1],
            "mapping_confidence": [1.0, 1.0, 1.0],
            "novelty_score": [0.9, 0.8, 0.4],
            "source_diversity": [1.0, 1.0, 2.0],
            "source_count": [1.0, 1.0, 2.0],
            "session_bucket": ["mixed", "pre_market", "mixed"],
            "source_mix": ["google_news_rss", "google_news_rss", "google_news_rss|yahoo_finance_rss"],
            "stale_data_flag": [False, False, False],
        }
    )

    analysis = build_news_feature_utility_comparison(ohlcv=ohlcv, news=news)

    weighted_variants = {item["variant"]: item for item in analysis["weighted_variants"]}
    best_variant = max(weighted_variants.values(), key=lambda item: float(item["abs_ic"]))
    assert analysis["best_weighted_variant"] == best_variant["variant"]
    assert analysis["best_weighted_variant_abs_ic"] == best_variant["abs_ic"]
    assert analysis["source_advantage_analysis"]["google_single_source_present"] is True
    assert analysis["source_advantage_analysis"]["dominant_reason"] in {
        "overlap_advantage",
        "coverage_advantage",
        "multi_source_dilution",
        "mixed_or_no_clear_advantage",
    }
    assert "mixed_session_present" in analysis["mixed_session_conditions"]


def test_news_feature_utility_comparison_supports_weighting_mode_switch_and_learned_summary() -> None:
    ohlcv = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 7,
            "timestamp_utc": pd.date_range("2026-04-01", periods=7, freq="B", tz="UTC"),
            "close": [100.0, 101.0, 102.5, 101.5, 103.0, 104.5, 105.0],
        }
    )
    news = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "available_at": pd.to_datetime(["2026-04-02", "2026-04-03", "2026-04-06", "2026-04-07"], utc=True),
            "sentiment_score": [0.3, 0.1, 0.4, -0.2],
            "sentiment_score_unweighted": [0.3, 0.1, 0.4, -0.2],
            "sentiment_score_source_session_weighted": [0.25, 0.05, 0.45, -0.1],
            "relevance_score": [0.8, 0.7, 0.9, 0.6],
            "relevance_score_unweighted": [0.8, 0.7, 0.9, 0.6],
            "relevance_score_source_session_weighted": [0.75, 0.65, 0.95, 0.55],
            "headline_count": [2, 1, 2, 1],
            "mapping_confidence": [1.0, 1.0, 1.0, 1.0],
            "novelty_score": [0.9, 0.8, 0.7, 0.4],
            "source_diversity": [2.0, 1.0, 2.0, 1.0],
            "source_count": [2.0, 1.0, 2.0, 1.0],
            "source_mix": [
                "alpha|beta",
                "alpha",
                "alpha|beta",
                "beta",
            ],
            "session_bucket": ["mixed", "pre_market", "mixed", "post_market"],
            "source_session_breakdown": [
                '{"alpha::pre_market":{"source_name":"alpha","session_bucket":"pre_market","base_weight_sum":1.0,"sentiment_weighted_sum":0.5,"relevance_weighted_sum":0.8,"headline_count":1,"article_count":1},"beta::post_market":{"source_name":"beta","session_bucket":"post_market","base_weight_sum":0.7,"sentiment_weighted_sum":0.1,"relevance_weighted_sum":0.49,"headline_count":1,"article_count":1}}',
                '{"alpha::pre_market":{"source_name":"alpha","session_bucket":"pre_market","base_weight_sum":0.7,"sentiment_weighted_sum":0.07,"relevance_weighted_sum":0.49,"headline_count":1,"article_count":1}}',
                '{"alpha::pre_market":{"source_name":"alpha","session_bucket":"pre_market","base_weight_sum":0.9,"sentiment_weighted_sum":0.54,"relevance_weighted_sum":0.81,"headline_count":1,"article_count":1},"beta::post_market":{"source_name":"beta","session_bucket":"post_market","base_weight_sum":0.8,"sentiment_weighted_sum":0.08,"relevance_weighted_sum":0.64,"headline_count":1,"article_count":1}}',
                '{"beta::post_market":{"source_name":"beta","session_bucket":"post_market","base_weight_sum":0.6,"sentiment_weighted_sum":-0.12,"relevance_weighted_sum":0.36,"headline_count":1,"article_count":1}}',
            ],
            "stale_data_flag": [False, False, False, False],
        }
    )

    analysis = build_news_feature_utility_comparison(
        ohlcv=ohlcv,
        news=news,
        weighting_mode="learned",
        learned_weighting=LearnedWeightingConfig(
            regularization_lambda=5.0,
            min_samples=1,
            lookback_days=252,
            target="abs_ic",
            min_weight=0.05,
            fallback_mode="fixed",
        ),
    )

    assert analysis["selected_weighting_mode"] == "learned"
    assert analysis["selected_weighting_variant"] == "learned_weighted_1d"
    learned_summary = analysis["learned_weighting"]
    assert learned_summary["target"] == "abs_ic"
    assert learned_summary["source_weights"]
    assert learned_summary["session_weights"]
    mode_comparison = {item["mode"]: item for item in analysis["weighting_mode_comparison"]}
    assert set(mode_comparison) == {"unweighted", "fixed", "learned"}
