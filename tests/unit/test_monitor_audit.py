from __future__ import annotations

import pandas as pd

from market_prediction_agent.audits.monitor_audit import build_monitor_audit
from market_prediction_agent.config import load_settings
from market_prediction_agent.pipeline import PipelineRunResult
from market_prediction_agent.schemas.validator import validate_payload


def test_monitor_audit_payload_matches_schema() -> None:
    settings = load_settings("config/default.yaml")
    result = PipelineRunResult(
        backtest_result={
            "backtest_id": "11111111-1111-4111-8111-111111111111",
            "config": {"start_date": "2024-01-01", "end_date": "2026-04-03", "horizon": "1d"},
            "aggregate_metrics": {"hit_rate_mean": 0.4, "ece_mean": 0.03},
            "cost_adjusted_metrics": {"information_ratio": 0.2},
            "portfolio_rule_analysis": {"selected_rule": {"strategy_name": "classified_directional"}},
            "pbo": 0.35,
            "cpcv": {"pbo_summary": {"label": "moderate_overfit_risk"}},
            "feature_catalog": [
                {
                    "feature": "news_sentiment_1d",
                    "feature_family": "news",
                    "data_source": "yahoo_finance_rss",
                    "domain": "news",
                    "missing_rate": 0.1,
                    "stale_rate": 0.0,
                }
            ],
            "feature_importance_summary": [
                {
                    "feature": "news_sentiment_1d",
                    "mean_abs_shap": 0.12,
                    "feature_family": "news",
                    "data_source": "yahoo_finance_rss",
                    "missing_rate": 0.1,
                    "stale_rate": 0.0,
                }
            ],
            "feature_family_importance_summary": [
                {
                    "feature_family": "news",
                    "feature_count": 4,
                    "mean_abs_shap": 0.12,
                    "mean_missing_rate": 0.1,
                    "mean_stale_rate": 0.0,
                    "data_sources": ["yahoo_finance_rss"],
                }
            ],
            "drift_monitor": {"max_psi": 0.1},
            "regime_monitor": {"current_regime": "low_vol"},
            "retraining_monitor": {"should_retrain": False, "trigger_count": 0, "triggers": []},
        },
        forecast_output={
            "forecast_id": "22222222-2222-4222-8222-222222222222",
            "generated_at": pd.Timestamp("2026-04-03", tz="UTC").isoformat(),
            "predictions": [],
        },
        evidence_bundle={
            "data_snapshot": {
                "source_metadata": {
                    "used_source": "fred_market_proxy",
                    "feature_sources": {
                        "news": {
                            "requested_source": "yahoo_finance_rss",
                            "used_source": "yahoo_finance_rss",
                            "requested_sources": ["yahoo_finance_rss", "google_news_rss"],
                            "used_sources": ["yahoo_finance_rss", "google_news_rss"],
                            "missing_rate": 0.1,
                            "stale_rate": 0.0,
                            "transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                            "utility_comparison": {
                                "baseline_1d": {"window_days": 1, "coverage": 0.4, "abs_ic": 0.02},
                                "best_aggregation_window": 5,
                                "best_aggregation_coverage": 0.8,
                                "best_decay_halflife": 3,
                                "best_decay_coverage": 0.85,
                                "best_feature_variant": "headline_count_1d",
                                "selected_weighting_mode": "fixed",
                                "weighting_mode_comparison": [
                                    {"mode": "unweighted", "variant": "unweighted_1d", "abs_ic": 0.02, "overlap_rate": 0.4},
                                    {"mode": "fixed", "variant": "source_session_weighted_1d", "abs_ic": 0.03, "overlap_rate": 0.5},
                                    {"mode": "learned", "variant": "learned_weighted_1d", "abs_ic": 0.04, "overlap_rate": 0.55},
                                ],
                                "learned_weighting": {
                                    "target": "abs_ic",
                                    "fit_day_count": 4,
                                    "fallback_day_count": 1,
                                    "fallback_rate": 0.2,
                                    "source_weights": [{"source_name": "google_news_rss", "mean_weight": 0.6}],
                                    "session_weights": [{"session_bucket": "pre_market", "mean_weight": 0.7}],
                                },
                                "multi_source_weighting_improvement": {
                                    "learned_abs_ic_delta_vs_fixed": 0.01,
                                    "learned_overlap_rate_delta_vs_fixed": 0.05,
                                },
                                "session_buckets": [{"session_bucket": "pre_market", "coverage": 0.2}],
                            },
                        },
                        "fundamental": {
                            "requested_source": "sec_companyfacts",
                            "used_source": "offline_fundamental_proxy",
                            "fallback_used": True,
                            "fallback_reason": "No usable companyfacts rows were available for the requested ticker set.",
                            "stale_rate": 0.0,
                            "transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                        },
                        "sector": {"used_source": "static_sector_map", "stale_rate": 0.0},
                    },
                    "fallback_used": False,
                    "fallback_reason": None,
                    "dummy_mode": None,
                    "public_data_transport": {"origins": ["network"], "cache_used": False, "snapshot_used": False},
                    "macro_public_data_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                }
            }
        },
        risk_review={},
        report_payload={"generated_at": pd.Timestamp("2026-04-03", tz="UTC").isoformat()},
        paper_trading_batch={
            "batch_id": "33333333-3333-4333-8333-333333333333",
            "approval": "MANUAL_REVIEW_REQUIRED",
            "metrics": {
                "avg_round_trip_cost_bps": 20.0,
                "avg_participation_rate": 0.01,
                "liquidity_capped_trades_this_run": 1,
                "liquidity_blocked_trades_this_run": 0,
            },
            "execution_diagnostics": {
                "fill_rate": 0.9,
                "partial_fill_rate": 0.2,
                "missed_trade_rate": 0.1,
                "realized_vs_intended_exposure": 0.85,
                "execution_cost_drag": 0.003,
                "gap_slippage_bps": 8.0,
            },
        },
        weekly_review={
            "execution_diagnostics": {
                "fill_rate": 0.9,
                "partial_fill_rate": 0.2,
                "missed_trade_rate": 0.1,
                "realized_vs_intended_exposure": 0.85,
                "execution_cost_drag": 0.003,
                "gap_slippage_bps": 8.0,
            }
        },
        retraining_event=None,
    )
    payload = build_monitor_audit(
        settings=settings,
        result=result,
        dataset_type="public_real_market",
        tickers=["SPY", "QQQ"],
        macro_source="fred_csv",
        notes=["x"],
    )
    validate_payload("monitor_audit", payload)
    assert payload["data_sources"]["ohlcv_source"] == "fred_market_proxy"
    assert payload["data_sources"]["proxy_ohlcv_used"] is True
    assert "ohlcv_transport" in payload["data_sources"]
    assert "feature_sources" in payload["data_sources"]
    assert payload["data_sources"]["feature_sources"]["fundamental"]["requested_source"] == "sec_companyfacts"
    assert payload["data_sources"]["feature_sources"]["fundamental"]["fallback_used"] is True
    assert payload["data_sources"]["feature_sources"]["news"]["utility_comparison"]["best_feature_variant"] == "headline_count_1d"
    assert payload["data_sources"]["feature_sources"]["news"]["used_sources"] == ["yahoo_finance_rss", "google_news_rss"]
    assert any("News weighting modes:" in note for note in payload["notes"])
    assert any("Learned weighting summary:" in note for note in payload["notes"])
    assert "portfolio_rule_analysis" in payload["backtest"]
    assert payload["feature_lineage"]["feature_catalog"]
    assert "execution_diagnostics" in payload["paper_trading_summary"]
