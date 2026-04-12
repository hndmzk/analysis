from __future__ import annotations

import json
import os
from pathlib import Path
import time
import shutil
from uuid import uuid4

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.reporting.audit_reports import (
    build_audit_report,
    latest_artifact_path,
    render_audit_report_markdown,
    resolve_artifact_bundle,
)
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.utils.paths import resolve_repo_path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-audit-reports" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _monitor_audit_payload() -> dict[str, object]:
    return {
        "audit_id": "11111111-1111-1111-1111-111111111111",
        "dataset_type": "public_real_market",
        "generated_at": "2026-04-04T00:00:00+00:00",
        "window": {"start_date": "2025-01-01", "end_date": "2026-04-03", "horizon": "1d"},
        "data_sources": {
            "source_mode": "live",
            "ohlcv_source": "yahoo_chart",
            "proxy_ohlcv_used": False,
            "macro_source": "fred_csv",
            "feature_sources": {
                "news": {
                    "requested_source": "yahoo_finance_rss",
                    "used_source": "yahoo_finance_rss",
                    "missing_rate": 0.08,
                    "stale_rate": 0.0,
                    "transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
                    "utility_comparison": {
                        "baseline_1d": {"window_days": 1, "coverage": 0.40, "abs_ic": 0.02},
                        "best_aggregation_window": 5,
                        "best_aggregation_coverage": 0.82,
                        "best_aggregation_abs_ic": 0.05,
                        "best_decay_halflife": 3,
                        "best_decay_coverage": 0.88,
                        "best_decay_abs_ic": 0.04,
                        "best_feature_variant": "headline_count_1d",
                        "selected_weighting_mode": "fixed",
                        "weighting_mode_comparison": [
                            {"mode": "unweighted", "variant": "unweighted_1d", "abs_ic": 0.02, "overlap_rate": 0.40},
                            {"mode": "fixed", "variant": "source_session_weighted_1d", "abs_ic": 0.03, "overlap_rate": 0.45},
                            {"mode": "learned", "variant": "learned_weighted_1d", "abs_ic": 0.04, "overlap_rate": 0.48},
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
                            "learned_overlap_rate_delta_vs_fixed": 0.03,
                        },
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
            "ohlcv_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
            "macro_transport": {"origins": ["cache"], "cache_used": True, "snapshot_used": False},
        },
        "universe": ["SPY", "QQQ", "GLD"],
        "backtest": {
            "backtest_id": "22222222-2222-2222-2222-222222222222",
            "hit_rate_mean": 0.44,
            "ece_mean": 0.07,
            "information_ratio": 0.56,
            "pbo": 0.42,
            "pbo_summary": {"label": "high_overfit_risk"},
            "cluster_adjusted_pbo": 0.0,
            "cluster_adjusted_pbo_summary": {"label": "low_overfit_risk"},
            "portfolio_rule_analysis": {"selected_rule": {"strategy_name": "classified_directional"}},
            "feature_importance_summary": [
                {
                    "feature": "news_sentiment_1d",
                    "mean_abs_shap": 0.14,
                    "feature_family": "news",
                    "data_source": "yahoo_finance_rss",
                    "missing_rate": 0.08,
                    "stale_rate": 0.0,
                }
            ],
            "feature_family_importance_summary": [
                {
                    "feature_family": "news",
                    "feature_count": 4,
                    "mean_abs_shap": 0.14,
                    "mean_missing_rate": 0.08,
                    "mean_stale_rate": 0.0,
                    "data_sources": ["yahoo_finance_rss"],
                }
            ],
        },
        "feature_lineage": {
            "feature_catalog": [
                {
                    "feature": "news_sentiment_1d",
                    "feature_family": "news",
                    "data_source": "yahoo_finance_rss",
                    "domain": "news",
                    "missing_rate": 0.08,
                    "stale_rate": 0.0,
                }
            ],
            "feature_importance_summary": [
                {
                    "feature": "news_sentiment_1d",
                    "mean_abs_shap": 0.14,
                    "feature_family": "news",
                    "data_source": "yahoo_finance_rss",
                    "missing_rate": 0.08,
                    "stale_rate": 0.0,
                }
            ],
            "feature_family_importance_summary": [
                {
                    "feature_family": "news",
                    "feature_count": 4,
                    "mean_abs_shap": 0.14,
                    "mean_missing_rate": 0.08,
                    "mean_stale_rate": 0.0,
                    "data_sources": ["yahoo_finance_rss"],
                }
            ],
        },
        "drift_monitor": {
            "max_psi": 0.31,
            "supplementary_analysis": {"primary_cause": "regime_shift_likely"},
        },
        "regime_monitor": {
            "current_regime": "transition",
            "dominant_recent_regime": "low_vol",
            "state_probability": 0.58,
            "transition_rate": 0.0,
        },
        "retraining_monitor": {
            "base_should_retrain": True,
            "should_retrain": False,
            "base_trigger_names": ["regime_shift", "transition_regime"],
            "effective_trigger_names": [],
            "suppressed_trigger_names": ["regime_shift", "transition_regime"],
            "policy_decision": "watch_only",
            "policy_notes": ["Regime shift is kept as watch-only unless feature drift co-occurs in the same run."],
            "observations": ["Raw watch-only regime_shift remains monitored."],
            "candidate_level_pbo": 0.42,
            "candidate_level_pbo_label": "high_overfit_risk",
            "pbo": 0.0,
            "pbo_label": "low_overfit_risk",
            "drift_signal": {"trigger_families": ["macro"], "proxy_sensitive_profile": "mixed"},
            "calibration_signal": {"ece_breach_ratio": 0.0, "calibration_gap_breach_ratio": 0.0},
            "regime_signal": {
                "transition_profile": "watch_transition",
                "stable_transition": False,
                "unstable_transition": False,
            },
        },
        "paper_trading_summary": {
            "batch_id": "33333333-3333-3333-3333-333333333333",
            "approval": "MANUAL_REVIEW_REQUIRED",
            "avg_round_trip_cost_bps": 15.0,
            "avg_participation_rate": 0.03,
            "liquidity_capped_trades_this_run": 1,
            "liquidity_blocked_trades_this_run": 0,
            "execution_diagnostics": {
                "fill_rate": 0.75,
                "partial_fill_rate": 0.2,
                "missed_trade_rate": 0.05,
                "realized_vs_intended_exposure": 0.74,
                "execution_cost_drag": 0.003,
                "execution_cost_drag_bps": 30.0,
                "gap_slippage_bps": 4.0,
            },
            "weekly_execution_diagnostics": {
                "fill_rate": 0.78,
                "partial_fill_rate": 0.18,
                "missed_trade_rate": 0.04,
                "realized_vs_intended_exposure": 0.76,
                "execution_cost_drag": 0.0025,
                "execution_cost_drag_bps": 25.0,
                "gap_slippage_bps": 3.5,
            },
        },
        "notes": [
            "suite_profile=full_light",
            "suite_profile_role=routine_monitoring",
            "suite_as_of_date=2026-04-03",
            "Execution diagnostics refreshed from the latest pipeline run.",
        ],
    }


def _backtest_payload() -> dict[str, object]:
    return {
        "backtest_id": "22222222-2222-2222-2222-222222222222",
        "completed_at": "2026-04-04T00:00:00+00:00",
        "config": {"model_name": "lightgbm_multiclass_calibrated"},
        "aggregate_metrics": {"hit_rate_mean": 0.44, "ece_mean": 0.07},
        "cost_adjusted_metrics": {
            "information_ratio": 0.56,
            "selection_stability": 0.89,
            "avg_daily_turnover": 0.02,
            "cost_drag_annual_return": 0.003,
        },
        "portfolio_rule_analysis": {"selected_rule": {"strategy_name": "classified_directional"}},
        "feature_catalog": [
            {
                "feature": "news_sentiment_1d",
                "feature_family": "news",
                "data_source": "yahoo_finance_rss",
                "domain": "news",
                "missing_rate": 0.08,
                "stale_rate": 0.0,
            }
        ],
        "feature_importance_summary": [
            {
                "feature": "news_sentiment_1d",
                "mean_abs_shap": 0.14,
                "feature_family": "news",
                "data_source": "yahoo_finance_rss",
                "missing_rate": 0.08,
                "stale_rate": 0.0,
            }
        ],
        "feature_family_importance_summary": [
            {
                "feature_family": "news",
                "feature_count": 4,
                "mean_abs_shap": 0.14,
                "mean_missing_rate": 0.08,
                "mean_stale_rate": 0.0,
                "data_sources": ["yahoo_finance_rss"],
            }
        ],
        "retraining_monitor": _monitor_audit_payload()["retraining_monitor"],
        "regime_monitor": _monitor_audit_payload()["regime_monitor"],
        "drift_monitor": _monitor_audit_payload()["drift_monitor"],
    }


def _paper_payload() -> dict[str, object]:
    return {
        "batch_id": "33333333-3333-3333-3333-333333333333",
        "forecast_id": "44444444-4444-4444-4444-444444444444",
        "created_at": "2026-04-04T00:00:00+00:00",
        "forecast_date": "2026-04-03",
        "week_id": "2026-W14",
        "ledger_path": "storage/outputs/paper_trading/trade_ledger.parquet",
        "approval": "MANUAL_REVIEW_REQUIRED",
        "should_retrain": False,
        "metrics": {"new_trades": 3},
        "execution_diagnostics": {
            "fill_rate": 0.75,
            "partial_fill_rate": 0.2,
            "missed_trade_rate": 0.05,
            "realized_vs_intended_exposure": 0.74,
            "execution_cost_drag": 0.003,
            "execution_cost_drag_bps": 30.0,
            "gap_slippage_bps": 4.0,
        },
        "trades": [],
    }


def _weekly_payload() -> dict[str, object]:
    return {
        "review_id": "55555555-5555-5555-5555-555555555555",
        "week_id": "2026-W14",
        "generated_at": "2026-04-04T00:00:00+00:00",
        "window_start": "2026-03-30",
        "window_end": "2026-04-03",
        "summary": "Weekly execution review.",
        "metrics": {"total_trades": 12},
        "execution_diagnostics": {
            "fill_rate": 0.8,
            "partial_fill_rate": 0.15,
            "missed_trade_rate": 0.05,
            "realized_vs_intended_exposure": 0.79,
            "execution_cost_drag": 0.002,
            "execution_cost_drag_bps": 20.0,
            "gap_slippage_bps": 3.0,
        },
        "approval_breakdown": {"MANUAL_REVIEW_REQUIRED": 1},
    }


def _suite_payload() -> dict[str, object]:
    return {
        "suite_id": "66666666-6666-6666-6666-666666666666",
        "generated_at": "2026-04-04T00:00:00+00:00",
        "dataset_type": "public_real_market",
        "profile_name": "full_light",
        "profile_role": "routine_monitoring",
        "analysis_mode": "live_suite",
        "as_of_dates": ["2026-03-27", "2026-04-03"],
        "ticker_sets": [["SPY", "QQQ", "GLD"]],
        "run_count": 2,
        "cpcv_max_splits": 1,
        "runs": [
            {
                "audit_id": "11111111-1111-1111-1111-111111111111",
                "backtest_id": "22222222-2222-2222-2222-222222222222",
                "as_of_date": "2026-04-03",
                "ticker_set": "SPY,QQQ,GLD",
                "information_ratio": 0.56,
                "pbo": 0.42,
                "pbo_label": "high_overfit_risk",
                "cluster_adjusted_pbo": 0.0,
                "cluster_adjusted_pbo_label": "low_overfit_risk",
                "candidate_pbo_competition_dominated": True,
                "pbo_competition_dominated": False,
                "candidate_pbo_dominant_axis": "threshold",
                "pbo_dominant_axis": "threshold",
                "candidate_pbo_dominant_value": "0.35",
                "pbo_dominant_value": "0.30-0.35",
                "candidate_pbo_close_competition_ratio": 1.0,
                "pbo_close_competition_ratio": 0.5,
                "selection_stability": 0.89,
                "hit_rate_mean": 0.44,
                "current_regime": "transition",
                "dominant_recent_regime": "low_vol",
                "regime_shift_flag": True,
                "state_probability": 0.58,
                "transition_rate": 0.0,
                "ohlcv_source": "yahoo_chart",
                "macro_source": "fred_csv",
                "ohlcv_transport_origin": "cache",
                "macro_transport_origin": "cache",
                "news_used_source": "yahoo_finance_rss",
                "news_requested_source": "yahoo_finance_rss",
                "news_fallback_used": False,
                "news_transport_origin": "cache",
                "news_feature_missing_rate": 0.12,
                "news_feature_coverage": 0.88,
                "news_feature_stale_rate": 0.0,
                "transition_profile": "watch_transition",
                "transition_history_matches": 2,
                "transition_observation_count": 2,
                "transition_span_business_days": 7,
                "transition_persistent": True,
                "stable_transition": False,
                "unstable_transition": False,
                "immediate_transition": False,
                "state_probability_bucket": "mid",
                "transition_rate_bucket": "zero",
                "drift_weighted_score": 1.4,
                "drift_pre_suppression_weighted_score": 1.4,
                "drift_weighted_threshold": 2.2,
                "drift_low_vol_threshold": 2.0,
                "drift_immediate_threshold": 3.0,
                "drift_trigger_feature_count": 2,
                "drift_trigger_family_count": 1,
                "drift_pre_suppression_trigger_feature_count": 2,
                "drift_pre_suppression_trigger_family_count": 1,
                "drift_trigger_families": ["macro"],
                "drift_pre_suppression_trigger_families": ["macro"],
                "drift_proxy_sensitive_profile": "mixed",
                "drift_proxy_sensitive_trigger_feature_count": 0,
                "drift_non_proxy_trigger_feature_count": 2,
                "drift_pre_suppression_proxy_sensitive_trigger_feature_count": 0,
                "drift_pre_suppression_non_proxy_trigger_feature_count": 2,
                "drift_history_matches": 2,
                "drift_span_business_days": 7,
                "drift_suppressed_families": [],
                "drift_stable_transition_suppressed_families": [],
                "drift_primary_cause": "regime_shift_likely",
                "drift_family_persistence_would_suppress": False,
                "drift_family_persistence_counterfactual_score": 1.4,
                "drift_threshold_delta_to_suppress": 0.8,
                "drift_low_vol_threshold_would_suppress": False,
                "drift_stable_transition_suppression_would_suppress": False,
                "drift_stable_transition_counterfactual_score": 1.4,
                "ece_breach_count": 0,
                "ece_breach_ratio": 0.0,
                "calibration_gap_breach_count": 0,
                "calibration_gap_breach_ratio": 0.0,
                "ece_fold_persistent": False,
                "calibration_gap_fold_persistent": False,
                "ece_run_persistent": False,
                "calibration_gap_run_persistent": False,
                "ece_history_matches": 0,
                "calibration_gap_history_matches": 0,
                "ece_observation_count": 0,
                "calibration_gap_observation_count": 0,
                "ece_span_business_days": 0,
                "calibration_gap_span_business_days": 0,
                "base_should_retrain": True,
                "should_retrain": False,
                "policy_decision": "watch_only",
                "base_trigger_names": ["regime_shift"],
                "trigger_names": [],
                "suppressed_trigger_names": ["regime_shift"],
                "policy_notes": ["Raw watch-only regime_shift remains monitored without changing policy defaults."]
            }
        ],
        "distribution_summary": {
            "information_ratio": {"count": 1, "mean": 0.56, "median": 0.56, "min": 0.56, "max": 0.56},
            "pbo": {"count": 1, "mean": 0.42, "median": 0.42, "min": 0.42, "max": 0.42},
            "cluster_adjusted_pbo": {"count": 1, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "selection_stability": {"count": 1, "mean": 0.89, "median": 0.89, "min": 0.89, "max": 0.89},
            "news_feature_coverage": {"count": 1, "mean": 0.88, "median": 0.88, "min": 0.88, "max": 0.88},
            "news_feature_missing_rate": {"count": 1, "mean": 0.12, "median": 0.12, "min": 0.12, "max": 0.12},
            "news_feature_staleness": {"count": 1, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "news_used_source_counts": {"yahoo_finance_rss": 1},
            "news_transport_origin_counts": {"cache": 1},
            "news_fallback_rate": 0.0,
            "news_coverage_analysis": {
                "lookback_windows": [
                    {"window_days": 1, "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.03}},
                    {"window_days": 3, "coverage": {"count": 1, "mean": 0.93}, "abs_ic": {"count": 1, "mean": 0.05}},
                ],
                "decay_halflives": [
                    {"halflife_days": 3, "coverage": {"count": 1, "mean": 0.95}, "abs_ic": {"count": 1, "mean": 0.04}}
                ],
                "feature_variants": [
                    {"variant": "headline_count_1d", "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.03}}
                ],
                "weighting_mode_comparison": [
                    {"mode": "unweighted", "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.03}},
                    {"mode": "fixed", "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.04}},
                    {"mode": "learned", "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.05}},
                ],
                "best_aggregation_window_counts": {"3": 1},
                "best_decay_halflife_counts": {"3": 1},
                "best_feature_variant_counts": {"headline_count_1d": 1},
                "best_aggregation_coverage": {"count": 1, "mean": 0.93},
                "best_aggregation_abs_ic": {"count": 1, "mean": 0.05},
                "best_decay_coverage": {"count": 1, "mean": 0.95},
                "best_decay_abs_ic": {"count": 1, "mean": 0.04},
                "best_feature_variant_coverage": {"count": 1, "mean": 0.88},
                "best_feature_variant_abs_ic": {"count": 1, "mean": 0.03},
                "session_buckets": [
                    {"session_bucket": "pre_market", "coverage": {"count": 1, "mean": 0.44}, "abs_ic": {"count": 1, "mean": 0.02}}
                ],
                "source_diversity_buckets": [
                    {"diversity_bucket": "single_source", "coverage": {"count": 1, "mean": 0.88}, "abs_ic": {"count": 1, "mean": 0.03}}
                ],
                "learned_weighting": {
                    "fit_day_count": {"count": 1, "mean": 4.0},
                    "fallback_rate": {"count": 1, "mean": 0.2},
                    "source_weights": [
                        {"source_name": "google_news_rss", "mean_weight": {"count": 1, "mean": 0.6}}
                    ],
                    "session_weights": [
                        {"session_bucket": "pre_market", "mean_weight": {"count": 1, "mean": 0.7}}
                    ],
                },
                "aggregation_improves_coverage_rate": 1.0,
                "aggregation_improves_utility_rate": 1.0,
                "decay_improves_coverage_rate": 1.0,
                "decay_improves_utility_rate": 1.0
            },
            "base_retraining_rate": 1.0,
            "retraining_rate": 0.0,
            "base_trigger_counts": {"regime_shift": 1},
            "effective_trigger_counts": {},
            "policy_decision_counts": {"watch_only": 1},
            "pbo_competition_dominated_rate": 0.0,
            "candidate_pbo_competition_dominated_rate": 1.0,
            "pbo_dominant_axis_counts": {"threshold": 1},
            "candidate_pbo_dominant_axis_counts": {"threshold": 1},
            "drift_dominated_analysis": {
                "raw_run_count": 1,
                "effective_run_count": 0,
                "trigger_family_counts": {"macro": 1},
                "current_regime_counts": {"transition": 1},
                "proxy_sensitive_profile_counts": {"mixed": 1},
                "ohlcv_transport_origin_counts": {"cache": 1},
                "macro_transport_origin_counts": {"cache": 1},
                "ohlcv_source_counts": {"yahoo_chart": 1}
            },
            "calibration_dominated_analysis": {
                "run_count": 0,
                "rate": 0.0,
                "fold_persistence_counts": {},
                "run_persistence_counts": {},
                "ticker_set_counts": {},
                "ohlcv_transport_origin_counts": {},
                "macro_transport_origin_counts": {},
                "ohlcv_source_counts": {},
                "current_regime_counts": {},
                "dominant_recent_regime_counts": {},
                "trigger_metric_counts": {}
            },
            "regime_dominated_analysis": {
                "raw_run_count": 1,
                "effective_run_count": 0,
                "base_trigger_counts": {"regime_shift": 1},
                "effective_trigger_counts": {},
                "current_regime_counts": {"transition": 1},
                "transition_profile_counts": {"watch_transition": 1},
                "state_probability_bucket_counts": {"mid": 1},
                "transition_rate_bucket_counts": {"zero": 1},
                "co_trigger_family_counts": {"macro": 1}
            }
        }
    }


def _write_bundle_artifacts(storage_root: Path) -> tuple[Path, Path]:
    audit_path = _write_json(
        storage_root / "outputs" / "monitor_audits" / "public_real_market" / "2026-04-04" / "audit.json",
        _monitor_audit_payload(),
    )
    _write_json(storage_root / "outputs" / "backtests" / "backtest.json", _backtest_payload())
    _write_json(storage_root / "outputs" / "paper_trading" / "2026-04-04" / "paper.json", _paper_payload())
    _write_json(storage_root / "outputs" / "weekly_reviews" / "2026-W14" / "weekly.json", _weekly_payload())
    suite_path = _write_json(
        storage_root / "outputs" / "monitor_audit_suites" / "public_real_market" / "2026-04-04" / "suite.json",
        _suite_payload(),
    )
    return audit_path, suite_path


def test_render_markdown_from_monitor_audit() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(load_settings(), data={"storage_path": str(temp_dir / "storage")})
        audit_path, _ = _write_bundle_artifacts(Path(settings.data.storage_path))
        bundle = resolve_artifact_bundle(settings, primary_path=audit_path)
        report = build_audit_report(bundle)
        markdown = render_audit_report_markdown(report)
        validate_payload("audit_report", report)
        assert "## Executive Summary" in markdown
        assert "## Retraining Decision" in markdown
        assert "Raw signal triggers:" in markdown
        assert "Effective triggers:" in markdown
        assert "candidate-level PBO" in markdown
        assert "cluster-adjusted PBO" in markdown
        assert "feature_family_contribution" in markdown
        assert "feature_sources" in markdown
        assert "news_utility_comparison" in markdown
        assert "news_weighting_summary" in markdown
        assert "yahoo_finance_rss" in markdown
        assert "sec_companyfacts" in markdown
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_render_markdown_from_suite_artifact() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(load_settings(), data={"storage_path": str(temp_dir / "storage")})
        _, suite_path = _write_bundle_artifacts(Path(settings.data.storage_path))
        bundle = resolve_artifact_bundle(settings, primary_path=suite_path)
        report = build_audit_report(bundle)
        markdown = render_audit_report_markdown(report)
        validate_payload("audit_report", report)
        assert report["report_type"] == "monitor_audit_suite_report"
        assert "## Drift / Calibration / Regime Review" in markdown
        assert "drift_dominated_analysis" in markdown
        assert "calibration_dominated_analysis" in markdown
        assert "regime_dominated_analysis" in markdown
        assert "feature_family_contribution" in markdown
        assert report["data_sources_transport"]["feature_sources"]["news"]["coverage_analysis"]["lookback_windows"]
        assert report["data_sources_transport"]["feature_sources"]["news"]["utility_comparison"]["best_feature_variant_counts"]["headline_count_1d"] == 1
        assert "session_buckets" in report["data_sources_transport"]["feature_sources"]["news"]["utility_comparison"]
        assert "weighting_summary" in report["data_sources_transport"]["feature_sources"]["news"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_latest_artifact_resolution() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(load_settings(), data={"storage_path": str(temp_dir / "storage")})
        older = _monitor_audit_payload()
        older["generated_at"] = "2026-04-03T00:00:00+00:00"
        newer = _monitor_audit_payload()
        newer["generated_at"] = "2026-04-04T00:00:00+00:00"
        older_path = _write_json(Path(settings.data.storage_path) / "outputs" / "monitor_audits" / "old.json", older)
        newer_path = _write_json(Path(settings.data.storage_path) / "outputs" / "monitor_audits" / "new.json", newer)
        now = time.time()
        os.utime(older_path, (now - 10, now - 10))
        os.utime(newer_path, (now, now))
        resolved = latest_artifact_path(settings, "monitor_audit")
        assert resolved == newer_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_watch_only_and_effective_are_separated_in_json() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(load_settings(), data={"storage_path": str(temp_dir / "storage")})
        audit_path, _ = _write_bundle_artifacts(Path(settings.data.storage_path))
        bundle = resolve_artifact_bundle(settings, primary_path=audit_path)
        report = build_audit_report(bundle)
        retraining = report["retraining_decision"]
        watch_only = report["watch_only_findings"]
        assert retraining["base_trigger_names"] == ["regime_shift", "transition_regime"]
        assert retraining["effective_trigger_names"] == []
        assert "Raw-only triggers: regime_shift, transition_regime" in watch_only["findings"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_candidate_and_cluster_adjusted_pbo_are_both_present() -> None:
    temp_dir = _workspace_temp_dir()
    try:
        settings = update_settings(load_settings(), data={"storage_path": str(temp_dir / "storage")})
        audit_path, suite_path = _write_bundle_artifacts(Path(settings.data.storage_path))
        single_report = build_audit_report(resolve_artifact_bundle(settings, primary_path=audit_path))
        suite_report = build_audit_report(resolve_artifact_bundle(settings, primary_path=suite_path))
        assert single_report["model_portfolio_performance"]["candidate_level_pbo"] == 0.42
        assert single_report["model_portfolio_performance"]["cluster_adjusted_pbo"] == 0.0
        assert suite_report["model_portfolio_performance"]["candidate_level_pbo"]["mean"] == 0.42
        assert suite_report["model_portfolio_performance"]["cluster_adjusted_pbo"]["mean"] == 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
