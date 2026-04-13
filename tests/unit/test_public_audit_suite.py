from __future__ import annotations

from pathlib import Path
import json
import shutil
from uuid import uuid4

import pytest

from market_prediction_agent.audits.public_audit_suite import (
    build_public_audit_suite,
    parse_as_of_dates,
    parse_ticker_sets,
    replay_public_audit_suite,
    resolve_public_audit_profile,
)
from market_prediction_agent.config import load_settings, update_settings


def test_public_audit_suite_helpers_build_distribution_summary() -> None:
    payload = build_public_audit_suite(
        runs=[
            {
                "audit_id": "11111111-1111-4111-8111-111111111111",
                "backtest_id": "22222222-2222-4222-8222-222222222222",
                "as_of_date": "2026-03-27",
                "ticker_set": "SPY,QQQ,DIA,GLD",
                "information_ratio": 0.4,
                "pbo": 0.3,
                "cluster_adjusted_pbo": 0.2,
                "cluster_adjusted_pbo_label": "moderate_overfit_risk",
                "candidate_pbo_competition_dominated": True,
                "pbo_competition_dominated": True,
                "candidate_pbo_dominant_axis": "threshold",
                "pbo_dominant_axis": "threshold",
                "candidate_pbo_dominant_value": "0.30",
                "pbo_dominant_value": "0.30",
                "candidate_pbo_close_competition_ratio": 0.90,
                "pbo_close_competition_ratio": 0.75,
                "selection_stability": 0.8,
                "current_regime": "transition",
                "dominant_recent_regime": "low_vol",
                "regime_shift_flag": True,
                "state_probability": 0.82,
                "transition_rate": 0.05,
                "ohlcv_source": "yahoo_chart",
                "macro_source": "fred_csv",
                "ohlcv_transport_origin": "live",
                "macro_transport_origin": "cache",
                "news_used_source": "yahoo_finance_rss",
                "news_requested_source": "yahoo_finance_rss",
                "news_fallback_used": False,
                "news_transport_origin": "network",
                "fundamental_used_source": "sec_companyfacts",
                "fundamental_requested_source": "sec_companyfacts",
                "fundamental_fallback_used": False,
                "fundamental_transport_origin": "network",
                "news_feature_missing_rate": 0.10,
                "news_feature_coverage": 0.90,
                "news_feature_stale_rate": 0.00,
                "news_best_aggregation_window": 5,
                "news_best_aggregation_coverage": 0.96,
                "news_best_aggregation_abs_ic": 0.08,
                "news_best_decay_halflife": 3,
                "news_best_decay_coverage": 0.98,
                "news_best_decay_abs_ic": 0.07,
                "news_aggregation_improves_coverage": True,
                "news_aggregation_improves_utility": True,
                "news_decay_improves_coverage": True,
                "news_decay_improves_utility": False,
                "news_coverage_lookback_1d": 0.90,
                "news_coverage_lookback_3d": 0.94,
                "news_coverage_lookback_5d": 0.96,
                "news_coverage_lookback_10d": 0.97,
                "news_abs_ic_lookback_1d": 0.03,
                "news_abs_ic_lookback_3d": 0.05,
                "news_abs_ic_lookback_5d": 0.08,
                "news_abs_ic_lookback_10d": 0.06,
                "news_decay_coverage_halflife_3d": 0.98,
                "news_decay_coverage_halflife_5d": 0.97,
                "news_decay_abs_ic_halflife_3d": 0.07,
                "news_decay_abs_ic_halflife_5d": 0.06,
                "transition_profile": "stable_transition",
                "transition_history_matches": 2,
                "transition_observation_count": 3,
                "transition_span_business_days": 10,
                "transition_persistent": True,
                "stable_transition": True,
                "unstable_transition": False,
                "immediate_transition": False,
                "state_probability_bucket": "stable",
                "transition_rate_bucket": "low",
                "drift_weighted_score": 2.65,
                "drift_pre_suppression_weighted_score": 2.65,
                "drift_weighted_threshold": 2.205,
                "drift_low_vol_threshold": 2.45,
                "drift_immediate_threshold": 2.80,
                "drift_trigger_feature_count": 3,
                "drift_trigger_family_count": 3,
                "drift_pre_suppression_trigger_feature_count": 4,
                "drift_pre_suppression_trigger_family_count": 4,
                "drift_trigger_families": ["macro", "price_momentum", "volatility"],
                "drift_pre_suppression_trigger_families": ["macro", "price_momentum", "volatility", "volume"],
                "drift_proxy_sensitive_profile": "mixed",
                "drift_proxy_sensitive_trigger_feature_count": 1,
                "drift_non_proxy_trigger_feature_count": 2,
                "drift_pre_suppression_proxy_sensitive_trigger_feature_count": 1,
                "drift_pre_suppression_non_proxy_trigger_feature_count": 3,
                "drift_history_matches": 0,
                "drift_span_business_days": 0,
                "drift_suppressed_families": [],
                "drift_stable_transition_suppressed_families": [],
                "drift_primary_cause": "regime_shift_likely",
                "drift_family_persistence_would_suppress": True,
                "drift_family_persistence_counterfactual_score": 2.15,
                "drift_threshold_delta_to_suppress": 0.445,
                "drift_low_vol_threshold_would_suppress": False,
                "drift_stable_transition_suppression_would_suppress": True,
                "drift_stable_transition_counterfactual_score": 2.15,
                "ece_fold_persistent": True,
                "calibration_gap_fold_persistent": False,
                "ece_run_persistent": True,
                "calibration_gap_run_persistent": False,
                "ece_history_matches": 1,
                "calibration_gap_history_matches": 0,
                "ece_observation_count": 2,
                "calibration_gap_observation_count": 0,
                "ece_span_business_days": 10,
                "calibration_gap_span_business_days": 0,
                "base_should_retrain": True,
                "should_retrain": True,
                "policy_decision": "trigger",
                "trigger_names": ["calibration_ece"],
                "base_trigger_names": ["calibration_ece", "feature_drift", "regime_shift"],
            },
            {
                "audit_id": "33333333-3333-4333-8333-333333333333",
                "backtest_id": "44444444-4444-4444-8444-444444444444",
                "as_of_date": "2026-04-03",
                "ticker_set": "SPY,QQQ,GLD",
                "information_ratio": 0.2,
                "pbo": 0.5,
                "cluster_adjusted_pbo": 0.1,
                "cluster_adjusted_pbo_label": "low_overfit_risk",
                "candidate_pbo_competition_dominated": False,
                "pbo_competition_dominated": False,
                "candidate_pbo_dominant_axis": "holding_days",
                "pbo_dominant_axis": "holding_days",
                "candidate_pbo_dominant_value": "5",
                "pbo_dominant_value": "5",
                "candidate_pbo_close_competition_ratio": 0.25,
                "pbo_close_competition_ratio": 0.25,
                "selection_stability": 0.9,
                "current_regime": "low_vol",
                "dominant_recent_regime": "low_vol",
                "regime_shift_flag": False,
                "state_probability": 0.99,
                "transition_rate": 0.0,
                "ohlcv_source": "yahoo_chart",
                "macro_source": "fred_csv",
                "ohlcv_transport_origin": "cache",
                "macro_transport_origin": "cache",
                "news_used_source": "offline_news_proxy",
                "news_requested_source": "yahoo_finance_rss",
                "news_fallback_used": True,
                "news_transport_origin": "snapshot",
                "fundamental_used_source": "offline_fundamental_proxy",
                "fundamental_requested_source": "sec_companyfacts",
                "fundamental_fallback_used": True,
                "fundamental_transport_origin": "generated",
                "news_feature_missing_rate": 0.35,
                "news_feature_coverage": 0.65,
                "news_feature_stale_rate": 0.20,
                "news_best_aggregation_window": 3,
                "news_best_aggregation_coverage": 0.75,
                "news_best_aggregation_abs_ic": 0.03,
                "news_best_decay_halflife": 5,
                "news_best_decay_coverage": 0.82,
                "news_best_decay_abs_ic": 0.02,
                "news_aggregation_improves_coverage": True,
                "news_aggregation_improves_utility": False,
                "news_decay_improves_coverage": True,
                "news_decay_improves_utility": False,
                "news_coverage_lookback_1d": 0.65,
                "news_coverage_lookback_3d": 0.75,
                "news_coverage_lookback_5d": 0.78,
                "news_coverage_lookback_10d": 0.81,
                "news_abs_ic_lookback_1d": 0.01,
                "news_abs_ic_lookback_3d": 0.03,
                "news_abs_ic_lookback_5d": 0.02,
                "news_abs_ic_lookback_10d": 0.02,
                "news_decay_coverage_halflife_3d": 0.80,
                "news_decay_coverage_halflife_5d": 0.82,
                "news_decay_abs_ic_halflife_3d": 0.01,
                "news_decay_abs_ic_halflife_5d": 0.02,
                "transition_profile": "not_transition",
                "transition_history_matches": 0,
                "transition_observation_count": 0,
                "transition_span_business_days": 0,
                "transition_persistent": False,
                "stable_transition": False,
                "unstable_transition": False,
                "immediate_transition": False,
                "state_probability_bucket": "stable",
                "transition_rate_bucket": "zero",
                "drift_weighted_score": 0.0,
                "drift_pre_suppression_weighted_score": 0.0,
                "drift_weighted_threshold": 2.45,
                "drift_low_vol_threshold": 2.45,
                "drift_immediate_threshold": 2.95,
                "drift_trigger_feature_count": 0,
                "drift_trigger_family_count": 0,
                "drift_pre_suppression_trigger_feature_count": 0,
                "drift_pre_suppression_trigger_family_count": 0,
                "drift_trigger_families": [],
                "drift_pre_suppression_trigger_families": [],
                "drift_proxy_sensitive_profile": "none",
                "drift_proxy_sensitive_trigger_feature_count": 0,
                "drift_non_proxy_trigger_feature_count": 0,
                "drift_pre_suppression_proxy_sensitive_trigger_feature_count": 0,
                "drift_pre_suppression_non_proxy_trigger_feature_count": 0,
                "drift_history_matches": 0,
                "drift_span_business_days": 0,
                "drift_suppressed_families": [],
                "drift_stable_transition_suppressed_families": [],
                "drift_primary_cause": "stable",
                "drift_family_persistence_would_suppress": False,
                "drift_family_persistence_counterfactual_score": 0.0,
                "drift_threshold_delta_to_suppress": 0.0,
                "drift_low_vol_threshold_would_suppress": False,
                "drift_stable_transition_suppression_would_suppress": False,
                "drift_stable_transition_counterfactual_score": 0.0,
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
                "policy_decision": "suppressed_by_cooloff",
                "trigger_names": [],
                "base_trigger_names": ["feature_drift"],
            },
        ],
        as_of_dates=["2026-03-27", "2026-04-03"],
        ticker_sets=[["SPY", "QQQ", "DIA", "GLD"], ["SPY", "QQQ", "GLD"]],
        cpcv_max_splits=2,
        profile_name="standard",
        profile_role="research_comparison",
        analysis_mode="live_suite",
    )
    assert payload["run_count"] == 2
    assert payload["profile_name"] == "standard"
    assert payload["profile_role"] == "research_comparison"
    assert payload["distribution_summary"]["base_retraining_rate"] == 1.0
    assert payload["distribution_summary"]["retraining_rate"] == 0.5
    assert payload["distribution_summary"]["pbo_competition_dominated_rate"] == 0.5
    assert payload["distribution_summary"]["candidate_pbo_competition_dominated_rate"] == 0.5
    assert payload["distribution_summary"]["cluster_adjusted_pbo"]["mean"] == pytest.approx(0.15)
    assert payload["distribution_summary"]["news_feature_coverage"]["mean"] == pytest.approx(0.775)
    assert payload["distribution_summary"]["news_feature_missing_rate"]["mean"] == pytest.approx(0.225)
    assert payload["distribution_summary"]["news_feature_staleness"]["mean"] == pytest.approx(0.1)
    assert payload["distribution_summary"]["news_used_source_counts"]["yahoo_finance_rss"] == 1
    assert payload["distribution_summary"]["news_transport_origin_counts"]["network"] == 1
    assert payload["distribution_summary"]["news_fallback_rate"] == pytest.approx(0.5)
    assert payload["distribution_summary"]["fundamental_used_source_counts"]["sec_companyfacts"] == 1
    assert payload["distribution_summary"]["fundamental_transport_origin_counts"]["generated"] == 1
    assert payload["distribution_summary"]["fundamental_fallback_rate"] == pytest.approx(0.5)
    coverage_analysis = payload["distribution_summary"]["news_coverage_analysis"]
    utility_comparison = payload["distribution_summary"]["news_utility_comparison"]
    lookbacks = {item["window_days"]: item for item in coverage_analysis["lookback_windows"]}
    decays = {item["halflife_days"]: item for item in coverage_analysis["decay_halflives"]}
    assert lookbacks[1]["coverage"]["mean"] == pytest.approx(0.775)
    assert lookbacks[5]["coverage"]["mean"] == pytest.approx(0.87)
    assert decays[3]["coverage"]["mean"] == pytest.approx(0.89)
    assert "session_buckets" in utility_comparison
    assert "weighted_variants" in utility_comparison
    assert "weighting_mode_comparison" in utility_comparison
    assert "learned_weighting" in utility_comparison
    assert "multi_source_weighting_improvement" in utility_comparison
    assert "source_diversity_buckets" in utility_comparison
    assert "source_advantage_analysis" in utility_comparison
    assert "mixed_session_conditions" in utility_comparison
    assert coverage_analysis["best_aggregation_window_counts"]["5"] == 1
    assert coverage_analysis["best_decay_halflife_counts"]["3"] == 1
    assert coverage_analysis["aggregation_improves_coverage_rate"] == pytest.approx(1.0)
    assert payload["distribution_summary"]["pbo_dominant_axis_counts"]["threshold"] == 1
    assert payload["distribution_summary"]["candidate_pbo_dominant_axis_counts"]["threshold"] == 1
    assert payload["distribution_summary"]["base_trigger_counts"]["calibration_ece"] == 1
    assert payload["distribution_summary"]["effective_trigger_counts"]["calibration_ece"] == 1
    drift_analysis = payload["distribution_summary"]["drift_dominated_analysis"]
    assert drift_analysis["raw_run_count"] == 1
    assert drift_analysis["effective_run_count"] == 0
    assert drift_analysis["trigger_family_counts"]["volume"] == 1
    assert drift_analysis["proxy_sensitive_profile_counts"]["mixed"] == 1
    assert drift_analysis["regime_suppression_relief_count"] == 1
    assert drift_analysis["family_persistence_relief_count"] == 1
    assert drift_analysis["severity_threshold_low_vol_relief_count"] == 0
    calibration_analysis = payload["distribution_summary"]["calibration_dominated_analysis"]
    assert calibration_analysis["run_count"] == 1
    assert calibration_analysis["fold_persistence_counts"]["ece_only"] == 1
    assert calibration_analysis["run_persistence_counts"]["ece_only"] == 1
    assert calibration_analysis["current_regime_counts"]["transition"] == 1
    assert calibration_analysis["ticker_set_counts"]["SPY,QQQ,DIA,GLD"] == 1
    assert calibration_analysis["ohlcv_transport_origin_counts"]["live"] == 1
    regime_analysis = payload["distribution_summary"]["regime_dominated_analysis"]
    assert regime_analysis["raw_run_count"] == 1
    assert regime_analysis["effective_run_count"] == 0
    assert regime_analysis["base_trigger_counts"]["regime_shift"] == 1
    assert regime_analysis["transition_profile_counts"]["stable_transition"] == 1
    assert regime_analysis["state_probability_bucket_counts"]["stable"] == 1
    assert regime_analysis["transition_rate_bucket_counts"]["low"] == 1
    assert regime_analysis["co_trigger_family_counts"]["macro"] == 1
    by_ticker_set = {item["ticker_set"]: item for item in payload["by_ticker_set"]}
    assert by_ticker_set["SPY,QQQ,DIA,GLD"]["news_feature_coverage"]["mean"] == pytest.approx(0.9)
    assert by_ticker_set["SPY,QQQ,GLD"]["news_used_source_counts"]["offline_news_proxy"] == 1
    assert by_ticker_set["SPY,QQQ,GLD"]["fundamental_fallback_rate"] == pytest.approx(1.0)
    assert (
        by_ticker_set["SPY,QQQ,DIA,GLD"]["news_coverage_analysis"]["lookback_windows"][2]["coverage"]["mean"]
        == pytest.approx(0.96)
    )
    by_transport = {item["news_transport_origin"]: item for item in payload["by_news_transport_origin"]}
    assert by_transport["network"]["news_feature_coverage"]["mean"] == pytest.approx(0.9)
    assert by_transport["snapshot"]["news_feature_missing_rate"]["mean"] == pytest.approx(0.35)
    assert by_transport["snapshot"]["news_feature_staleness"]["mean"] == pytest.approx(0.2)
    assert by_transport["network"]["news_coverage_analysis"]["best_aggregation_window_counts"]["5"] == 1


def test_public_audit_suite_parsers_normalize_inputs() -> None:
    assert parse_as_of_dates("2026-04-03,2026-03-27") == ["2026-03-27", "2026-04-03"]
    assert parse_ticker_sets("spy,qqq,dia,gld|spy,qqq,gld", ["SPY"]) == [
        ["SPY", "QQQ", "DIA", "GLD"],
        ["SPY", "QQQ", "GLD"],
    ]


def test_resolve_public_audit_profile_uses_profile_defaults() -> None:
    profile = resolve_public_audit_profile(
        profile_name="full",
        default_tickers=["SPY", "QQQ", "DIA", "GLD"],
        anchor_date="2026-04-04",
    )
    assert profile.name == "full"
    assert profile.role == "replay_or_deep_dive"
    assert len(profile.as_of_dates) == 12
    assert profile.as_of_dates[-1] == "2026-04-03"
    assert profile.ticker_sets == [["SPY", "QQQ", "DIA", "GLD"], ["SPY", "QQQ", "GLD"]]
    assert profile.history_days == 1100
    assert profile.cpcv_max_splits == 2


def test_resolve_public_audit_profile_supports_full_light() -> None:
    profile = resolve_public_audit_profile(
        profile_name="full_light",
        default_tickers=["SPY", "QQQ", "DIA", "GLD"],
        anchor_date="2026-04-04",
    )
    assert profile.name == "full_light"
    assert profile.role == "routine_monitoring"
    assert len(profile.as_of_dates) == 12
    assert profile.ticker_sets == [["SPY", "QQQ", "DIA", "GLD"]]
    assert profile.cpcv_max_splits == 1


def test_replay_public_audit_suite_recomputes_policy_rates() -> None:
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"storage_path": str(artifact_root / "storage")},
    )
    backtest_dir = Path(settings.data.storage_path) / "outputs" / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    backtest_id = "22222222-2222-4222-8222-222222222222"
    backtest_payload = {
        "aggregate_metrics": {
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        "cost_adjusted_metrics": {"information_ratio": 0.5, "selection_stability": 0.9},
        "drift_monitor": {
            "supplementary_analysis": {
                "primary_cause": "regime_shift_likely",
                "feature_diagnostics": [
                    {"feature": "vix", "family": "macro", "status": "FAIL", "retrain_action": "trigger"},
                    {
                        "feature": "realized_vol_20d",
                        "family": "volatility",
                        "status": "FAIL",
                        "retrain_action": "trigger",
                    },
                    {
                        "feature": "volume_ratio_20d",
                        "family": "volume",
                        "status": "WARNING",
                        "retrain_action": "trigger",
                    },
                    {
                        "feature": "log_return_1d",
                        "family": "price_momentum",
                        "status": "WARNING",
                        "retrain_action": "trigger",
                    },
                ],
            }
        },
        "regime_monitor": {
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        "pbo": 0.1,
        "cpcv": {
            "pbo_summary": {"label": "low_overfit_risk"},
            "cluster_adjusted_pbo": 0.05,
            "cluster_adjusted_pbo_summary": {"label": "low_overfit_risk"},
        },
    }
    (backtest_dir / f"{backtest_id}.json").write_text(json.dumps(backtest_payload), encoding="utf-8")
    source_suite = {
        "suite_id": "11111111-1111-4111-8111-111111111111",
        "generated_at": "2026-04-04T00:00:00+00:00",
        "dataset_type": "public_real_market",
        "profile_name": "fast",
        "profile_role": "development_smoke",
        "as_of_dates": ["2026-04-03"],
        "ticker_sets": [["SPY", "QQQ", "GLD"]],
        "run_count": 1,
        "cpcv_max_splits": 1,
        "runs": [
            {
                "audit_id": "aaaaaaaa-1111-4111-8111-111111111111",
                "backtest_id": backtest_id,
                "as_of_date": "2026-04-03",
                "ticker_set": "SPY,QQQ,GLD",
                "information_ratio": 0.5,
                "pbo": 0.1,
                "cluster_adjusted_pbo": 0.05,
                "cluster_adjusted_pbo_label": "low_overfit_risk",
                "candidate_pbo_competition_dominated": False,
                "pbo_competition_dominated": False,
                "candidate_pbo_dominant_axis": "",
                "pbo_dominant_axis": "",
                "candidate_pbo_dominant_value": "",
                "pbo_dominant_value": "",
                "candidate_pbo_close_competition_ratio": 0.0,
                "pbo_close_competition_ratio": 0.0,
                "selection_stability": 0.9,
                "hit_rate_mean": 0.45,
                "base_should_retrain": False,
                "should_retrain": False,
                "policy_decision": "watch_only",
                "trigger_names": [],
            }
        ],
        "distribution_summary": {"base_retraining_rate": 0.0, "retraining_rate": 0.0},
    }

    replay = replay_public_audit_suite(settings=settings, source_payload=source_suite, source_suite_path="suite.json")

    assert replay["analysis_mode"] == "retraining_policy_replay"
    assert replay["profile_role"] == "development_smoke"
    assert replay["distribution_summary"]["base_retraining_rate"] == 1.0
    assert replay["distribution_summary"]["retraining_rate"] == 1.0
    assert replay["comparison_to_source"]["base_retraining_rate_delta"] == 1.0
    assert replay["comparison_to_source"]["retraining_rate_delta"] == 1.0
    assert replay["runs"][0]["base_trigger_names"] == ["feature_drift"]
    assert replay["runs"][0]["trigger_names"] == ["feature_drift"]
