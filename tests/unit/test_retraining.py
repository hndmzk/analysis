from __future__ import annotations

from market_prediction_agent.config import load_settings
from market_prediction_agent.evaluation.retraining import build_retraining_monitor


def test_retraining_monitor_triggers_on_persistent_drift_calibration_and_regime() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.08,
            "calibration_gap_mean": 0.02,
            "ece_warning_breach_count": 3,
            "ece_warning_breach_ratio": 0.75,
            "calibration_gap_warning_breach_count": 1,
            "calibration_gap_warning_breach_ratio": 0.25,
        },
        drift_summary={
            "max_psi": 0.3,
            "max_raw_psi": 0.6,
            "supplementary_analysis": {
                "primary_cause": "regime_shift_likely",
                "non_proxy_flagged_features": ["vix", "realized_vol_20d"],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [
                    {
                        "feature": "vix",
                        "family": "macro",
                        "status": "FAIL",
                        "retrain_action": "trigger",
                    },
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
                ],
            },
        },
        regime_summary={
            "current_regime": "transition",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": True,
            "state_probability": 0.5,
            "transition_rate": 0.2,
        },
        pbo=0.85,
        pbo_summary={"label": "severe_overfit_risk"},
        settings=settings,
    )
    assert monitor["should_retrain"] is True
    trigger_names = {item["name"] for item in monitor["triggers"]}
    assert {"feature_drift", "calibration_ece", "regime_shift", "transition_regime", "pbo"} <= trigger_names


def test_retraining_monitor_suppresses_proxy_only_drift_trigger() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.32,
            "max_raw_psi": 0.95,
            "supplementary_analysis": {
                "primary_cause": "proxy_artifact_likely",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": ["garman_klass_vol", "atr_ratio"],
                "feature_diagnostics": [
                    {
                        "feature": "garman_klass_vol",
                        "family": "volatility",
                        "status": "FAIL",
                        "retrain_action": "watch",
                    },
                    {
                        "feature": "atr_ratio",
                        "family": "volatility",
                        "status": "WARNING",
                        "retrain_action": "watch",
                    },
                ],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
    )
    assert monitor["should_retrain"] is False
    assert any(item["name"] == "feature_drift_proxy_watch" for item in monitor["observations"])


def test_retraining_monitor_requires_drift_persistence_for_single_family_noise() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.31,
            "max_raw_psi": 0.44,
            "supplementary_analysis": {
                "primary_cause": "mixed_or_unknown",
                "non_proxy_flagged_features": ["yield_curve_slope"],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [
                    {
                        "feature": "yield_curve_slope",
                        "family": "macro",
                        "status": "FAIL",
                        "retrain_action": "trigger",
                    }
                ],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
    )
    assert monitor["should_retrain"] is False
    assert any(item["name"] == "feature_drift_persistence_watch" for item in monitor["observations"])


def test_retraining_monitor_requires_calibration_persistence() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.06,
            "calibration_gap_mean": 0.02,
            "ece_warning_breach_count": 1,
            "ece_warning_breach_ratio": 0.25,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
    )
    assert monitor["should_retrain"] is False
    assert any(item["name"] == "calibration_ece_persistence_watch" for item in monitor["observations"])


def test_retraining_monitor_requires_calibration_run_persistence() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.06,
            "calibration_gap_mean": 0.02,
            "ece_warning_breach_count": 2,
            "ece_warning_breach_ratio": 0.5,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={"as_of_date": "2026-04-03", "history": []},
    )
    assert monitor["should_retrain"] is False
    calibration_signal = monitor["calibration_signal"]
    assert calibration_signal["ece_fold_persistent"] is True
    assert calibration_signal["ece_run_persistent"] is False
    assert any(item["name"] == "calibration_ece_run_persistence_watch" for item in monitor["observations"])


def test_retraining_monitor_triggers_on_persistent_calibration_history() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.06,
            "calibration_gap_mean": 0.02,
            "ece_warning_breach_count": 2,
            "ece_warning_breach_ratio": 0.5,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-04-03",
            "history": [
                {
                    "as_of_date": "2026-03-20",
                    "regime_bucket": "low_vol",
                    "base_trigger_names": ["calibration_ece"],
                    "base_cause_keys": ["calibration_ece:low_vol"],
                }
            ],
        },
    )
    assert monitor["should_retrain"] is True
    assert {item["name"] for item in monitor["triggers"]} == {"calibration_ece"}
    calibration_signal = monitor["calibration_signal"]
    assert calibration_signal["ece_history_matches"] == 1
    assert calibration_signal["ece_observation_count"] == 2
    assert calibration_signal["ece_run_persistent"] is True


def test_retraining_monitor_applies_high_vol_family_suppression() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.33,
            "max_raw_psi": 0.52,
            "supplementary_analysis": {
                "primary_cause": "regime_shift_likely",
                "non_proxy_flagged_features": ["vix", "realized_vol_20d", "log_return_1d", "volume_ratio_20d"],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [
                    {"feature": "vix", "family": "macro", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "realized_vol_20d", "family": "volatility", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "log_return_1d", "family": "price_momentum", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "volume_ratio_20d", "family": "volume", "status": "FAIL", "retrain_action": "trigger"},
                ],
            },
        },
        regime_summary={
            "current_regime": "high_vol",
            "dominant_recent_regime": "high_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={"as_of_date": "2026-04-03", "history": []},
    )
    assert monitor["base_should_retrain"] is False
    assert monitor["should_retrain"] is False
    assert set(monitor["drift_signal"]["suppressed_families"]) == {"price_momentum", "volume"}
    assert any(item["name"] == "feature_drift_family_suppression_watch" for item in monitor["observations"])


def test_retraining_monitor_applies_stable_transition_volume_suppression() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.31,
            "max_raw_psi": 0.44,
            "supplementary_analysis": {
                "primary_cause": "regime_shift_likely",
                "non_proxy_flagged_features": ["vix", "realized_vol_20d", "macd_signal"],
                "proxy_sensitive_flagged_features": ["obv_slope_10d"],
                "feature_diagnostics": [
                    {"feature": "vix", "family": "macro", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "realized_vol_20d", "family": "volatility", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "macd_signal", "family": "price_momentum", "status": "FAIL", "retrain_action": "trigger"},
                    {
                        "feature": "obv_slope_10d",
                        "family": "volume",
                        "status": "WARNING",
                        "retrain_action": "trigger",
                        "proxy_sensitive": True,
                    },
                ],
            },
        },
        regime_summary={
            "current_regime": "transition",
            "dominant_recent_regime": "transition",
            "regime_shift_flag": False,
            "state_probability": 0.99,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={"as_of_date": "2026-04-03", "history": []},
    )
    assert monitor["base_should_retrain"] is False
    assert monitor["should_retrain"] is False
    drift_signal = monitor["drift_signal"]
    assert drift_signal["pre_suppression_weighted_score"] == 2.65
    assert drift_signal["weighted_score"] == 2.15
    assert drift_signal["stable_transition_suppression_would_suppress"] is True
    assert "volume" in drift_signal["stable_transition_suppressed_families"]


def test_retraining_monitor_treats_persistent_low_churn_transition_as_stable() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.31,
            "max_raw_psi": 0.44,
            "supplementary_analysis": {
                "primary_cause": "regime_shift_likely",
                "non_proxy_flagged_features": ["vix", "realized_vol_20d", "macd_signal"],
                "proxy_sensitive_flagged_features": ["obv_slope_10d"],
                "feature_diagnostics": [
                    {"feature": "vix", "family": "macro", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "realized_vol_20d", "family": "volatility", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "macd_signal", "family": "price_momentum", "status": "FAIL", "retrain_action": "trigger"},
                    {
                        "feature": "obv_slope_10d",
                        "family": "volume",
                        "status": "WARNING",
                        "retrain_action": "trigger",
                        "proxy_sensitive": True,
                    },
                ],
            },
        },
        regime_summary={
            "current_regime": "transition",
            "dominant_recent_regime": "transition",
            "regime_shift_flag": False,
            "state_probability": 0.57,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-03-13",
            "history": [
                {"as_of_date": "2026-02-06", "current_regime": "transition"},
                {"as_of_date": "2026-02-13", "current_regime": "transition"},
                {"as_of_date": "2026-02-20", "current_regime": "transition"},
                {"as_of_date": "2026-02-27", "current_regime": "transition"},
                {"as_of_date": "2026-03-06", "current_regime": "transition"},
            ],
        },
    )
    assert monitor["base_should_retrain"] is False
    assert monitor["should_retrain"] is False
    assert monitor["regime_signal"]["transition_profile"] == "stable_transition"
    assert monitor["regime_signal"]["transition_persistent"] is True
    assert "volume" in monitor["drift_signal"]["stable_transition_suppressed_families"]


def test_retraining_monitor_applies_run_to_run_cooloff() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.31,
            "max_raw_psi": 0.44,
            "supplementary_analysis": {
                "primary_cause": "mixed_or_unknown",
                "non_proxy_flagged_features": ["yield_curve_slope", "realized_vol_20d", "volume_ratio_20d", "log_return_1d"],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [
                    {"feature": "yield_curve_slope", "family": "macro", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "realized_vol_20d", "family": "volatility", "status": "FAIL", "retrain_action": "trigger"},
                    {"feature": "volume_ratio_20d", "family": "volume", "status": "WARNING", "retrain_action": "trigger"},
                    {"feature": "log_return_1d", "family": "price_momentum", "status": "WARNING", "retrain_action": "trigger"},
                ],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-03-27",
            "history": [
                {
                    "as_of_date": "2026-03-20",
                    "current_regime": "low_vol",
                    "should_retrain": True,
                    "base_should_retrain": True,
                    "trigger_names": ["feature_drift"],
                    "drift_trigger_families": ["macro", "volatility"],
                }
            ],
        },
    )
    assert monitor["base_should_retrain"] is True
    assert monitor["should_retrain"] is False
    assert monitor["policy_decision"] == "suppressed_by_cooloff"
    assert "feature_drift" in monitor["suppressed_trigger_names"]


def test_retraining_monitor_requires_pbo_persistence_when_standalone() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.6,
        pbo_summary={"label": "high_overfit_risk"},
        settings=settings,
        policy_context={"as_of_date": "2026-04-03", "history": []},
    )
    assert monitor["base_should_retrain"] is False
    assert monitor["should_retrain"] is False
    assert any(item["name"] == "pbo_persistence_watch" for item in monitor["observations"])


def test_retraining_monitor_triggers_on_persistent_pbo_history() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.6,
        pbo_summary={"label": "high_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-04-03",
            "history": [
                {
                    "as_of_date": "2026-02-27",
                    "should_retrain": False,
                    "base_should_retrain": False,
                    "base_cause_keys": ["pbo:high_overfit_risk"],
                    "effective_cause_keys": [],
                    "pbo": 0.62,
                },
                {
                    "as_of_date": "2026-03-20",
                    "should_retrain": False,
                    "base_should_retrain": False,
                    "base_cause_keys": ["pbo:high_overfit_risk"],
                    "effective_cause_keys": [],
                    "pbo": 0.64,
                }
            ],
        },
    )
    assert monitor["base_should_retrain"] is True
    assert monitor["should_retrain"] is True
    assert "pbo" in monitor["effective_trigger_names"]


def test_retraining_monitor_treats_competition_dominated_severe_pbo_as_watch() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=1.0,
        pbo_summary={"label": "severe_overfit_risk"},
        pbo_diagnostics={
            "near_candidate_competition": {
                "competition_dominated": True,
                "competition_reason": "Threshold variants were near-ties across CPCV splits.",
            }
        },
        settings=settings,
        policy_context={"as_of_date": "2026-04-03", "history": []},
    )
    assert monitor["base_should_retrain"] is False
    assert monitor["should_retrain"] is False
    assert any(item["name"] == "pbo_competition_watch" for item in monitor["observations"])


def test_retraining_monitor_suppresses_standalone_regime_shift() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "high_vol",
            "regime_shift_flag": True,
            "state_probability": 0.85,
            "transition_rate": 0.05,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
    )
    assert monitor["base_should_retrain"] is True
    assert monitor["should_retrain"] is False
    assert monitor["policy_decision"] == "watch_regime_shift_only"


def test_retraining_monitor_requires_drift_to_keep_regime_shift_effective() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.09,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 2,
            "ece_warning_breach_ratio": 1.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "high_vol",
            "regime_shift_flag": True,
            "state_probability": 0.4,
            "transition_rate": 0.2,
        },
        pbo=0.1,
        pbo_summary={"label": "low_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-04-03",
            "history": [
                {
                    "as_of_date": "2026-03-20",
                    "base_cause_keys": ["calibration_ece:low_vol"],
                    "effective_cause_keys": [],
                }
            ],
        },
    )
    assert monitor["base_should_retrain"] is True
    assert monitor["should_retrain"] is True
    assert "calibration_ece" in monitor["effective_trigger_names"]
    assert "regime_shift" not in monitor["effective_trigger_names"]
    assert "regime_shift" in monitor["suppressed_trigger_names"]


def test_retraining_monitor_suppresses_repeated_pbo_cause_within_cooloff() -> None:
    settings = load_settings("config/default.yaml")
    monitor = build_retraining_monitor(
        aggregate_metrics={
            "ece_mean": 0.01,
            "calibration_gap_mean": 0.01,
            "ece_warning_breach_count": 0,
            "ece_warning_breach_ratio": 0.0,
            "calibration_gap_warning_breach_count": 0,
            "calibration_gap_warning_breach_ratio": 0.0,
        },
        drift_summary={
            "max_psi": 0.1,
            "max_raw_psi": 0.1,
            "supplementary_analysis": {
                "primary_cause": "stable",
                "non_proxy_flagged_features": [],
                "proxy_sensitive_flagged_features": [],
                "feature_diagnostics": [],
            },
        },
        regime_summary={
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
            "transition_rate": 0.0,
        },
        pbo=0.85,
        pbo_summary={"label": "severe_overfit_risk"},
        settings=settings,
        policy_context={
            "as_of_date": "2026-04-03",
            "history": [
                {
                    "as_of_date": "2026-03-31",
                    "current_regime": "low_vol",
                    "regime_bucket": "low_vol",
                    "should_retrain": True,
                    "base_should_retrain": True,
                    "effective_cause_keys": ["pbo:severe_overfit_risk"],
                }
            ],
        },
    )
    assert monitor["base_should_retrain"] is True
    assert monitor["should_retrain"] is False
    assert monitor["policy_decision"] == "suppressed_by_cooloff"
