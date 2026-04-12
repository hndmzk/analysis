from __future__ import annotations

import pandas as pd

from market_prediction_agent.agents.risk_agent import RiskAgent
from market_prediction_agent.config import load_settings
from market_prediction_agent.reporting.builders import build_forecast_output
from market_prediction_agent.schemas.validator import validate_payload


def test_risk_agent_review_returns_schema_compliant_payload() -> None:
    settings = load_settings("config/default.yaml")
    agent = RiskAgent(settings)
    predictions = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": pd.Timestamp("2026-01-02", tz="UTC"),
                "stale_data_flag": False,
                "prob_down": 0.2,
                "prob_flat": 0.3,
                "prob_up": 0.5,
                "direction": "UP",
                "expected_return": 0.01,
                "predicted_volatility": 0.2,
                "confidence": "high",
                "top_features": [{"name": "log_return_1d", "shap_value": 0.2}],
            }
        ]
    )
    forecast_output = build_forecast_output(
        predictions=predictions,
        model_version=settings.model_settings.version,
        horizon=settings.data.forecast_horizon,
        regime="low_vol",
    )
    backtest_result = {
        "config": {
            "dummy_mode": "predictable_momentum",
        },
        "cost_adjusted_metrics": {
            "max_drawdown": -0.05,
        },
        "drift_monitor": {
            "max_psi": 0.05,
            "warning_threshold": 0.2,
            "critical_threshold": 0.25,
            "warning_features": [],
            "critical_features": [],
            "supplementary_analysis": {"primary_cause": "stable"},
        },
        "regime_monitor": {
            "current_regime": "low_vol",
            "regime_shift_flag": False,
            "state_probability": 0.9,
        },
        "retraining_monitor": {
            "should_retrain": False,
            "trigger_count": 0,
            "triggers": [],
        },
    }

    review = agent.review(forecast_output, backtest_result)

    validate_payload("risk_review", review)
    assert review["forecast_id"] == forecast_output["forecast_id"]
    assert review["checks"]

