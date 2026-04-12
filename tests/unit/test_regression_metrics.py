from __future__ import annotations

import pytest
import pandas as pd

from market_prediction_agent.evaluation.metrics import return_regression_metrics, volatility_regression_metrics


def test_return_regression_metrics_match_expected_values() -> None:
    predictions = pd.DataFrame(
        {
            "expected_return": [0.01, 0.02, 0.03],
            "future_simple_return": [0.02, 0.04, 0.06],
            "target_return": [0.00, 0.01, 0.05],
            "predicted_volatility": [0.0, 0.0, 0.0],
            "future_volatility_20d": [0.0, 0.0, 0.0],
        }
    )

    metrics = return_regression_metrics(predictions)

    assert metrics["return_ic"] == pytest.approx(1.0)
    assert metrics["return_rank_ic"] == pytest.approx(1.0)
    assert metrics["return_mae"] == pytest.approx(0.013333333333333334)
    assert metrics["return_rmse"] == pytest.approx(0.01414213562373095)


def test_volatility_regression_metrics_match_expected_values() -> None:
    predictions = pd.DataFrame(
        {
            "expected_return": [0.0, 0.0, 0.0],
            "future_simple_return": [0.0, 0.0, 0.0],
            "target_return": [0.0, 0.0, 0.0],
            "predicted_volatility": [0.10, 0.20, 0.30],
            "future_volatility_20d": [0.10, 0.25, 0.20],
        }
    )

    metrics = volatility_regression_metrics(predictions)

    assert metrics["vol_mae"] == pytest.approx(0.05)
    assert metrics["vol_mape"] == pytest.approx(0.2333333333333333)
    assert metrics["vol_rmse"] == pytest.approx(0.06454972243679027)


def test_volatility_regression_metrics_handles_all_zero_actuals_for_mape() -> None:
    predictions = pd.DataFrame(
        {
            "expected_return": [0.0, 0.0],
            "future_simple_return": [0.0, 0.0],
            "target_return": [0.0, 0.0],
            "predicted_volatility": [0.10, 0.00],
            "future_volatility_20d": [0.00, 0.00],
        }
    )

    metrics = volatility_regression_metrics(predictions)

    assert metrics["vol_mae"] == pytest.approx(0.05)
    assert metrics["vol_mape"] == pytest.approx(0.0)
    assert metrics["vol_rmse"] == pytest.approx(0.07071067811865475)
