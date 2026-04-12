from __future__ import annotations

import pandas as pd

from market_prediction_agent.backtest.walk_forward import run_model_comparisons
from market_prediction_agent.config import load_settings, update_settings


def test_run_model_comparisons_summarizes_primary_deltas(monkeypatch) -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={"comparison_models": ["xgboost_multiclass_calibrated"]},
    )
    calls: list[tuple[str | None, bool, bool]] = []

    def fake_run_walk_forward_backtest(
        training_frame: pd.DataFrame,
        feature_columns: list[str],
        runtime_settings,
        *,
        model_name: str | None = None,
        include_feature_importance: bool = True,
        include_cpcv: bool = True,
        feature_catalog: list[dict[str, object]] | None = None,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        del training_frame, feature_columns, runtime_settings, feature_catalog
        calls.append((model_name, include_feature_importance, include_cpcv))
        return (
            {
                "aggregate_metrics": {
                    "hit_rate_mean": 0.43,
                    "log_loss_mean": 0.98,
                    "ece_mean": 0.04,
                },
                "cost_adjusted_metrics": {
                    "information_ratio": 0.35,
                },
                "diebold_mariano": {
                    "vs_baseline": "class_prior",
                    "dm_statistic": 0.1,
                    "p_value": 0.9,
                },
            },
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.run_walk_forward_backtest",
        fake_run_walk_forward_backtest,
    )
    primary_backtest_result = {
        "config": {"model_name": "lightgbm_multiclass_calibrated"},
        "aggregate_metrics": {
            "hit_rate_mean": 0.40,
            "log_loss_mean": 1.02,
            "ece_mean": 0.05,
        },
        "cost_adjusted_metrics": {"information_ratio": 0.20},
    }
    results = run_model_comparisons(
        training_frame=pd.DataFrame(),
        feature_columns=["feature_a"],
        settings=settings,
        primary_backtest_result=primary_backtest_result,
    )
    assert calls == [("xgboost_multiclass_calibrated", False, False)]
    assert len(results) == 1
    assert results[0]["status"] == "completed"
    assert round(results[0]["comparison_to_primary"]["hit_rate_mean_delta"], 6) == 0.03
    assert round(results[0]["comparison_to_primary"]["information_ratio_delta"], 6) == 0.15

