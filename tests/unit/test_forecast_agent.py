from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.agents.forecast_agent import ForecastAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


FAST_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "realized_vol_5d",
    "garman_klass_vol",
    "bb_width",
    "atr_ratio",
    "volume_ratio_20d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "fed_funds_rate",
    "yield_curve_slope",
    "vix",
    "vix_change_5d",
    "news_sentiment_5d",
    "news_relevance_5d",
    "fundamental_revenue_growth",
    "fundamental_profitability",
    "sector_relative_momentum_20d",
    "sector_strength_20d",
    "day_of_week",
    "month",
    "is_month_end",
]


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-forecast-agent" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_forecast_fixture(monkeypatch):
    monkeypatch.setattr("market_prediction_agent.features.pipeline.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    monkeypatch.setattr("market_prediction_agent.agents.forecast_agent.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    temp_dir = _workspace_temp_dir()
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "dummy",
            "dummy_mode": "predictable_momentum",
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 40,
                "eval_days": 15,
                "step_days": 15,
                "embargo_days": 0,
            },
            "cpcv": {
                "group_count": 3,
                "test_groups": 1,
                "max_splits": 2,
            },
            "lightgbm": {
                "n_estimators": 20,
                "num_leaves": 7,
                "min_child_samples": 5,
                "max_shap_samples": 25,
            },
            "calibration": {
                "min_days": 10,
                "fraction": 0.1,
            },
            "comparison_models": [],
        },
    )
    store = ParquetStore(temp_dir)
    data_agent = DataAgent(settings, store)
    artifacts = data_agent.generate_or_fetch(
        tickers=["AAA", "BBB", "CCC", "DDD"],
        start_date="2024-01-01",
        end_date="2024-09-30",
        as_of_time="2024-10-01T00:00:00Z",
    )
    return settings, artifacts, temp_dir


def test_forecast_agent_predict_returns_schema_compliant_output(monkeypatch) -> None:
    settings, artifacts, temp_dir = _build_forecast_fixture(monkeypatch)
    try:
        agent = ForecastAgent(settings)
        forecast_output = agent.predict(
            artifacts.processed_ohlcv,
            artifacts.processed_macro,
            artifacts.processed_news,
            artifacts.processed_fundamentals,
            artifacts.processed_sector_map,
            tickers=["AAA", "BBB", "CCC", "DDD"],
            as_of_time=artifacts.processed_ohlcv["timestamp_utc"].max(),
            source_metadata=artifacts.ohlcv_metadata,
        )
        validate_payload("forecast_output", forecast_output)
        assert len(forecast_output["predictions"]) == 4
        assert {"UP", "DOWN", "FLAT"} == set(forecast_output["predictions"][0]["probabilities"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_forecast_agent_run_end_to_end_with_small_fixture(monkeypatch) -> None:
    settings, artifacts, temp_dir = _build_forecast_fixture(monkeypatch)
    try:
        agent = ForecastAgent(settings)
        forecast_artifacts = agent.run(
            artifacts.processed_ohlcv,
            artifacts.processed_macro,
            artifacts.processed_news,
            artifacts.processed_fundamentals,
            artifacts.processed_sector_map,
            tickers=["AAA", "BBB", "CCC", "DDD"],
            as_of_time=artifacts.processed_ohlcv["timestamp_utc"].max(),
            source_metadata=artifacts.ohlcv_metadata,
        )
        assert not forecast_artifacts.training_frame.empty
        assert not forecast_artifacts.latest_predictions.empty
        assert forecast_artifacts.backtest_result["folds"]
        assert "cpcv" in forecast_artifacts.backtest_result
        assert "max_psi" in forecast_artifacts.drift_summary
        assert "current_regime" in forecast_artifacts.regime_summary
        assert "should_retrain" in forecast_artifacts.retraining_monitor
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

