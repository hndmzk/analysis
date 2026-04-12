from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline
from market_prediction_agent.schemas.validator import validate_payload


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


def _assert_pipeline_outputs(result, artifact_root: Path) -> None:
    validate_payload("forecast_output", result.forecast_output)
    validate_payload("evidence_bundle", result.evidence_bundle)
    validate_payload("risk_review", result.risk_review)
    validate_payload("report_payload", result.report_payload)
    validate_payload("backtest_result", result.backtest_result)
    validate_payload("paper_trading_batch", result.paper_trading_batch)
    validate_payload("weekly_review", result.weekly_review)
    if result.retraining_event is not None:
        validate_payload("retraining_event", result.retraining_event)
    assert result.backtest_result["feature_importance_summary"]
    assert result.backtest_result["feature_catalog"]
    assert result.backtest_result["feature_family_importance_summary"]
    assert "drift_monitor" in result.backtest_result
    assert "regime_monitor" in result.backtest_result
    assert "retraining_monitor" in result.backtest_result
    assert "cpcv" in result.backtest_result
    assert result.backtest_result["pbo"] is not None
    assert "pbo_summary" in result.backtest_result["cpcv"]
    assert "ece_mean" in result.backtest_result["aggregate_metrics"]
    assert result.backtest_result["folds"][0]["feature_importance"]
    assert "enabled" in result.backtest_result["folds"][0]["calibration"]
    assert result.evidence_bundle["model_info"]["feature_importance_top"]
    assert result.evidence_bundle["data_snapshot"]["feature_catalog"]
    assert result.evidence_bundle["model_info"]["feature_family_importance"]
    assert "calibration_summary" in result.evidence_bundle["model_info"]
    assert "regime_monitor" in result.evidence_bundle["data_snapshot"]
    assert "retraining_monitor" in result.evidence_bundle["model_info"]
    assert result.paper_trading_batch["metrics"]["new_trades"] > 0
    assert "avg_round_trip_cost_bps" in result.paper_trading_batch["metrics"]
    assert "execution_diagnostics" in result.paper_trading_batch
    assert "fill_rate" in result.paper_trading_batch["execution_diagnostics"]
    assert "partial_fill_rate" in result.paper_trading_batch["execution_diagnostics"]
    assert "missed_trade_rate" in result.paper_trading_batch["execution_diagnostics"]
    assert "retraining_batches" in result.weekly_review["metrics"]
    assert "liquidity_blocked_trades" in result.weekly_review["metrics"]
    assert "execution_diagnostics" in result.weekly_review
    assert "realized_vs_intended_exposure" in result.weekly_review["execution_diagnostics"]
    assert "execution_cost_drag" in result.weekly_review["execution_diagnostics"]
    assert "gap_slippage_bps" in result.weekly_review["execution_diagnostics"]
    feature_families = {
        item["feature_family"]
        for item in result.backtest_result["feature_catalog"]
    }
    assert {"news", "fundamental", "sector"}.issubset(feature_families)
    assert (artifact_root / "storage" / "outputs" / "backtests").exists()


def test_end_to_end_pipeline_generates_outputs(monkeypatch) -> None:
    config_path = Path("config") / "default.yaml"
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings(),
        data={
            "storage_path": str(artifact_root / "storage"),
            "dummy_ticker_count": 20,
            "dummy_days": 900,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 500,
                "eval_days": 60,
                "step_days": 60,
            },
            "cpcv": {
                "group_count": 4,
                "test_groups": 1,
                "max_splits": 2,
            },
        },
    )
    pipeline = MarketPredictionPipeline(settings)
    result = pipeline.run()
    _assert_pipeline_outputs(result, artifact_root)


def test_end_to_end_pipeline_fast_fixture(monkeypatch) -> None:
    config_path = Path("config") / "default.yaml"
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setattr("market_prediction_agent.features.pipeline.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    monkeypatch.setattr("market_prediction_agent.agents.forecast_agent.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings(),
        data={
            "storage_path": str(artifact_root / "storage"),
            "dummy_ticker_count": 10,
            "dummy_days": 200,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 100,
                "eval_days": 30,
                "step_days": 30,
            },
            "cpcv": {
                "group_count": 4,
                "test_groups": 1,
                "max_splits": 2,
            },
        },
    )
    pipeline = MarketPredictionPipeline(settings)
    result = pipeline.run()
    _assert_pipeline_outputs(result, artifact_root)

