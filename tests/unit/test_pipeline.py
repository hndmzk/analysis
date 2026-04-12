from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline, PipelineRunResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n_tickers: int = 2, n_dates: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_tickers):
        for d in range(n_dates):
            rows.append({
                "ticker": f"T{i}",
                "timestamp_utc": pd.Timestamp("2026-04-01", tz="UTC") + pd.tseries.offsets.BDay(d),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                "volume": 1e6, "source": "dummy",
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "data_age_hours": 0.0, "stale_data_flag": False,
            })
    return pd.DataFrame(rows)


def _make_macro(n_dates: int = 5) -> pd.DataFrame:
    rows = []
    for d in range(n_dates):
        rows.append({
            "date": pd.Timestamp("2026-04-01", tz="UTC") + pd.tseries.offsets.BDay(d),
            "fed_funds_rate": 4.5, "yield_curve_slope": 0.3,
            "vix": 15.0, "vix_change_5d": -0.5,
            "source": "dummy", "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "data_age_hours": 0.0, "stale_data_flag": False,
        })
    return pd.DataFrame(rows)


def _make_news(n_tickers: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_tickers):
        rows.append({
            "ticker": f"T{i}",
            "headline_count": 3, "sentiment_score": 0.1, "novelty_score": 0.5,
            "relevance_score": 0.7, "source": "offline_news_proxy",
            "published_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "available_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "data_age_hours": 0.0, "stale_data_flag": False,
            "session_bucket": "regular", "source_diversity": 2.0,
        })
    return pd.DataFrame(rows)


@dataclass
class _FakeDataArtifacts:
    raw_ohlcv: pd.DataFrame
    raw_macro: pd.DataFrame
    raw_news: pd.DataFrame
    raw_fundamentals: pd.DataFrame
    raw_sector_map: pd.DataFrame
    processed_ohlcv: pd.DataFrame
    processed_macro: pd.DataFrame
    processed_news: pd.DataFrame
    processed_fundamentals: pd.DataFrame
    processed_sector_map: pd.DataFrame
    ohlcv_metadata: dict[str, object]


@dataclass
class _FakeForecastArtifacts:
    feature_frame: pd.DataFrame
    feature_catalog: list[dict[str, object]]
    training_frame: pd.DataFrame
    backtest_result: dict[str, object]
    backtest_predictions: pd.DataFrame
    latest_predictions: pd.DataFrame
    model: Any
    drift_summary: dict[str, object]
    regime_summary: dict[str, object]
    retraining_monitor: dict[str, object]


@dataclass
class _FakePaperTradingArtifacts:
    batch_log: dict[str, object]
    weekly_review: dict[str, object]
    retraining_event: dict[str, object] | None


def _build_data_artifacts() -> _FakeDataArtifacts:
    ohlcv = _make_ohlcv()
    return _FakeDataArtifacts(
        raw_ohlcv=ohlcv, raw_macro=_make_macro(),
        raw_news=_make_news(), raw_fundamentals=pd.DataFrame(),
        raw_sector_map=pd.DataFrame(),
        processed_ohlcv=ohlcv, processed_macro=_make_macro(),
        processed_news=_make_news(), processed_fundamentals=pd.DataFrame(),
        processed_sector_map=pd.DataFrame(),
        ohlcv_metadata={
            "requested_source": "dummy", "used_source": "dummy",
            "fallback_used": False, "fallback_reason": None,
            "feature_sources": {
                "news": {"requested_source": "offline_news_proxy", "used_source": "offline_news_proxy"},
            },
        },
    )


def _build_forecast_artifacts() -> _FakeForecastArtifacts:
    preds = pd.DataFrame({
        "ticker": ["T0", "T1"],
        "predicted_class": [2, 0],
        "prob_down": [0.1, 0.5],
        "prob_neutral": [0.2, 0.3],
        "prob_up": [0.7, 0.2],
        "stale_data_flag": [False, False],
    })
    model = MagicMock()
    model.feature_columns = ["log_return_1d", "realized_vol_5d"]
    model.trained_at = "2026-04-01T00:00:00Z"
    model.training_samples = 100
    model.hyperparameters.return_value = {"n_estimators": 100}
    model.calibration_summary = {"enabled": True, "method": "isotonic"}
    feature_frame = preds.copy()
    feature_frame["date"] = pd.Timestamp("2026-04-10", tz="UTC")
    feature_frame["log_return_1d"] = 0.01
    feature_frame["realized_vol_5d"] = 0.15
    return _FakeForecastArtifacts(
        feature_frame=feature_frame,
        feature_catalog=[
            {"name": "log_return_1d", "feature_family": "price", "missing_rate": 0.0},
            {"name": "news_sentiment_5d", "feature_family": "news", "missing_rate": 0.05},
        ],
        training_frame=pd.DataFrame(),
        backtest_result={
            "backtest_id": uuid4().hex,
            "aggregate_metrics": {"hit_rate_mean": 0.45, "ece_mean": 0.03},
            "cost_adjusted_metrics": {"information_ratio": 1.5, "selection_stability": 0.9},
            "folds": [{"feature_importance": [{"feature": "x", "mean_abs_shap": 0.1}], "calibration": {"enabled": True}}],
            "feature_importance_summary": [{"feature": "x", "mean_abs_shap": 0.1}],
            "feature_family_importance_summary": [{"family": "price", "importance": 0.8}],
            "feature_catalog": [
                {"name": "log_return_1d", "feature_family": "price", "missing_rate": 0.0},
                {"name": "news_sentiment_5d", "feature_family": "news", "missing_rate": 0.05},
            ],
            "drift_monitor": {"max_psi": 0.02},
            "regime_monitor": {"current_regime": "normal"},
            "retraining_monitor": {"should_retrain": False},
            "cpcv": {"pbo_summary": {"label": "low"}, "cluster_adjusted_pbo": 0.0, "cluster_adjusted_pbo_summary": {"label": "low"}},
            "pbo": 0.1,
        },
        backtest_predictions=preds,
        latest_predictions=preds,
        model=model,
        drift_summary={"max_psi": 0.02, "mean_psi": 0.01},
        regime_summary={"current_regime": "normal", "state_probability": 0.9},
        retraining_monitor={"should_retrain": False, "state": "stable"},
    )


def _build_paper_trading_artifacts() -> _FakePaperTradingArtifacts:
    return _FakePaperTradingArtifacts(
        batch_log={
            "batch_id": uuid4().hex,
            "metrics": {"new_trades": 2, "avg_round_trip_cost_bps": 5.0},
            "execution_diagnostics": {"fill_rate": 1.0, "partial_fill_rate": 0.0, "missed_trade_rate": 0.0},
        },
        weekly_review={
            "review_id": uuid4().hex,
            "week_id": "2026-W15",
            "metrics": {"retraining_batches": 0, "liquidity_blocked_trades": 0},
            "execution_diagnostics": {
                "realized_vs_intended_exposure": 0.98,
                "execution_cost_drag": 1.2,
                "gap_slippage_bps": 0.5,
            },
        },
        retraining_event=None,
    )


@pytest.fixture()
def settings():
    return update_settings(
        load_settings(),
        data={"storage_path": f"./.test-artifacts/pipeline-unit-{uuid4().hex}/storage"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_pipeline_creates_agents(self, settings) -> None:
        pipeline = MarketPredictionPipeline(settings)
        assert pipeline.data_agent is not None
        assert pipeline.forecast_agent is not None
        assert pipeline.risk_agent is not None
        assert pipeline.report_agent is not None
        assert pipeline.paper_trading_service is not None
        assert pipeline.store is not None

    def test_default_tickers_delegates(self, settings) -> None:
        pipeline = MarketPredictionPipeline(settings)
        with patch.object(pipeline.data_agent, "default_tickers", return_value=["SPY", "QQQ"]):
            result = pipeline._default_tickers()
        assert result == ["SPY", "QQQ"]


class TestPipelineRun:
    def _run_pipeline(self, settings) -> PipelineRunResult:
        pipeline = MarketPredictionPipeline(settings)
        data_artifacts = _build_data_artifacts()
        forecast_artifacts = _build_forecast_artifacts()
        paper_artifacts = _build_paper_trading_artifacts()

        with (
            patch.object(pipeline.data_agent, "default_tickers", return_value=["T0", "T1"]),
            patch.object(pipeline.data_agent, "generate_or_fetch", return_value=data_artifacts),
            patch.object(pipeline.forecast_agent, "run", return_value=forecast_artifacts),
            patch.object(pipeline.risk_agent, "review", return_value={"review_id": uuid4().hex, "approval": True, "flags": []}),
            patch.object(pipeline.report_agent, "generate", return_value={"report_id": uuid4().hex, "sections": []}),
            patch.object(pipeline.paper_trading_service, "update", return_value=paper_artifacts),
            patch("market_prediction_agent.pipeline.validate_payload"),
            patch("market_prediction_agent.pipeline.build_forecast_output", return_value={"forecast_id": uuid4().hex, "generated_at": "2026-04-10T00:00:00Z", "predictions": []}),
            patch("market_prediction_agent.pipeline.build_evidence_bundle", return_value={"bundle_id": uuid4().hex, "model_info": {}, "data_snapshot": {}}),
            patch("market_prediction_agent.pipeline.build_news_feature_utility_comparison", return_value={}),
        ):
            return pipeline.run()

    def test_run_returns_pipeline_result(self, settings) -> None:
        result = self._run_pipeline(settings)
        assert isinstance(result, PipelineRunResult)

    def test_run_populates_all_fields(self, settings) -> None:
        result = self._run_pipeline(settings)
        assert result.backtest_result is not None
        assert result.forecast_output is not None
        assert result.evidence_bundle is not None
        assert result.risk_review is not None
        assert result.report_payload is not None
        assert result.paper_trading_batch is not None
        assert result.weekly_review is not None

    def test_run_with_explicit_tickers(self, settings) -> None:
        pipeline = MarketPredictionPipeline(settings)
        data_artifacts = _build_data_artifacts()
        forecast_artifacts = _build_forecast_artifacts()
        paper_artifacts = _build_paper_trading_artifacts()

        with (
            patch.object(pipeline.data_agent, "generate_or_fetch", return_value=data_artifacts) as mock_fetch,
            patch.object(pipeline.forecast_agent, "run", return_value=forecast_artifacts),
            patch.object(pipeline.risk_agent, "review", return_value={"review_id": uuid4().hex, "approval": True, "flags": []}),
            patch.object(pipeline.report_agent, "generate", return_value={"report_id": uuid4().hex, "sections": []}),
            patch.object(pipeline.paper_trading_service, "update", return_value=paper_artifacts),
            patch("market_prediction_agent.pipeline.validate_payload"),
            patch("market_prediction_agent.pipeline.build_forecast_output", return_value={"forecast_id": uuid4().hex, "generated_at": "2026-04-10T00:00:00Z", "predictions": []}),
            patch("market_prediction_agent.pipeline.build_evidence_bundle", return_value={"bundle_id": uuid4().hex, "model_info": {}, "data_snapshot": {}}),
            patch("market_prediction_agent.pipeline.build_news_feature_utility_comparison", return_value={}),
        ):
            pipeline.run(tickers=["AAPL", "GOOG"])

        call_kwargs = mock_fetch.call_args
        assert call_kwargs[1]["tickers"] == ["AAPL", "GOOG"] or call_kwargs.kwargs["tickers"] == ["AAPL", "GOOG"]

    def test_run_passes_retraining_history(self, settings) -> None:
        pipeline = MarketPredictionPipeline(settings)
        data_artifacts = _build_data_artifacts()
        forecast_artifacts = _build_forecast_artifacts()
        paper_artifacts = _build_paper_trading_artifacts()
        history = [{"event": "retrain", "date": "2026-03-01"}]

        with (
            patch.object(pipeline.data_agent, "default_tickers", return_value=["T0"]),
            patch.object(pipeline.data_agent, "generate_or_fetch", return_value=data_artifacts),
            patch.object(pipeline.forecast_agent, "run", return_value=forecast_artifacts) as mock_forecast,
            patch.object(pipeline.risk_agent, "review", return_value={"review_id": uuid4().hex, "approval": True, "flags": []}),
            patch.object(pipeline.report_agent, "generate", return_value={"report_id": uuid4().hex, "sections": []}),
            patch.object(pipeline.paper_trading_service, "update", return_value=paper_artifacts),
            patch("market_prediction_agent.pipeline.validate_payload"),
            patch("market_prediction_agent.pipeline.build_forecast_output", return_value={"forecast_id": uuid4().hex, "generated_at": "2026-04-10T00:00:00Z", "predictions": []}),
            patch("market_prediction_agent.pipeline.build_evidence_bundle", return_value={"bundle_id": uuid4().hex, "model_info": {}, "data_snapshot": {}}),
            patch("market_prediction_agent.pipeline.build_news_feature_utility_comparison", return_value={}),
        ):
            pipeline.run(retraining_policy_history=history)

        assert mock_forecast.call_args.kwargs["retraining_policy_history"] == history


class TestPersistOutputs:
    def test_persist_creates_files(self, settings) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        try:
            settings = update_settings(settings, data={"storage_path": str(tmp_path / "storage")})
            pipeline = MarketPredictionPipeline(settings)
            forecast_id = uuid4().hex
            bundle_id = uuid4().hex
            review_id = uuid4().hex
            report_id = uuid4().hex
            backtest_id = uuid4().hex
            batch_id = uuid4().hex
            weekly_id = uuid4().hex

            with patch("market_prediction_agent.pipeline.validate_payload"):
                pipeline._persist_outputs(
                    forecast_output={"forecast_id": forecast_id, "generated_at": "2026-04-10T00:00:00Z"},
                    evidence_bundle={"bundle_id": bundle_id},
                    risk_review={"review_id": review_id},
                    report_payload={"report_id": report_id},
                    backtest_result={"backtest_id": backtest_id},
                    paper_trading_batch={"batch_id": batch_id},
                    weekly_review={"review_id": weekly_id, "week_id": "2026-W15"},
                    retraining_event=None,
                )

            storage = tmp_path / "storage"
            assert (storage / "outputs" / "forecasts" / "2026-04-10" / f"{forecast_id}.json").exists()
            assert (storage / "outputs" / "evidence" / "2026-04-10" / f"{bundle_id}.json").exists()
            assert (storage / "outputs" / "risk_reviews" / "2026-04-10" / f"{review_id}.json").exists()
            assert (storage / "outputs" / "reports" / "2026-04-10" / f"{report_id}.json").exists()
            assert (storage / "outputs" / "backtests" / f"{backtest_id}.json").exists()
            assert (storage / "outputs" / "paper_trading" / "2026-04-10" / f"{batch_id}.json").exists()
            assert (storage / "outputs" / "weekly_reviews" / "2026-W15" / f"{weekly_id}.json").exists()
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_persist_writes_valid_json(self, settings) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        try:
            settings = update_settings(settings, data={"storage_path": str(tmp_path / "storage")})
            pipeline = MarketPredictionPipeline(settings)
            forecast_id = uuid4().hex

            with patch("market_prediction_agent.pipeline.validate_payload"):
                pipeline._persist_outputs(
                    forecast_output={"forecast_id": forecast_id, "generated_at": "2026-04-10T00:00:00Z", "data": [1, 2, 3]},
                    evidence_bundle={"bundle_id": uuid4().hex},
                    risk_review={"review_id": uuid4().hex},
                    report_payload={"report_id": uuid4().hex},
                    backtest_result={"backtest_id": uuid4().hex},
                    paper_trading_batch={"batch_id": uuid4().hex},
                    weekly_review={"review_id": uuid4().hex, "week_id": "2026-W15"},
                    retraining_event=None,
                )

            forecast_path = tmp_path / "storage" / "outputs" / "forecasts" / "2026-04-10" / f"{forecast_id}.json"
            parsed = json.loads(forecast_path.read_text(encoding="utf-8"))
            assert parsed["data"] == [1, 2, 3]
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_persist_with_retraining_event(self, settings) -> None:
        tmp_path = Path(tempfile.mkdtemp())
        try:
            settings = update_settings(settings, data={"storage_path": str(tmp_path / "storage")})
            pipeline = MarketPredictionPipeline(settings)
            event_id = uuid4().hex

            with patch("market_prediction_agent.pipeline.validate_payload"):
                pipeline._persist_outputs(
                    forecast_output={"forecast_id": uuid4().hex, "generated_at": "2026-04-10T00:00:00Z"},
                    evidence_bundle={"bundle_id": uuid4().hex},
                    risk_review={"review_id": uuid4().hex},
                    report_payload={"report_id": uuid4().hex},
                    backtest_result={"backtest_id": uuid4().hex},
                    paper_trading_batch={"batch_id": uuid4().hex},
                    weekly_review={"review_id": uuid4().hex, "week_id": "2026-W15"},
                    retraining_event={"event_id": event_id, "type": "drift"},
                )

            retraining_path = tmp_path / "storage" / "outputs" / "retraining_events" / "2026-04-10" / f"{event_id}.json"
            assert retraining_path.exists()
            parsed = json.loads(retraining_path.read_text(encoding="utf-8"))
            assert parsed["type"] == "drift"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
