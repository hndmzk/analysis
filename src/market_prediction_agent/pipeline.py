from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import cast

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.agents.forecast_agent import ForecastAgent
from market_prediction_agent.agents.report_agent import ReportAgent
from market_prediction_agent.agents.risk_agent import RiskAgent
from market_prediction_agent.config import Settings, resolve_storage_path
from market_prediction_agent.evaluation.news_analysis import build_news_feature_utility_comparison
from market_prediction_agent.paper_trading.service import PaperTradingService
from market_prediction_agent.reporting.builders import build_evidence_bundle, build_forecast_output
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.time_utils import to_utc_timestamp


@dataclass(slots=True)
class PipelineRunResult:
    backtest_result: dict[str, object]
    forecast_output: dict[str, object]
    evidence_bundle: dict[str, object]
    risk_review: dict[str, object]
    report_payload: dict[str, object]
    paper_trading_batch: dict[str, object]
    weekly_review: dict[str, object]
    retraining_event: dict[str, object] | None


class MarketPredictionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage_path = resolve_storage_path(settings)
        self.store = ParquetStore(self.storage_path)
        self.data_agent = DataAgent(settings, self.store)
        self.forecast_agent = ForecastAgent(settings)
        self.risk_agent = RiskAgent(settings)
        self.report_agent = ReportAgent()
        self.paper_trading_service = PaperTradingService(settings, self.store)

    def _default_tickers(self) -> list[str]:
        return self.data_agent.default_tickers()

    def run(
        self,
        tickers: list[str] | None = None,
        *,
        as_of_time: str | pd.Timestamp | None = None,
        retraining_policy_history: list[dict[str, object]] | None = None,
    ) -> PipelineRunResult:
        as_of_timestamp = to_utc_timestamp(as_of_time or pd.Timestamp.now(tz="UTC"))
        end_date = as_of_timestamp.normalize()
        start_date = end_date - pd.tseries.offsets.BDay(self.settings.data.dummy_days - 1)
        selected_tickers = tickers or self._default_tickers()
        data = self.data_agent.generate_or_fetch(
            tickers=selected_tickers,
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            as_of_time=as_of_timestamp,
        )
        news_feature_source = cast(dict[str, object], data.ohlcv_metadata.get("feature_sources", {})).get("news")
        if isinstance(news_feature_source, dict):
            news_feature_source["utility_comparison"] = build_news_feature_utility_comparison(
                ohlcv=data.processed_ohlcv,
                news=data.processed_news,
                weighting_mode=self.settings.data.news_weighting_mode,
                learned_weighting=self.settings.data.learned_weighting,
            )
        forecast = self.forecast_agent.run(
            data.processed_ohlcv,
            data.processed_macro,
            data.processed_news,
            data.processed_fundamentals,
            data.processed_sector_map,
            tickers=selected_tickers,
            as_of_time=as_of_timestamp,
            retraining_policy_history=retraining_policy_history,
            source_metadata=data.ohlcv_metadata,
        )
        if isinstance(news_feature_source, dict):
            news_catalog = [
                item
                for item in cast(list[dict[str, object]], forecast.backtest_result.get("feature_catalog", []))
                if str(item.get("feature_family", "")) == "news"
            ]
            if news_catalog:
                news_feature_source["missing_rate"] = float(
                    sum(float(cast(float, item.get("missing_rate", 0.0))) for item in news_catalog)
                    / max(len(news_catalog), 1)
                )
        news_summary = []
        for _, row in (
            data.processed_news.sort_values(["available_at", "relevance_score"], ascending=[False, False]).head(10).iterrows()
        ):
            news_summary.append(
                {
                    "headline": f"{row['ticker']} offline news proxy",
                    "source": row["source"],
                    "published_at": pd.to_datetime(row["published_at"], utc=True).isoformat(),
                    "sentiment_score": float(row["sentiment_score"]),
                    "relevance_score": float(row["relevance_score"]),
                }
            )
        forecast_output = build_forecast_output(
            predictions=forecast.latest_predictions,
            model_version=self.settings.model_settings.version,
            horizon=self.settings.data.forecast_horizon,
            regime=cast(str, forecast.regime_summary["current_regime"]),
        )
        stale_tickers = forecast.latest_predictions.loc[forecast.latest_predictions["stale_data_flag"], "ticker"].tolist()
        evidence_bundle = build_evidence_bundle(
            forecast_id=cast(str, forecast_output["forecast_id"]),
            features=forecast.feature_frame.loc[
                forecast.feature_frame["date"] == forecast.feature_frame["date"].max(),
                ["ticker", "date"] + forecast.model.feature_columns,
            ],
            macro=data.processed_macro,
            model_name=self.settings.model_settings.primary,
            model_version=self.settings.model_settings.version,
            trained_at=forecast.model.trained_at or pd.Timestamp.now(tz="UTC").isoformat(),
            training_samples=forecast.model.training_samples,
            stale_tickers=stale_tickers,
            source_metadata=data.ohlcv_metadata,
            hyperparameters=forecast.model.hyperparameters(),
            feature_importance=cast(list[dict[str, object]], forecast.backtest_result.get("feature_importance_summary", [])),
            feature_catalog=forecast.feature_catalog,
            feature_family_importance=cast(
                list[dict[str, object]],
                forecast.backtest_result.get("feature_family_importance_summary", []),
            ),
            calibration_summary=forecast.model.calibration_summary,
            drift_summary=forecast.drift_summary,
            regime_summary=forecast.regime_summary,
            retraining_monitor=forecast.retraining_monitor,
            news_summary=news_summary,
        )
        risk_review = self.risk_agent.review(forecast_output, forecast.backtest_result)
        paper_trading = self.paper_trading_service.update(
            forecast_output=forecast_output,
            backtest_result=forecast.backtest_result,
            risk_review=risk_review,
            source_metadata=data.ohlcv_metadata,
            feature_frame=forecast.feature_frame,
        )
        report_payload = self.report_agent.generate(forecast_output, forecast.backtest_result, risk_review)
        self._persist_outputs(
            forecast_output,
            evidence_bundle,
            risk_review,
            report_payload,
            forecast.backtest_result,
            paper_trading.batch_log,
            paper_trading.weekly_review,
            paper_trading.retraining_event,
        )
        return PipelineRunResult(
            backtest_result=forecast.backtest_result,
            forecast_output=forecast_output,
            evidence_bundle=evidence_bundle,
            risk_review=risk_review,
            report_payload=report_payload,
            paper_trading_batch=paper_trading.batch_log,
            weekly_review=paper_trading.weekly_review,
            retraining_event=paper_trading.retraining_event,
        )

    def _persist_outputs(
        self,
        forecast_output: dict[str, object],
        evidence_bundle: dict[str, object],
        risk_review: dict[str, object],
        report_payload: dict[str, object],
        backtest_result: dict[str, object],
        paper_trading_batch: dict[str, object],
        weekly_review: dict[str, object],
        retraining_event: dict[str, object] | None,
    ) -> None:
        validate_payload("forecast_output", forecast_output)
        validate_payload("evidence_bundle", evidence_bundle)
        validate_payload("risk_review", risk_review)
        validate_payload("report_payload", report_payload)
        validate_payload("backtest_result", backtest_result)
        validate_payload("paper_trading_batch", paper_trading_batch)
        validate_payload("weekly_review", weekly_review)
        if retraining_event is not None:
            validate_payload("retraining_event", retraining_event)
        forecast_date = pd.Timestamp(forecast_output["generated_at"]).date().isoformat()
        week_id = cast(str, weekly_review["week_id"])
        outputs = [
            (Path("outputs") / "forecasts" / forecast_date / f"{forecast_output['forecast_id']}.json", forecast_output),
            (Path("outputs") / "evidence" / forecast_date / f"{evidence_bundle['bundle_id']}.json", evidence_bundle),
            (Path("outputs") / "risk_reviews" / forecast_date / f"{risk_review['review_id']}.json", risk_review),
            (Path("outputs") / "reports" / forecast_date / f"{report_payload['report_id']}.json", report_payload),
            (Path("outputs") / "backtests" / f"{backtest_result['backtest_id']}.json", backtest_result),
            (Path("outputs") / "paper_trading" / forecast_date / f"{paper_trading_batch['batch_id']}.json", paper_trading_batch),
            (Path("outputs") / "weekly_reviews" / week_id / f"{weekly_review['review_id']}.json", weekly_review),
        ]
        if retraining_event is not None:
            outputs.append(
                (Path("outputs") / "retraining_events" / forecast_date / f"{retraining_event['event_id']}.json", retraining_event)
            )
        for relative_path, payload in outputs:
            target = self.storage_path / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

