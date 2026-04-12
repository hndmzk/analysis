from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from market_prediction_agent.backtest.walk_forward import run_model_comparisons, run_walk_forward_backtest
from market_prediction_agent.config import Settings
from market_prediction_agent.data.universe import resolve_active_constituents
from market_prediction_agent.evaluation.drift import compute_feature_drift
from market_prediction_agent.evaluation.regime import detect_regime
from market_prediction_agent.evaluation.retraining import build_retraining_monitor
from market_prediction_agent.features.pipeline import FEATURE_COLUMNS, build_feature_frame, build_training_frame
from market_prediction_agent.models.factory import build_model
from market_prediction_agent.retraining.ledger_service import RetrainingEventLedgerService
from market_prediction_agent.reporting.builders import build_forecast_output


DRIFT_EXCLUDED_FEATURES = {"day_of_week", "month", "is_month_end"}


@dataclass(slots=True)
class ForecastArtifacts:
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


class ForecastAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retraining_ledger_service = RetrainingEventLedgerService(settings)

    def run(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        news: pd.DataFrame,
        fundamentals: pd.DataFrame,
        sector_map: pd.DataFrame,
        *,
        tickers: list[str] | None = None,
        as_of_time: pd.Timestamp | None = None,
        retraining_policy_history: list[dict[str, object]] | None = None,
        source_metadata: dict[str, object] | None = None,
    ) -> ForecastArtifacts:
        feature_result = build_feature_frame(
            ohlcv=ohlcv,
            macro=macro,
            news=news,
            fundamentals=fundamentals,
            sector_map=sector_map,
            horizon_days=self.settings.data.horizon_days,
            direction_threshold=self.settings.model_settings.direction_threshold,
            source_metadata=source_metadata,
        )
        feature_frame = feature_result.feature_frame
        feature_catalog = feature_result.feature_catalog
        training_frame = build_training_frame(feature_frame)
        backtest_result, backtest_predictions = run_walk_forward_backtest(
            training_frame,
            FEATURE_COLUMNS,
            self.settings,
            feature_catalog=feature_catalog,
        )
        if self.settings.model_settings.comparison_models:
            backtest_result["model_comparison"] = run_model_comparisons(
                training_frame=training_frame,
                feature_columns=FEATURE_COLUMNS,
                settings=self.settings,
                primary_backtest_result=backtest_result,
                feature_catalog=feature_catalog,
            )
        latest_date = feature_frame["date"].max()
        latest_rows = feature_frame.loc[feature_frame["date"] == latest_date, ["ticker", "date", "stale_data_flag"] + FEATURE_COLUMNS].dropna()
        fit_frame = training_frame
        active_constituents = resolve_active_constituents(
            self.settings,
            as_of_date=as_of_time or latest_date,
        )
        if active_constituents is not None:
            filtered_fit_frame = training_frame.loc[training_frame["ticker"].isin(active_constituents)].copy()
            if not filtered_fit_frame.empty:
                fit_frame = filtered_fit_frame
            latest_rows = latest_rows.loc[latest_rows["ticker"].isin(active_constituents)].copy()
        model = build_model(
            settings=self.settings,
            model_name=self.settings.model_settings.primary,
            version=self.settings.model_settings.version,
        )
        model.fit(fit_frame, FEATURE_COLUMNS)
        latest_predictions = model.predict(latest_rows)
        regime_summary = detect_regime(feature_frame=feature_frame, macro=macro, settings=self.settings)
        drift_features = [feature for feature in FEATURE_COLUMNS if feature not in DRIFT_EXCLUDED_FEATURES]
        drift_window = self.settings.model_settings.walk_forward.eval_days
        feature_dates = sorted(feature_frame["date"].drop_duplicates().tolist())
        recent_dates = feature_dates[-drift_window:]
        reference_dates = feature_dates[-(2 * drift_window) : -drift_window]
        if not reference_dates:
            reference_dates = feature_dates[:-drift_window]
        drift_current_frame = feature_frame.loc[feature_frame["date"].isin(recent_dates), drift_features].dropna()
        drift_reference_frame = feature_frame.loc[feature_frame["date"].isin(reference_dates), drift_features].dropna()
        drift_summary = compute_feature_drift(
            reference_frame=drift_reference_frame,
            current_frame=drift_current_frame,
            feature_columns=drift_features,
            psi_warning=self.settings.risk.psi_warning,
            psi_critical=self.settings.risk.psi_critical,
            bucket_count=self.settings.risk.drift.bucket_count,
            proxy_ohlcv_used=bool(ohlcv["source"].eq("fred_market_proxy").any()),
            regime_summary=regime_summary,
            family_thresholds={
                "price_momentum": {
                    "warning": self.settings.risk.drift.price_momentum.warning,
                    "critical": self.settings.risk.drift.price_momentum.critical,
                },
                "volatility": {
                    "warning": self.settings.risk.drift.volatility.warning,
                    "critical": self.settings.risk.drift.volatility.critical,
                },
                "volume": {
                    "warning": self.settings.risk.drift.volume.warning,
                    "critical": self.settings.risk.drift.volume.critical,
                },
                "macro": {
                    "warning": self.settings.risk.drift.macro.warning,
                    "critical": self.settings.risk.drift.macro.critical,
                },
                "calendar": {
                    "warning": self.settings.risk.drift.calendar.warning,
                    "critical": self.settings.risk.drift.calendar.critical,
                },
            },
            proxy_sensitive_features=self.settings.risk.drift.proxy_sensitive_features,
            feature_catalog=feature_catalog,
        )
        effective_policy_history = retraining_policy_history
        if effective_policy_history is None and self.settings.data.source_mode == "live" and tickers and as_of_time is not None:
            effective_policy_history = self.retraining_ledger_service.load_policy_history(
                tickers=tickers,
                as_of_date=as_of_time.date().isoformat(),
            )
        cpcv_result = cast(dict[str, object], backtest_result.get("cpcv", {}))
        retraining_monitor = build_retraining_monitor(
            aggregate_metrics=cast(dict[str, float], backtest_result["aggregate_metrics"]),
            drift_summary=drift_summary,
            regime_summary=regime_summary,
            pbo=cast(float | None, cpcv_result.get("cluster_adjusted_pbo", backtest_result.get("pbo"))),
            pbo_summary=cast(
                dict[str, object],
                cpcv_result.get(
                    "cluster_adjusted_pbo_summary",
                    cast(dict[str, object], cpcv_result.get("pbo_summary", {})),
                ),
            ),
            pbo_diagnostics=cast(
                dict[str, object] | None,
                cpcv_result.get("cluster_adjusted_pbo_diagnostics", cpcv_result.get("pbo_diagnostics")),
            ),
            candidate_level_pbo=cast(float | None, backtest_result.get("pbo")),
            candidate_level_pbo_summary=cast(dict[str, object] | None, cpcv_result.get("pbo_summary")),
            candidate_level_pbo_diagnostics=cast(dict[str, object] | None, cpcv_result.get("pbo_diagnostics")),
            settings=self.settings,
            policy_context={
                "as_of_date": (as_of_time or pd.Timestamp.now(tz="UTC")).date().isoformat(),
                "history": effective_policy_history or [],
            },
        )
        backtest_result["drift_monitor"] = drift_summary
        backtest_result["regime_monitor"] = regime_summary
        backtest_result["retraining_monitor"] = retraining_monitor
        return ForecastArtifacts(
            feature_frame=feature_frame,
            feature_catalog=feature_catalog,
            training_frame=training_frame,
            backtest_result=backtest_result,
            backtest_predictions=backtest_predictions,
            latest_predictions=latest_predictions,
            model=model,
            drift_summary=drift_summary,
            regime_summary=regime_summary,
            retraining_monitor=retraining_monitor,
        )

    def predict(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        news: pd.DataFrame,
        fundamentals: pd.DataFrame,
        sector_map: pd.DataFrame,
        *,
        tickers: list[str] | None = None,
        as_of_time: pd.Timestamp | None = None,
        retraining_policy_history: list[dict[str, object]] | None = None,
        source_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        artifacts = self.run(
            ohlcv,
            macro,
            news,
            fundamentals,
            sector_map,
            tickers=tickers,
            as_of_time=as_of_time,
            retraining_policy_history=retraining_policy_history,
            source_metadata=source_metadata,
        )
        return build_forecast_output(
            predictions=artifacts.latest_predictions,
            model_version=self.settings.model_settings.version,
            horizon=self.settings.data.forecast_horizon,
            regime=cast(str, artifacts.regime_summary["current_regime"]),
        )

