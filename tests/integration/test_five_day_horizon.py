from __future__ import annotations

import numpy as np
import pandas as pd

from market_prediction_agent.config import Settings, load_settings, update_settings
from market_prediction_agent.data.adapters import (
    DummyMacroAdapter,
    DummyOHLCVAdapter,
    FundamentalsRequest,
    MacroRequest,
    NewsRequest,
    OHLCVRequest,
    OfflineFundamentalProxyAdapter,
    OfflineNewsProxyAdapter,
    SectorRequest,
    StaticSectorMapAdapter,
)
from market_prediction_agent.data.normalizer import (
    normalize_fundamentals,
    normalize_macro,
    normalize_news,
    normalize_ohlcv,
    normalize_sector_map,
)
from market_prediction_agent.features.pipeline import build_feature_frame, build_training_frame
from market_prediction_agent.models.factory import build_model


FIVE_DAY_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "realized_vol_5d",
    "realized_vol_20d",
    "garman_klass_vol",
    "rsi_14",
    "roc_20d",
    "volume_ratio_20d",
    "price_vs_sma_20d",
    "bb_width",
    "atr_ratio",
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


def _five_day_test_settings() -> Settings:
    return update_settings(
        load_settings("config/default.yaml"),
        data={
            "horizon_days": 5,
            "forecast_horizon": "5d",
        },
        model_settings={
            "calibration": {
                "min_days": 10,
                "fraction": 0.2,
            },
            "lightgbm": {
                "n_estimators": 8,
                "num_leaves": 7,
                "min_child_samples": 5,
                "max_shap_samples": 40,
            },
        },
    )


def test_five_day_horizon_produces_regression_outputs(monkeypatch) -> None:
    monkeypatch.setattr("market_prediction_agent.features.pipeline.FEATURE_COLUMNS", FIVE_DAY_FEATURE_COLUMNS)
    settings = _five_day_test_settings()
    tickers = [f"TICK{i:02d}" for i in range(10)]
    dates = pd.bdate_range("2024-01-01", periods=50)
    start_date = dates[0].date().isoformat()
    end_date = dates[-1].date().isoformat()
    ohlcv = normalize_ohlcv(
        DummyOHLCVAdapter(seed=42).fetch(
            OHLCVRequest(tickers=tickers, start_date=start_date, end_date=end_date)
        )
    )
    macro = normalize_macro(
        DummyMacroAdapter(seed=42).fetch(
            MacroRequest(series_ids=["FEDFUNDS", "T10Y2Y", "VIXCLS"], start_date=start_date, end_date=end_date)
        )
    )
    news = normalize_news(
        OfflineNewsProxyAdapter(seed=42, mode="null_random_walk").fetch(
            NewsRequest(tickers=tickers, start_date=start_date, end_date=end_date)
        )
    )
    fundamentals = normalize_fundamentals(
        OfflineFundamentalProxyAdapter(seed=42).fetch(
            FundamentalsRequest(tickers=tickers, start_date=start_date, end_date=end_date)
        )
    )
    sector_map = normalize_sector_map(StaticSectorMapAdapter().fetch(SectorRequest(tickers=tickers)))

    feature_result = build_feature_frame(
        ohlcv=ohlcv,
        macro=macro,
        news=news,
        fundamentals=fundamentals,
        sector_map=sector_map,
        horizon_days=settings.data.horizon_days,
        direction_threshold=settings.model_settings.direction_threshold,
    )
    feature_frame = feature_result.feature_frame
    training_frame = build_training_frame(feature_frame)
    assert not training_frame.empty

    sample_ticker_frame = feature_frame.loc[feature_frame["ticker"] == tickers[0]].sort_values("date").reset_index(drop=True)
    sample_index = 24
    manual_five_day_return = sample_ticker_frame.loc[sample_index + 5, "close"] / sample_ticker_frame.loc[sample_index, "close"] - 1.0
    assert sample_ticker_frame.loc[sample_index, "future_simple_return"] == manual_five_day_return

    model = build_model(
        settings=settings,
        model_name=settings.model_settings.primary,
        version="test",
    )
    model.fit(training_frame, FIVE_DAY_FEATURE_COLUMNS)
    eval_frame = training_frame.loc[
        training_frame["date"] == training_frame["date"].max(),
        ["ticker", "date", "stale_data_flag", *FIVE_DAY_FEATURE_COLUMNS],
    ].copy()

    predictions = model.predict(eval_frame, include_explanations=False)

    assert {"expected_return", "predicted_volatility"}.issubset(predictions.columns)
    assert np.isfinite(predictions["expected_return"].to_numpy(dtype=float)).all()
    assert np.isfinite(predictions["predicted_volatility"].to_numpy(dtype=float)).all()
    assert (predictions["predicted_volatility"].to_numpy(dtype=float) >= 0.0).all()

