from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_prediction_agent.config import Settings, load_settings, update_settings
from market_prediction_agent.features.labels import make_direction_label
from market_prediction_agent.models.base import ForecastModel
from market_prediction_agent.models.baseline import BaselineModel
from market_prediction_agent.models.factory import build_model, create_model
from market_prediction_agent.models.lightgbm_calibrated import (
    LightGBMCalibratedModel,
    _safe_softmax as lightgbm_safe_softmax,
)
from market_prediction_agent.models.xgboost_calibrated import (
    XGBoostCalibratedModel,
    _safe_softmax as xgboost_safe_softmax,
    xgb,
)


FEATURE_COLUMNS = ["feature_a", "feature_b", "feature_c", "feature_d"]
PROBABILITY_COLUMNS = ["prob_down", "prob_flat", "prob_up"]


def _test_settings() -> Settings:
    return update_settings(
        load_settings("config/default.yaml"),
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
            "xgboost": {
                "n_estimators": 8,
                "max_depth": 3,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "max_shap_samples": 40,
            },
        },
    )


def _build_training_frame(
    *,
    ticker_count: int = 20,
    day_count: int = 100,
    threshold: float = 0.005,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=day_count)
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for ticker_index in range(ticker_count):
            volatility = abs(np.cos((day_index - ticker_index) / 6.0))
            future_simple_return = 0.02 * np.sin((day_index + 2 * ticker_index) / 4.5)
            rows.append(
                {
                    "ticker": f"T{ticker_index:03d}",
                    "date": date,
                    "stale_data_flag": False,
                    "feature_a": np.sin(day_index / 4.0) + np.cos(ticker_index / 3.0),
                    "feature_b": np.sin((day_index + ticker_index) / 5.0),
                    "feature_c": volatility,
                    "feature_d": ((ticker_index % 5) - 2) / 3.0,
                    "future_simple_return": future_simple_return,
                    "target_return": np.log1p(future_simple_return),
                    "future_volatility_20d": 0.01 + 0.02 * volatility,
                }
            )
    frame = pd.DataFrame(rows)
    frame["direction_label"] = make_direction_label(frame["future_simple_return"], threshold=threshold).astype(int)
    return frame


def _build_balanced_constant_frame() -> pd.DataFrame:
    returns = np.array([-0.02, 0.0, 0.02] * 12, dtype=float)
    dates = pd.bdate_range("2024-01-01", periods=len(returns))
    frame = pd.DataFrame(
        {
            "ticker": [f"T{index:03d}" for index in range(len(returns))],
            "date": dates,
            "stale_data_flag": False,
            "feature_a": 0.0,
            "feature_b": 0.0,
            "feature_c": 0.0,
            "feature_d": 0.0,
            "future_simple_return": returns,
            "target_return": np.log1p(returns),
            "future_volatility_20d": 0.02,
        }
    )
    frame["direction_label"] = make_direction_label(frame["future_simple_return"], threshold=0.005).astype(int)
    return frame


def _assert_probability_frame(predictions: pd.DataFrame) -> None:
    row_sums = predictions[PROBABILITY_COLUMNS].sum(axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(row_sums, np.ones(len(predictions)), rtol=0.0, atol=1e-6)
    assert predictions["direction"].isin({"DOWN", "FLAT", "UP"}).all()


def test_forecast_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        ForecastModel()


def test_lightgbm_fit_sets_booster_and_feature_columns() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)

    model = LightGBMCalibratedModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)

    assert model.classifier is not None
    assert model.classifier.booster_ is not None
    assert model.feature_columns == FEATURE_COLUMNS
    assert model.training_samples == len(frame)
    assert model.fill_values is not None


def test_lightgbm_predict_returns_normalized_probabilities_and_shap_explanations() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.tail(12).reset_index(drop=True)

    model = LightGBMCalibratedModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    _assert_probability_frame(predictions)
    assert all(len(features) == len(FEATURE_COLUMNS) for features in predictions["top_features"])
    assert set(model.global_feature_importance_map) == set(FEATURE_COLUMNS)


def test_lightgbm_fit_enables_calibration_when_history_is_large_enough() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.tail(12).reset_index(drop=True)

    model = LightGBMCalibratedModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    assert model.calibrator is not None
    assert model.calibration_summary["enabled"] is True
    _assert_probability_frame(predictions)


def test_lightgbm_small_history_falls_back_without_calibrator() -> None:
    settings = update_settings(
        _test_settings(),
        model_settings={
            "calibration": {
                "min_days": 20,
                "fraction": 0.5,
            },
            "lightgbm": {
                "min_child_samples": 2,
            },
        },
    )
    frame = _build_training_frame(ticker_count=6, day_count=8, threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.tail(12).reset_index(drop=True)

    model = LightGBMCalibratedModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    assert model.calibrator is None
    assert model.calibration_summary["enabled"] is False
    _assert_probability_frame(predictions)


def test_lightgbm_safe_softmax_is_numerically_stable() -> None:
    matrix = np.array([[10000.0, 10001.0, 10002.0], [-10000.0, -9999.0, -9998.0]], dtype=float)

    probabilities = lightgbm_safe_softmax(matrix)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.array([1.0, 1.0]), rtol=0.0, atol=1e-9)
    assert probabilities.argmax(axis=1).tolist() == [2, 2]


@pytest.mark.skipif(xgb is None, reason="xgboost is not installed")
def test_xgboost_fit_predict_and_calibration() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.tail(12).reset_index(drop=True)

    model = XGBoostCalibratedModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    assert model.classifier is not None
    assert model.calibrator is not None
    assert model.feature_columns == FEATURE_COLUMNS
    assert set(model.global_feature_importance_map) == set(FEATURE_COLUMNS)
    assert all(len(features) == len(FEATURE_COLUMNS) for features in predictions["top_features"])
    _assert_probability_frame(predictions)


def test_xgboost_safe_softmax_is_numerically_stable() -> None:
    matrix = np.array([[10000.0, 10001.0, 10002.0], [-10000.0, -9999.0, -9998.0]], dtype=float)

    probabilities = xgboost_safe_softmax(matrix)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.array([1.0, 1.0]), rtol=0.0, atol=1e-9)
    assert probabilities.argmax(axis=1).tolist() == [2, 2]


def test_baseline_model_fit_and_predict() -> None:
    frame = _build_training_frame()
    eval_frame = frame.tail(12).reset_index(drop=True)

    model = BaselineModel(version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    assert model.feature_columns == FEATURE_COLUMNS
    assert model.training_samples == len(frame)
    assert all(len(features) == len(FEATURE_COLUMNS) for features in predictions["top_features"])
    _assert_probability_frame(predictions)


def test_baseline_model_returns_uniform_probabilities_for_balanced_constant_features() -> None:
    frame = _build_balanced_constant_frame()
    eval_frame = frame.tail(6).reset_index(drop=True)

    model = BaselineModel(version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame)

    expected = np.full((len(predictions), 3), 1.0 / 3.0)
    np.testing.assert_allclose(predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float), expected, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(xgb is None, reason="xgboost is not installed")
def test_create_model_returns_expected_model_class() -> None:
    settings = _test_settings()

    lightgbm_model = create_model(
        settings=settings,
        model_name="lightgbm_multiclass_calibrated",
        version="test",
    )
    xgboost_model = create_model(
        settings=settings,
        model_name="xgboost_multiclass_calibrated",
        version="test",
    )

    assert isinstance(lightgbm_model, LightGBMCalibratedModel)
    assert isinstance(xgboost_model, XGBoostCalibratedModel)
    assert isinstance(
        build_model(settings=settings, model_name="lightgbm_multiclass_calibrated", version="test"),
        LightGBMCalibratedModel,
    )

