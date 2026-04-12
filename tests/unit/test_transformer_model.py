from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from market_prediction_agent.config import Settings, TransformerConfig, load_settings, update_settings
from market_prediction_agent.features.labels import make_direction_label
import market_prediction_agent.models.factory as model_factory
from market_prediction_agent.models.transformer_model import TransformerDirectionModel

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


FEATURE_COLUMNS = ["feature_a", "feature_b", "feature_c", "feature_d"]
PROBABILITY_COLUMNS = ["prob_down", "prob_flat", "prob_up"]


def _test_settings() -> Settings:
    return update_settings(
        load_settings("config/default.yaml"),
        model_settings={
            "transformer": {
                "d_model": 16,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 32,
                "dropout": 0.1,
                "patch_length": 2,
                "sequence_length": 5,
                "max_epochs": 3,
                "batch_size": 8,
                "learning_rate": 0.01,
                "patience": 2,
            }
        },
    )


def _build_training_frame(
    *,
    ticker_count: int = 6,
    day_count: int = 24,
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


def test_transformer_module_import_is_safe_when_torch_is_missing() -> None:
    module_name = "market_prediction_agent.models.transformer_model"
    with pytest.MonkeyPatch.context() as patch:
        sys.modules.pop(module_name, None)
        patch.setitem(sys.modules, "torch", None)
        module = importlib.import_module(module_name)
        assert module.torch is None
        assert module.TransformerDirectionModel is None
        assert module.TRANSFORMER_IMPORT_ERROR is not None
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


def test_create_model_raises_clear_error_when_torch_is_missing(monkeypatch) -> None:
    settings = _test_settings()
    monkeypatch.setattr(model_factory, "TransformerDirectionModel", None)
    monkeypatch.setattr(model_factory, "TRANSFORMER_IMPORT_ERROR", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="transformer_direction requires PyTorch"):
        model_factory.create_model(
            settings=settings,
            model_name="transformer_direction",
            version="test",
        )


def test_transformer_config_dataclass_initializes_expected_defaults() -> None:
    config = TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        patch_length=5,
        sequence_length=20,
        max_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        patience=5,
    )
    settings = load_settings("config/default.yaml")

    assert config.d_model == 64
    assert config.patch_length == 5
    assert settings.model_settings.transformer == config


@pytest.mark.skipif(torch is None or TransformerDirectionModel is None, reason="torch is not installed")
def test_transformer_fit_predict_returns_normalized_probabilities() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.groupby("ticker").tail(2).reset_index(drop=True)

    model = TransformerDirectionModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(
        eval_frame[["ticker", "date", "stale_data_flag", *FEATURE_COLUMNS]].copy(),
        include_explanations=False,
    )

    assert len(predictions) == len(eval_frame)
    row_sums = predictions[PROBABILITY_COLUMNS].sum(axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(row_sums, np.ones(len(predictions)), rtol=0.0, atol=1e-5)
    assert predictions["direction"].isin({"DOWN", "FLAT", "UP"}).all()
    assert np.isfinite(predictions["expected_return"].to_numpy(dtype=float)).all()
    assert np.isfinite(predictions["predicted_volatility"].to_numpy(dtype=float)).all()
    assert all(features == [] for features in predictions["top_features"])


@pytest.mark.skipif(torch is None or TransformerDirectionModel is None, reason="torch is not installed")
def test_transformer_calibration_summary_is_populated_after_fit() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)

    model = TransformerDirectionModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)

    assert model.calibration_summary["enabled"] is True
    assert int(model.calibration_summary["samples"]) > 0


@pytest.mark.skipif(torch is None or TransformerDirectionModel is None, reason="torch is not installed")
def test_transformer_feature_importance_returns_nonempty_list() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)

    model = TransformerDirectionModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    feature_importance = model.feature_importance_top()

    assert feature_importance
    assert all("feature" in item and "mean_abs_shap" in item for item in feature_importance)


@pytest.mark.skipif(torch is None or TransformerDirectionModel is None, reason="torch is not installed")
def test_transformer_top_features_returns_per_sample_attributions() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.groupby("ticker").tail(2).reset_index(drop=True)
    predict_frame = eval_frame[["ticker", "date", "stale_data_flag", *FEATURE_COLUMNS]].copy()

    model = TransformerDirectionModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions_with_explanations = model.predict(predict_frame.copy(), include_explanations=True)
    predictions_without_explanations = model.predict(predict_frame.copy(), include_explanations=False)

    assert all(len(features) > 0 for features in predictions_with_explanations["top_features"])
    assert all(features == [] for features in predictions_without_explanations["top_features"])

