from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from market_prediction_agent.config import LSTMConfig, Settings, load_settings, update_settings
from market_prediction_agent.features.labels import make_direction_label
import market_prediction_agent.models.factory as model_factory
from market_prediction_agent.models.lstm_model import LSTMDirectionModel

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
            "lstm": {
                "hidden_size": 8,
                "num_layers": 2,
                "dropout": 0.1,
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


def test_lstm_module_import_is_safe_when_torch_is_missing() -> None:
    module_name = "market_prediction_agent.models.lstm_model"
    with pytest.MonkeyPatch.context() as patch:
        sys.modules.pop(module_name, None)
        patch.setitem(sys.modules, "torch", None)
        module = importlib.import_module(module_name)
        assert module.torch is None
        assert module.LSTMDirectionModel is None
        assert module.LSTM_IMPORT_ERROR is not None
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


def test_create_model_raises_clear_error_when_torch_is_missing(monkeypatch) -> None:
    settings = _test_settings()
    monkeypatch.setattr(model_factory, "LSTMDirectionModel", None)
    monkeypatch.setattr(model_factory, "LSTM_IMPORT_ERROR", ImportError("No module named 'torch'"))

    with pytest.raises(RuntimeError, match="lstm_direction requires PyTorch"):
        model_factory.create_model(
            settings=settings,
            model_name="lstm_direction",
            version="test",
        )


def test_lstm_config_dataclass_initializes_expected_defaults() -> None:
    config = LSTMConfig(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        sequence_length=20,
        max_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        patience=5,
    )
    settings = load_settings("config/default.yaml")

    assert config.hidden_size == 64
    assert config.num_layers == 2
    assert settings.model_settings.lstm == config


@pytest.mark.skipif(torch is None or LSTMDirectionModel is None, reason="torch is not installed")
def test_lstm_fit_predict_returns_normalized_probabilities() -> None:
    settings = _test_settings()
    frame = _build_training_frame(threshold=settings.model_settings.direction_threshold)
    eval_frame = frame.groupby("ticker").tail(2).reset_index(drop=True)

    model = LSTMDirectionModel(settings=settings, version="test")
    model.fit(frame, FEATURE_COLUMNS)
    predictions = model.predict(eval_frame[["ticker", "date", "stale_data_flag", *FEATURE_COLUMNS]].copy())

    assert len(predictions) == len(eval_frame)
    row_sums = predictions[PROBABILITY_COLUMNS].sum(axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(row_sums, np.ones(len(predictions)), rtol=0.0, atol=1e-5)
    assert predictions["direction"].isin({"DOWN", "FLAT", "UP"}).all()
    assert np.isfinite(predictions["expected_return"].to_numpy(dtype=float)).all()
    assert np.isfinite(predictions["predicted_volatility"].to_numpy(dtype=float)).all()
    assert all(features == [] for features in predictions["top_features"])

