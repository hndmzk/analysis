from __future__ import annotations

from typing import Any

from market_prediction_agent.config import Settings
from market_prediction_agent.models.lightgbm_calibrated import LightGBMCalibratedModel
from market_prediction_agent.models.lstm_model import LSTMDirectionModel, LSTM_IMPORT_ERROR
from market_prediction_agent.models.transformer_model import TransformerDirectionModel, TRANSFORMER_IMPORT_ERROR
from market_prediction_agent.models.xgboost_calibrated import XGBoostCalibratedModel


def create_model(
    *,
    settings: Settings,
    model_name: str,
    version: str,
) -> Any:
    if model_name == "lightgbm_multiclass_calibrated":
        return LightGBMCalibratedModel(settings=settings, version=version)
    if model_name == "xgboost_multiclass_calibrated":
        return XGBoostCalibratedModel(settings=settings, version=version)
    if model_name == "lstm_direction":
        if LSTMDirectionModel is None:
            raise RuntimeError(
                "lstm_direction requires PyTorch. Install the optional dependency with "
                "`uv sync --extra ml-extra` or `pip install -e .[ml-extra]`."
                + (f" Original import error: {LSTM_IMPORT_ERROR}" if LSTM_IMPORT_ERROR is not None else "")
            )
        return LSTMDirectionModel(settings=settings, version=version)
    if model_name == "transformer_direction":
        if TransformerDirectionModel is None:
            raise RuntimeError(
                "transformer_direction requires PyTorch. Install the optional dependency with "
                "`uv sync --extra ml-extra` or `pip install -e .[ml-extra]`."
                + (
                    f" Original import error: {TRANSFORMER_IMPORT_ERROR}"
                    if TRANSFORMER_IMPORT_ERROR is not None
                    else ""
                )
            )
        return TransformerDirectionModel(settings=settings, version=version)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model(
    *,
    settings: Settings,
    model_name: str,
    version: str,
) -> Any:
    return create_model(settings=settings, model_name=model_name, version=version)
