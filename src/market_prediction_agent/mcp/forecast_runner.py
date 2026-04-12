from __future__ import annotations

from typing import Any

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.agents.forecast_agent import ForecastAgent
from market_prediction_agent.config import Settings, update_settings
from market_prediction_agent.schemas.validator import validate_payload

from .base import MCPTool, clone_settings, history_window_bounds, temporary_store


FORECAST_RUNNER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tickers": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "as_of_date": {"type": "string", "format": "date"},
        "model_name": {"type": "string", "minLength": 1},
    },
    "required": ["tickers", "as_of_date"],
    "additionalProperties": False,
}


def handle_forecast_runner(params: dict[str, Any], settings: Settings) -> dict[str, Any]:
    tickers = [str(item).upper() for item in params["tickers"]]
    tool_settings = clone_settings(settings)
    model_name = params.get("model_name")
    if model_name is not None:
        tool_settings = update_settings(tool_settings, model_settings={"primary": str(model_name)})

    start_date, end_date, as_of_timestamp = history_window_bounds(
        params["as_of_date"],
        tool_settings.data.dummy_days,
    )
    with temporary_store() as store:
        data_agent = DataAgent(tool_settings, store)
        artifacts = data_agent.generate_or_fetch(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            as_of_time=as_of_timestamp,
        )

    forecast_output = ForecastAgent(tool_settings).predict(
        artifacts.processed_ohlcv,
        artifacts.processed_macro,
        artifacts.processed_news,
        artifacts.processed_fundamentals,
        artifacts.processed_sector_map,
        tickers=tickers,
        as_of_time=as_of_timestamp,
        source_metadata=artifacts.ohlcv_metadata,
    )
    validate_payload("forecast_output", forecast_output)
    return forecast_output


FORECAST_RUNNER = MCPTool(
    name="forecast_runner",
    description="Run the configured forecasting pipeline and return a schema-compliant forecast_output payload.",
    input_schema=FORECAST_RUNNER_INPUT_SCHEMA,
    handler=handle_forecast_runner,
)

