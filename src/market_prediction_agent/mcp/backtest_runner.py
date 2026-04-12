from __future__ import annotations

from typing import Any

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.agents.forecast_agent import ForecastAgent
from market_prediction_agent.config import Settings, update_settings
from market_prediction_agent.schemas.validator import validate_payload
from market_prediction_agent.utils.time_utils import to_utc_timestamp

from .base import MCPTool, clone_settings, default_runner_tickers, temporary_store


BACKTEST_RUNNER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "minLength": 1},
        "start_date": {"type": "string", "format": "date"},
        "end_date": {"type": "string", "format": "date"},
    },
    "required": ["model_name", "start_date", "end_date"],
    "additionalProperties": False,
}


def handle_backtest_runner(params: dict[str, Any], settings: Settings) -> dict[str, Any]:
    tool_settings = update_settings(
        clone_settings(settings),
        model_settings={"primary": str(params["model_name"])},
    )
    tickers = default_runner_tickers(tool_settings)
    as_of_timestamp = to_utc_timestamp(str(params["end_date"])).normalize()

    with temporary_store() as store:
        data_agent = DataAgent(tool_settings, store)
        artifacts = data_agent.generate_or_fetch(
            tickers=tickers,
            start_date=str(params["start_date"]),
            end_date=str(params["end_date"]),
            as_of_time=as_of_timestamp,
        )

    forecast_artifacts = ForecastAgent(tool_settings).run(
        artifacts.processed_ohlcv,
        artifacts.processed_macro,
        artifacts.processed_news,
        artifacts.processed_fundamentals,
        artifacts.processed_sector_map,
        tickers=tickers,
        as_of_time=as_of_timestamp,
        source_metadata=artifacts.ohlcv_metadata,
    )
    validate_payload("backtest_result", forecast_artifacts.backtest_result)
    return forecast_artifacts.backtest_result


BACKTEST_RUNNER = MCPTool(
    name="backtest_runner",
    description="Run the existing walk-forward backtest stack and return a schema-compliant backtest_result payload.",
    input_schema=BACKTEST_RUNNER_INPUT_SCHEMA,
    handler=handle_backtest_runner,
)

