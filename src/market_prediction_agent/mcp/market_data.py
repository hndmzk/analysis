from __future__ import annotations

from typing import Any, cast

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import Settings, update_settings
from market_prediction_agent.data.adapters import OHLCVRequest
from market_prediction_agent.data.normalizer import normalize_ohlcv

from .base import MCPTool, clone_settings, dataframe_response, temporary_store


SUPPORTED_MARKET_SOURCES = [
    "dummy",
    "polygon",
    "alphavantage",
    "stooq",
    "yahoo_chart",
    "fred_market_proxy",
    "coingecko",
]

MARKET_DATA_FETCHER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "minLength": 1},
        "start_date": {"type": "string", "format": "date"},
        "end_date": {"type": "string", "format": "date"},
        "source": {
            "type": "string",
            "enum": SUPPORTED_MARKET_SOURCES,
        },
    },
    "required": ["ticker", "start_date", "end_date"],
    "additionalProperties": False,
}


def _adapter_metadata(adapter: object) -> dict[str, Any]:
    metadata = getattr(adapter, "last_fetch_metadata", {})
    if isinstance(metadata, dict):
        return cast(dict[str, Any], metadata)
    return {}


def handle_market_data_fetcher(params: dict[str, Any], settings: Settings) -> dict[str, Any]:
    ticker = str(params["ticker"]).upper()
    start_date = str(params["start_date"])
    end_date = str(params["end_date"])
    source = str(params["source"]) if params.get("source") is not None else None

    with temporary_store() as store:
        if source is None or source == "dummy":
            if source == "dummy":
                tool_settings = update_settings(settings, data={"source_mode": "dummy"})
            else:
                tool_settings = clone_settings(settings)
            agent = DataAgent(tool_settings, store)
            artifacts = agent.generate_or_fetch(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
            )
            return dataframe_response(
                artifacts.processed_ohlcv,
                metadata=cast(dict[str, Any], artifacts.ohlcv_metadata),
            )

        tool_settings = update_settings(settings, data={"source_mode": "live"})
        agent = DataAgent(tool_settings, store)
        adapter = agent._build_ohlcv_adapter(source)
        frame = normalize_ohlcv(
            adapter.fetch(
                OHLCVRequest(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        )
        return dataframe_response(
            frame,
            metadata={
                "requested_source": source,
                "used_source": source,
                "transport": _adapter_metadata(adapter),
            },
        )


MARKET_DATA_FETCHER = MCPTool(
    name="market_data_fetcher",
    description="Fetch normalized OHLCV market data for a ticker via the configured market-data adapters.",
    input_schema=MARKET_DATA_FETCHER_INPUT_SCHEMA,
    handler=handle_market_data_fetcher,
)
