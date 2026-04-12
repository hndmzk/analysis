from __future__ import annotations

from typing import Any, cast

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import Settings
from market_prediction_agent.data.adapters import DummyMacroAdapter, MacroAdapter, MacroRequest
from market_prediction_agent.data.normalizer import normalize_macro

from .base import MCPTool, clone_settings, dataframe_response, temporary_store


MACRO_DATA_FETCHER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "series_id": {"type": "string", "minLength": 1},
        "start_date": {"type": "string", "format": "date"},
        "end_date": {"type": "string", "format": "date"},
    },
    "required": ["series_id", "start_date", "end_date"],
    "additionalProperties": False,
}


def handle_macro_data_fetcher(params: dict[str, Any], settings: Settings) -> dict[str, Any]:
    series_id = str(params["series_id"])
    start_date = str(params["start_date"])
    end_date = str(params["end_date"])

    tool_settings = clone_settings(settings)
    with temporary_store() as store:
        agent = DataAgent(tool_settings, store)
        adapter: MacroAdapter
        if tool_settings.data.source_mode == "dummy":
            adapter = DummyMacroAdapter(seed=tool_settings.app.seed)
        else:
            adapter = agent._build_macro_adapter()
        frame = normalize_macro(
            adapter.fetch(
                MacroRequest(
                    series_ids=[series_id],
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        )

    return dataframe_response(
        frame,
        metadata={
            "series_id": series_id,
            "used_source": cast(str, getattr(adapter, "name", "unknown")),
            "transport": cast(dict[str, Any], getattr(adapter, "last_fetch_metadata", {})),
        },
    )


MACRO_DATA_FETCHER = MCPTool(
    name="macro_data_fetcher",
    description="Fetch normalized macroeconomic time series from the configured FRED adapter.",
    input_schema=MACRO_DATA_FETCHER_INPUT_SCHEMA,
    handler=handle_macro_data_fetcher,
)
