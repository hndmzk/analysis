from __future__ import annotations

from market_prediction_agent.mcp.backtest_runner import BACKTEST_RUNNER
from market_prediction_agent.mcp.base import MCPTool
from market_prediction_agent.mcp.forecast_runner import FORECAST_RUNNER
from market_prediction_agent.mcp.macro_data import MACRO_DATA_FETCHER
from market_prediction_agent.mcp.market_data import MARKET_DATA_FETCHER
from market_prediction_agent.mcp.sec_reader import SEC_FILING_READER


MCP_TOOLS: dict[str, MCPTool] = {
    tool.name: tool
    for tool in [
        MARKET_DATA_FETCHER,
        SEC_FILING_READER,
        MACRO_DATA_FETCHER,
        FORECAST_RUNNER,
        BACKTEST_RUNNER,
    ]
}


__all__ = ["MCPTool", "MCP_TOOLS"]
