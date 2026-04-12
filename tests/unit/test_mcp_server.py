from __future__ import annotations

from typing import Any

import market_prediction_agent.mcp.server as mcp_server
from market_prediction_agent.config import load_settings
from market_prediction_agent.mcp import MCP_TOOLS
from market_prediction_agent.mcp.base import MCPTool


def _settings():
    return load_settings("config/default.yaml")


def test_handle_request_initialize_returns_server_info() -> None:
    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        },
        _settings(),
    )

    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "market-prediction-agent"
    assert response["result"]["capabilities"] == {"tools": {}}


def test_handle_request_tools_list_returns_registered_tools() -> None:
    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        },
        _settings(),
    )

    tool_names = {tool["name"] for tool in response["result"]["tools"]}
    assert tool_names == set(MCP_TOOLS)


def test_handle_request_tools_call_executes_handler(monkeypatch) -> None:
    def handler(params: dict[str, Any], settings) -> dict[str, Any]:
        del settings
        return {"echo": params, "ok": True}

    fake_tool = MCPTool(
        name="echo_tool",
        description="Echo input parameters.",
        input_schema={"type": "object"},
        handler=handler,
    )
    monkeypatch.setattr(mcp_server, "MCP_TOOLS", {"echo_tool": fake_tool})

    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "echo_tool",
                "arguments": {"value": 42},
            },
        },
        _settings(),
    )

    assert response["result"]["isError"] is False
    assert response["result"]["structuredContent"] == {"echo": {"value": 42}, "ok": True}


def test_handle_request_returns_error_for_unknown_method() -> None:
    response = mcp_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
        },
        _settings(),
    )

    assert response["error"]["code"] == -32601
    assert "Method not found" in response["error"]["message"]
