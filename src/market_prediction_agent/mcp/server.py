from __future__ import annotations

import json
import sys
from typing import Any

from market_prediction_agent import __version__
from market_prediction_agent.config import Settings, load_settings
from market_prediction_agent.mcp import MCP_TOOLS


def _result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _error_response(
    request_id: Any,
    *,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error,
    }


def _tool_list_payload() -> dict[str, Any]:
    tools = []
    for tool in sorted(MCP_TOOLS.values(), key=lambda item: item.name):
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "annotations": {
                    "readOnlyHint": tool.read_only,
                },
            }
        )
    return {"tools": tools}


def handle_request(request: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if not isinstance(request, dict):
        return _error_response(None, code=-32600, message="Invalid Request")

    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return _error_response(request_id, code=-32600, message="Invalid Request")

    if method == "initialize":
        return _result_response(
            request_id,
            {
                "serverInfo": {
                    "name": "market-prediction-agent",
                    "version": __version__,
                },
                "capabilities": {
                    "tools": {},
                },
            },
        )

    if method == "tools/list":
        return _result_response(request_id, _tool_list_payload())

    if method == "tools/call":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error_response(request_id, code=-32602, message="Invalid params")
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name:
            return _error_response(request_id, code=-32602, message="Invalid params")
        if not isinstance(arguments, dict):
            return _error_response(request_id, code=-32602, message="Invalid params")
        tool = MCP_TOOLS.get(tool_name)
        if tool is None:
            return _error_response(request_id, code=-32601, message=f"Unknown tool: {tool_name}")
        try:
            structured_content = tool.handler(arguments, settings)
        except Exception as exc:
            return _error_response(
                request_id,
                code=-32000,
                message=f"Tool '{tool_name}' failed: {exc}",
            )
        return _result_response(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(structured_content, ensure_ascii=False),
                    }
                ],
                "structuredContent": structured_content,
                "isError": False,
            },
        )

    return _error_response(request_id, code=-32601, message=f"Method not found: {method}")


def main() -> None:
    """MCP stdio server main loop."""
    settings = load_settings()
    for line in sys.stdin:
        payload = line.strip()
        if not payload:
            continue
        try:
            request = json.loads(payload)
        except json.JSONDecodeError as exc:
            response = _error_response(
                None,
                code=-32700,
                message="Parse error",
                data={"detail": str(exc)},
            )
        else:
            response = handle_request(request, settings)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
