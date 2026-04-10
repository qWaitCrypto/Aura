from __future__ import annotations

import json
import os
import sys

import mcp.types as types


def _send_response(*, request_id: types.RequestId, result: types.ServerResult) -> None:
    msg = types.JSONRPCMessage(
        types.JSONRPCResponse(
            jsonrpc="2.0",
            id=request_id,
            result=result.model_dump(by_alias=True, exclude_none=True),
        )
    )
    sys.stdout.write(msg.model_dump_json(by_alias=True, exclude_none=True) + "\n")
    sys.stdout.flush()


def _send_error(*, request_id: types.RequestId, code: int, message: str) -> None:
    msg = types.JSONRPCMessage(
        types.JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=types.ErrorData(code=code, message=message, data=None),
        )
    )
    sys.stdout.write(msg.model_dump_json(by_alias=True, exclude_none=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    server_name = "aura-mcp-smoke"
    server_version = "0.0"
    debug = os.environ.get("AURA_MCP_SMOKE_DEBUG") == "1"

    saw_initialize = False
    saw_initialized = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = types.JSONRPCMessage.model_validate_json(line)
        except Exception:
            continue

        root = msg.root
        if isinstance(root, types.JSONRPCNotification):
            if debug:
                print(f"[smoke-server] notification {root.method}", file=sys.stderr, flush=True)
            if root.method == "notifications/initialized":
                saw_initialized = True
            continue

        if not isinstance(root, types.JSONRPCRequest):
            continue

        request_id = root.id
        method = root.method
        if debug:
            print(f"[smoke-server] request {request_id} {method}", file=sys.stderr, flush=True)

        if method == "initialize":
            try:
                params = types.InitializeRequestParams.model_validate(root.params or {})
            except Exception as e:
                _send_error(request_id=request_id, code=types.INVALID_REQUEST, message=str(e))
                continue

            saw_initialize = True
            protocol_version = params.protocolVersion
            _send_response(
                request_id=request_id,
                result=types.ServerResult(
                    types.InitializeResult(
                        protocolVersion=protocol_version,
                        capabilities=types.ServerCapabilities(
                            experimental={},
                            logging=None,
                            prompts=None,
                            resources=None,
                            tools=types.ToolsCapability(listChanged=False),
                            completions=None,
                        ),
                        serverInfo=types.Implementation(name=server_name, version=server_version),
                        instructions=None,
                        meta=None,
                    )
                ),
            )
            continue

        if not (saw_initialize and saw_initialized):
            _send_error(
                request_id=request_id,
                code=types.INVALID_REQUEST,
                message="Received request before initialization was complete.",
            )
            continue

        if method == "ping":
            _send_response(request_id=request_id, result=types.ServerResult(types.EmptyResult()))
            continue

        if method == "tools/list":
            _send_response(
                request_id=request_id,
                result=types.ServerResult(
                    types.ListToolsResult(
                        tools=[
                            types.Tool(
                                name="echo",
                                title=None,
                                description="Echo back the provided text.",
                                inputSchema={
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                    "additionalProperties": False,
                                },
                                outputSchema=None,
                                annotations=None,
                                meta=None,
                            )
                        ],
                        nextCursor=None,
                        meta=None,
                    )
                ),
            )
            continue

        if method == "tools/call":
            try:
                params = types.CallToolRequestParams.model_validate(root.params or {})
            except Exception as e:
                _send_error(request_id=request_id, code=types.INVALID_REQUEST, message=str(e))
                continue

            if params.name != "echo":
                _send_error(request_id=request_id, code=types.METHOD_NOT_FOUND, message=f"Unknown tool: {params.name}")
                continue

            text = ""
            if isinstance(params.arguments, dict) and isinstance(params.arguments.get("text"), str):
                text = params.arguments["text"]

            _send_response(
                request_id=request_id,
                result=types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(type="text", text=text, annotations=None, meta=None)],
                        structuredContent=None,
                        isError=False,
                        meta=None,
                    )
                ),
            )
            continue

        _send_error(request_id=request_id, code=types.METHOD_NOT_FOUND, message=f"Unknown method: {method}")


if __name__ == "__main__":
    main()
