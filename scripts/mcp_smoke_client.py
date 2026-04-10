from __future__ import annotations

import asyncio
import json


async def main() -> None:
    from agno.tools.mcp.mcp import MCPTools

    async with MCPTools(command="python -u scripts/mcp_smoke_server.py", transport="stdio", timeout_seconds=10) as mcp:
        await mcp.initialize()
        tools = mcp.get_async_functions()
        print("tools:", list(tools.keys()))
        echo = tools["echo"]
        result = await echo.entrypoint(text="hello")
        # ToolResult is a pydantic model.
        print("result:", json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
