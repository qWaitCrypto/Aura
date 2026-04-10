from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack

import pytest

from aura.runtime.tools.runtime import ToolApprovalMode


@pytest.mark.asyncio
async def test_mcp_tool_listing(make_runtime, repo_root):
    rt = make_runtime(tools_enabled=False, approval_mode=ToolApprovalMode.TRUSTED)
    engine = rt.engine

    cfg_path = rt.project_root / ".aura" / "config" / "mcp.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "smoke": {
                        "enabled": True,
                        "command": sys.executable,
                        "args": ["-u", str(repo_root / "scripts" / "mcp_smoke_server.py")],
                        "env": {},
                        "cwd": None,
                        "timeout_s": 10,
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    async with AsyncExitStack() as stack:
        functions, specs = await engine._load_mcp_tooling(stack=stack)  # type: ignore[attr-defined]

        assert functions
        assert specs

        tool_names = sorted(functions.keys())
        # Prefix format is `mcp__{server}_{digest}__` + tool name (joined by MCPTools).
        assert any(n.startswith("mcp__smoke_") and n.endswith("echo") for n in tool_names)

        spec_names = [s.name for s in specs]
        assert any(n.startswith("mcp__smoke_") and n.endswith("echo") for n in spec_names)

        echo_name = next(n for n in tool_names if n.endswith("echo"))
        result = await functions[echo_name].entrypoint(text="hello")  # type: ignore[attr-defined]
        content = getattr(result, "content", None)
        assert content == "hello"

