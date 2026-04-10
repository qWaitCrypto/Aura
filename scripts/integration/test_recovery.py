from __future__ import annotations

import asyncio

import pytest

from aura.runtime.engine import ToolDecision
from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.types import ToolCall
from aura.runtime.llm.types import LLMResponse
from aura.runtime.protocol import Op, OpKind
from aura.runtime.run_snapshots import run_snapshot_path
from aura.runtime.tools.runtime import ToolApprovalMode


@pytest.mark.asyncio
async def test_cross_process_recovery(make_runtime, monkeypatch):
    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    # In this sandbox, filesystem calls from worker threads can hang; keep tool execution inline.
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread, raising=True)

    rt = make_runtime(tools_enabled=True, approval_mode=ToolApprovalMode.STRICT)
    engine = rt.engine

    # Register a tool that avoids filesystem I/O inside the worker thread.
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass(frozen=True, slots=True)
    class TestNoopTool:
        name: str = "test__noop"
        description: str = "No-op tool for integration tests."
        input_schema: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            }
        )

        def execute(self, *, args: dict[str, Any], project_root):
            return {"ok": True, "text": str(args.get("text") or "")}

    assert engine.tool_registry is not None  # type: ignore[attr-defined]
    engine.tool_registry.register(TestNoopTool())  # type: ignore[attr-defined]

    call_id = "call_recover"

    async def _stub_pause(self, *, request, profile, request_id, turn_id, **_kwargs):
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="pause for approval",
            tool_calls=[ToolCall(tool_call_id=call_id, name="test__noop", arguments={"text": "hello"})],
            usage=None,
            stop_reason="tool_use",
            request_id="stub",
        )

    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_pause, raising=True)

    request_id = new_id("req")
    op = Op(
        kind=OpKind.CHAT.value,
        payload={"text": "trigger approval"},
        session_id=rt.session_id,
        request_id=request_id,
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )
    paused = await engine.arun(op)
    assert paused.status == "needs_approval"

    snap_path = run_snapshot_path(project_root=rt.project_root, run_id=request_id)
    assert snap_path.exists()

    # "Cross-process": create a new engine instance bound to the same project/session.
    async def _stub_resume(self, *, request, profile, request_id, turn_id, **_kwargs):
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="done",
            tool_calls=[],
            usage=None,
            stop_reason="stop",
            request_id="stub",
        )

    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_resume, raising=True)
    rt2 = make_runtime(tools_enabled=True, approval_mode=ToolApprovalMode.STRICT, session_id=rt.session_id)

    completed = await rt2.engine.continue_run(
        run_id=request_id,
        decisions=[ToolDecision(tool_call_id=call_id, decision="approve")],
    )
    assert completed.status == "completed"
    assert not snap_path.exists()
